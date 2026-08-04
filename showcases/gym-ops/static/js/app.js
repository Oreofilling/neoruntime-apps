"use strict";
/* GymOps workbench client.
 *
 * HD mode  : platform H.264 -> MSE via a proven minimal fMP4 remuxer (ported
 *            verbatim from model-showcase templates/index.html __H264Muxer +
 *            createHdPlayer), with a client-side COCO-17 skeleton overlay on
 *            #pose-canvas (landmarks are normalized [0,1]).
 * Sync mode: platform H.264 video stays on MSE while /ws/sync-preview carries
 *            only pose metadata for the matching inferred frames.
 * SSE /api/events drives persons/zones/equipment/alerts stats and the pose
 * overlay. /api/config supplies zone names + capacities + polygons + exercise.
 * requestAnimationFrame loop does temporal interpolation between SSE
 * snapshots for smooth 60fps skeleton tracking.
 */

// COCO-17 skeleton edges — must match config.SKELETON_EDGES.
var SKELETON = [
  [0, 1], [1, 3], [0, 2], [2, 4], [5, 6],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [11, 12], [5, 11], [6, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

var $ = function (id) { return document.getElementById(id); };

function detectAppBase() {
  var marker = "/static/js/app.js";
  var scripts = document.getElementsByTagName("script");
  for (var i = scripts.length - 1; i >= 0; i--) {
    var src = scripts[i].getAttribute("src") || "";
    var idx = src.indexOf(marker);
    if (idx >= 0) return src.slice(0, idx);
  }
  var path = location.pathname.replace(/\/+$/, "");
  var m = path.match(/^(.*\/apps\/[^/]+)/);
  return m ? m[1] : "";
}

var APP_BASE = detectAppBase();
var PTS_WRAP = 4294967296;
var PTS_HALF_WRAP = 2147483648;
var SYNC_VIDEO_DELAY_SECONDS = 2.4;

function appUrl(path) {
  if (!path) return APP_BASE || "/";
  if (path.charAt(0) !== "/") path = "/" + path;
  return APP_BASE + path;
}

function wrapPts90(v) {
  v = v % PTS_WRAP;
  return v < 0 ? v + PTS_WRAP : v;
}

function diffPts90(a, b) {
  var d = a - b;
  if (d > PTS_HALF_WRAP) d -= PTS_WRAP;
  else if (d < -PTS_HALF_WRAP) d += PTS_WRAP;
  return d;
}

var state = {
  mode: "hd",
  zones: [],
  exercise: "squat",
  zoneName: function (id) { return id; },
  latest: null,
  prevSnapshot: null,   // for temporal interpolation (velocity source)
  lastSnapshot: null,    // most recent SSE snapshot
  npuEma: null,          // smoothed NPU device_utilization (raw sample flickers 0↔2%)
  videoMode: false,      // uploaded-video mode bakes overlay server-side
  hd: null,
  sync: {
    active: false,
    ws: null,
    retryTimer: null,
    decoder: new TextDecoder("utf-8"),
    lastSeq: 0,
    poseBuffer: [],
    maxPoseSamples: 120,
    videoDelaySeconds: SYNC_VIDEO_DELAY_SECONDS,
  },
};

// ── minimal H264→fMP4 muxer (ported verbatim from model-showcase) ──
// Produces init (ftyp+moov) + media (moof+mdat) segments for MSE. The platform
// broadcasts AVCC-format NALs (4-byte length prefix), so each access unit is
// fed as one mdat sample — no Annex-B conversion needed.
(function () {
  function w32(a, o, v) { a[o] = (v >>> 24) & 255; a[o + 1] = (v >>> 16) & 255; a[o + 2] = (v >>> 8) & 255; a[o + 3] = v & 255; }
  function w16(a, o, v) { a[o] = (v >>> 8) & 255; a[o + 1] = v & 255; }
  function wstr(a, o, s) { for (var i = 0; i < s.length; i++) a[o + i] = s.charCodeAt(i); }
  function box(type) { var n = 8; for (var i = 1; i < arguments.length; i++) n += arguments[i].length; var r = new Uint8Array(n); w32(r, 0, n); wstr(r, 4, type); var o = 8; for (var j = 1; j < arguments.length; j++) { r.set(arguments[j], o); o += arguments[j].length; } return r; }
  function fullbox(t, ver, flags, c) { var fc = new Uint8Array(4 + c.length); fc[0] = ver; fc[1] = (flags >>> 16) & 255; fc[2] = (flags >>> 8) & 255; fc[3] = flags & 255; fc.set(c, 4); return box(t, fc); }
  function cat() { var n = 0; for (var i = 0; i < arguments.length; i++) n += arguments[i].length; var r = new Uint8Array(n); var o = 0; for (var k = 0; k < arguments.length; k++) { r.set(arguments[k], o); o += arguments[k].length; } return r; }
  function avcC(sps, pps) { var c = new Uint8Array(11 + sps.length + pps.length), i = 0; c[i++] = 1; c[i++] = sps[1]; c[i++] = sps[2]; c[i++] = sps[3]; c[i++] = 0xff; c[i++] = 0xe1; c[i++] = (sps.length >> 8) & 255; c[i++] = sps.length & 255; c.set(sps, i); i += sps.length; c[i++] = 1; c[i++] = (pps.length >> 8) & 255; c[i++] = pps.length & 255; c.set(pps, i); return box('avcC', c); }
  function avc1(sps, pps, w, h) { var a = avcC(sps, pps); var c = new Uint8Array(78 + a.length); w16(c, 6, 1); w16(c, 24, w); w16(c, 26, h); w32(c, 28, 0x00480000); w32(c, 32, 0x00480000); w16(c, 40, 1); w16(c, 74, 0x0018); w16(c, 76, 0xffff); c.set(a, 78); return box('avc1', c); }
  function stbl(sps, pps, w, h) { return box('stbl', fullbox('stsd', 0, 0, cat(new Uint8Array([0, 0, 0, 1]), avc1(sps, pps, w, h))), fullbox('stts', 0, 0, new Uint8Array([0, 0, 0, 0])), fullbox('stsc', 0, 0, new Uint8Array([0, 0, 0, 0])), fullbox('stsz', 0, 0, new Uint8Array(12)), fullbox('stco', 0, 0, new Uint8Array([0, 0, 0, 0]))); }
  function minf(sps, pps, w, h) { return box('minf', fullbox('vmhd', 0, 1, new Uint8Array(8)), box('dinf', fullbox('dref', 0, 0, cat(new Uint8Array([0, 0, 0, 1]), fullbox('url ', 0, 1, new Uint8Array(0))))), stbl(sps, pps, w, h)); }
  function mdia(sps, pps, w, h) { var mdhd = new Uint8Array(20); w32(mdhd, 8, 90000); w16(mdhd, 16, 0x55c4); var hdlr = new Uint8Array(21); wstr(hdlr, 4, 'vide'); return box('mdia', fullbox('mdhd', 0, 0, mdhd), fullbox('hdlr', 0, 0, hdlr), minf(sps, pps, w, h)); }
  function tkhd(w, h) { var c = new Uint8Array(80); w32(c, 8, 1); w32(c, 36, 0x00010000); w32(c, 52, 0x00010000); w32(c, 68, 0x40000000); w32(c, 72, w << 16); w32(c, 76, h << 16); return fullbox('tkhd', 0, 3, c); }
  function trak(sps, pps, w, h) { return box('trak', tkhd(w, h), mdia(sps, pps, w, h)); }
  function mvhd() { var c = new Uint8Array(96); w32(c, 8, 90000); w32(c, 16, 0x00010000); w16(c, 20, 0x0100); w32(c, 32, 0x00010000); w32(c, 48, 0x00010000); w32(c, 64, 0x40000000); w32(c, 92, 2); return fullbox('mvhd', 0, 0, c); }
  function mvex() { var c = new Uint8Array(20); w32(c, 0, 1); w32(c, 4, 1); return box('mvex', fullbox('trex', 0, 0, c)); }
  function moov(sps, pps, w, h) { return box('moov', mvhd(), trak(sps, pps, w, h), mvex()); }
  function ftyp() { var c = new Uint8Array(20); wstr(c, 0, 'isom'); w32(c, 4, 512); wstr(c, 8, 'isom'); wstr(c, 12, 'iso2'); wstr(c, 16, 'avc1'); return box('ftyp', c); }
  function ssiz_safe(n) { return n > 0 ? n : 0; }
  function moof(seq, dur, kf, bdt, ssz, cto) { var mfhdC = new Uint8Array(4); w32(mfhdC, 0, seq); var tfhd = fullbox('tfhd', 0, 0x20000, new Uint8Array([0, 0, 0, 1])); var tfdtC = new Uint8Array(4); w32(tfdtC, 0, bdt); var tfdt = fullbox('tfdt', 0, 0, tfdtC); var trunc = new Uint8Array(24); w32(trunc, 0, 1); w32(trunc, 4, 108); w32(trunc, 8, dur); w32(trunc, 12, ssiz_safe(ssz)); w32(trunc, 16, kf ? 0x02000000 : 0x01010000); w32(trunc, 20, (cto < 0 ? (cto >>> 0) : cto)); var trun = fullbox('trun', 1, 0xF01, trunc); return box('moof', fullbox('mfhd', 0, 0, mfhdC), box('traf', tfhd, tfdt, trun)); }
  window.__H264Muxer = function () { this.seq = 0; this.dur = 3000; this.createInit = function (sps, pps, w, h) { return cat(ftyp(), moov(sps, pps, w || 1920, h || 1080)); }; this.createSeg = function (nal, kf, dt, cto) { this.seq++; return cat(moof(this.seq, this.dur, kf, dt, nal.length, cto || 0), box('mdat', nal)); }; this.setFps = function (f) { var nd = Math.round(90000 / f); if (Math.abs(nd - this.dur) > 100) this.dur = nd; }; };
})();

// ── compact MSE player (ported verbatim from model-showcase) ──
// Wire format per WS message: 18-byte FrameHeaderV2 (magic 0xFFD8FF00, frameType,
// flags[bit0=hasDts], PTS@90kHz, DTS@90kHz, reserved) + AVCC NAL units.
// On connect the platform first sends cached SPS/PPS, each in its own frame.
function createHdPlayer(video, opts) {
  opts = opts || {};
  var liveDelay = opts.liveDelaySeconds != null ? opts.liveDelaySeconds : 0.08;
  var ws = null, ms = null, sb = null, muxer = null;
  var sps = null, pps = null, initialized = false, hasFirstI = false;
  var spsW = 1920, spsH = 1080;
  var firstPts, lastPts, fps = 30, nextTs = 0;
  var queue = [], url = '', active = false, destroyed = false;
  var mseOpen = false, retryTimer = null, mseErrs = 0, restartTimer = null;

  function eqArr(a, b) { if (!a || !b || a.length !== b.length) return false; for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }
  function codecFromSps(s) { function h2(n) { return ('0' + n.toString(16)).slice(-2); } return 'avc1.' + h2(s[1]) + h2(s[2]) + h2(s[3]); }
  function parseSpsRes(spsNal) {
    try {
      if (spsNal.length < 5) return { w: 1920, h: 1080 };
      var bp = 0, tot = spsNal.length * 8;
      function rb() { if (bp >= tot) return 0; var by = bp >> 3, bi = 7 - (bp & 7); bp++; return (spsNal[by] >> bi) & 1; }
      function rbs(n) { var v = 0; for (var i = 0; i < n; i++) v = (v << 1) | rb(); return v; }
      function rue() { var z = 0; while (rb() === 0 && z < 32) z++; return z > 0 ? ((1 << z) - 1 + rbs(z)) : 0; }
      bp = 32; var prof = spsNal[1]; rue();
      if ([100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134].indexOf(prof) >= 0) {
        var cf = rue(); if (cf === 3) rb(); rue(); rue(); rb();
        if (rb()) { var lim = cf !== 3 ? 8 : 12; for (var i = 0; i < lim; i++) { if (rb()) { var sz = i < 6 ? 16 : 64, ls = 8, ns = 8; for (var j = 0; j < sz; j++) { if (ns !== 0) { var cd = rue(); var dl = (cd & 1) ? (cd + 1) >> 1 : -(cd >> 1); ns = (ls + dl + 256) % 256; } ls = ns === 0 ? ls : ns; } } } }
      }
      rue(); var poct = rue();
      if (poct === 0) { rue(); } else if (poct === 1) { rb(); rue(); rue(); var cnt = rue(); for (var i = 0; i < cnt; i++) rue(); }
      rue(); rb();
      var pw = rue(), ph = rue(), fmo = rb();
      return { w: (pw + 1) * 16, h: (2 - fmo) * (ph + 1) * 16 };
    } catch (e) { return { w: 1920, h: 1080 }; }
  }

  function append(buf) { if (!sb || destroyed) return; if (sb.updating) { queue.push(buf); return; } try { sb.appendBuffer(buf); } catch (e) { onMseError(); } }
  function onUpd() {
    if (destroyed || !sb) return;
    if (queue.length) { var b = queue.shift(); try { sb.appendBuffer(b); } catch (e) { onMseError(); } return; }
    try {
      if (video.buffered.length) {
        var end = video.buffered.end(video.buffered.length - 1);
        var lead = end - video.currentTime;
        var target = Math.max(video.buffered.start(0), end - liveDelay);
        if ((lead > liveDelay + 0.35 || lead < Math.max(0.02, liveDelay - 0.35))
            && Math.abs(video.currentTime - target) > 0.05) {
          video.currentTime = target;
        }
        if (sb.buffered.length && video.currentTime > sb.buffered.start(0) + Math.max(4, liveDelay + 3)) { sb.remove(0, video.currentTime - Math.max(3, liveDelay + 2)); return; }
      }
    } catch (e) {}
  }
  function onMseError() { if (destroyed || !active) return; mseErrs++; if (mseErrs > 5) return; scheduleRestart(); }
  function scheduleRestart() { if (restartTimer || destroyed || !active) return; restartTimer = setTimeout(function () { restartTimer = null; if (active && !destroyed) { teardownMse(); openMse(); if (url) connectWs(url); } }, 800); }
  function resetDecoder() { initialized = false; muxer = null; hasFirstI = false; nextTs = 0; firstPts = undefined; lastPts = undefined; pps = null; queue = []; }
  function teardownMse() {
    try { if (sb) { sb.removeEventListener('updateend', onUpd); sb.removeEventListener('error', onMseError); if (ms && ms.readyState === 'open') { try { sb.abort(); } catch (e) {} } } } catch (e) {}
    sb = null; resetDecoder(); sps = null; mseOpen = false;
    try { if (ms) { ms.removeEventListener('sourceopen', onSourceOpen); if (ms.readyState === 'open') { try { ms.endOfStream(); } catch (e) {} } } } catch (e) {}
    try { if (video.src) URL.revokeObjectURL(video.src); } catch (e) {}
    video.src = ''; ms = null;
  }
  function openMse() { if (!window.MediaSource) return false; ms = new MediaSource(); video.src = URL.createObjectURL(ms); ms.addEventListener('sourceopen', onSourceOpen); return true; }
  function onSourceOpen() { mseOpen = true; try { video.play(); } catch (e) {} }

  function feedFrame(nal, kf, pts) {
    if (firstPts === undefined) firstPts = pts;
    if (lastPts !== undefined && pts > lastPts) { var d = pts - lastPts; if (d >= 1500 && d <= 18000) { var mf = Math.round(90000 / d); if (mf !== fps && mf >= 5 && mf <= 60) { fps = mf; if (muxer) muxer.setFps(mf); } } }
    lastPts = pts;
    var dt = nextTs; nextTs += Math.round(90000 / fps);
    var relPts = (pts - firstPts) | 0;
    var cto = relPts - dt;
    append(muxer.createSeg(nal, kf, dt, cto).buffer);
  }

  function handleMsg(buf) {
    if (destroyed) return;
    var dv = new DataView(buf), data, kf = false, pts = 0;
    if (buf.byteLength >= 10 && dv.getUint32(0, false) === 0xFFD8FF00) {
      var flags = dv.getUint8(5), hasDts = (flags & 0x01) !== 0;
      pts = dv.getUint32(6, false);
      data = (buf.byteLength >= 18 && hasDts) ? new Uint8Array(buf, 18) : new Uint8Array(buf, 10);
    } else { data = new Uint8Array(buf); }

    var off = 0, chunks = [], sawVCL = false;
    while (off + 4 <= data.length) {
      var len = ((data[off] << 24) | (data[off + 1] << 16) | (data[off + 2] << 8) | data[off + 3]) >>> 0;
      if (len === 0 || len > data.length - off - 4) break;
      var nal = data.subarray(off + 4, off + 4 + len);
      var nt = nal.length ? (nal[0] & 0x1f) : 0;
      if (nt === 7) {
        if (!eqArr(sps, nal)) { if (initialized) { scheduleRestart(); off += 4 + len; continue; } resetDecoder(); }
        sps = nal.slice(); var r = parseSpsRes(sps); spsW = r.w; spsH = r.h;
      } else if (nt === 8) {
        pps = nal.slice();
      } else if (nt !== 6 && nt !== 9) {
        kf = kf || (nt === 5);
        if (initialized && muxer) {
          if (!hasFirstI) { if (kf) { hasFirstI = true; } else { off += 4 + len; continue; } }
          var av = new Uint8Array(4 + nal.length); av[0] = (nal.length >>> 24) & 255; av[1] = (nal.length >>> 16) & 255; av[2] = (nal.length >>> 8) & 255; av[3] = nal.length & 255; av.set(nal, 4);
          chunks.push(av); sawVCL = true;
        }
      }
      off += 4 + len;
    }

    if (!initialized && sps && pps && mseOpen) {
      try {
        muxer = new window.__H264Muxer(); muxer.setFps(fps);
        var codec = codecFromSps(sps);
        if (!MediaSource.isTypeSupported('video/mp4; codecs="' + codec + '"')) { onMseError(); return; }
        sb = ms.addSourceBuffer('video/mp4; codecs="' + codec + '"');
        sb.mode = 'segments';
        sb.addEventListener('updateend', onUpd);
        sb.addEventListener('error', onMseError);
        initialized = true; mseErrs = 0;
        append(muxer.createInit(sps, pps, spsW, spsH).buffer);
      } catch (e) { onMseError(); }
    }

    if (sawVCL && chunks.length && muxer) {
      var tot = 0; for (var i = 0; i < chunks.length; i++) tot += chunks[i].length;
      var merged = new Uint8Array(tot), o = 0; for (var i = 0; i < chunks.length; i++) { merged.set(chunks[i], o); o += chunks[i].length; }
      feedFrame(merged, kf, pts);
    }
  }

  function connectWs(u) {
    try { ws = new WebSocket(u); } catch (e) { scheduleRestart(); return; }
    ws.binaryType = 'arraybuffer';
    ws.onmessage = function (ev) { handleMsg(ev.data); };
    ws.onopen = function () { mseErrs = 0; };
    ws.onclose = function () { if (active && !destroyed) { retryTimer = setTimeout(function () { retryTimer = null; if (active && !destroyed && url) connectWs(url); }, 1200); } };
    ws.onerror = function () { try { ws.close(); } catch (e) {} };
  }

  return {
    start: function (u) { url = u; active = true; destroyed = false; mseErrs = 0; resetDecoder(); sps = null; pps = null; openMse(); connectWs(u); },
    stop: function () {
      active = false;
      if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
      if (restartTimer) { clearTimeout(restartTimer); restartTimer = null; }
      if (ws) { ws.onclose = null; ws.onerror = null; ws.onmessage = null; try { ws.close(); } catch (e) {} ws = null; }
      teardownMse();
    },
    getSourcePts90: function () {
      if (firstPts === undefined || !isFinite(video.currentTime)) return null;
      return wrapPts90(firstPts + Math.round(video.currentTime * 90000));
    },
    getBufferedEndPts90: function () {
      if (firstPts === undefined || !video.buffered.length) return null;
      var end = video.buffered.end(video.buffered.length - 1);
      return wrapPts90(firstPts + Math.round(end * 90000));
    },
    setLiveDelaySeconds: function (seconds) {
      if (isFinite(seconds)) liveDelay = Math.max(0.08, Math.min(4.0, seconds));
    },
    getFps: function () { return fps; }   // measured stream fps from PTS deltas
  };
}

// ── config ──
function loadConfig() {
  fetch(appUrl("/api/config")).then(function (r) { return r.json(); }).then(function (c) {
    state.zones = c.zones || [];
    state.exercise = c.exercise || "squat";
    state.zoneName = function (id) {
      for (var i = 0; i < state.zones.length; i++) if (state.zones[i].id === id) return state.zones[i].name;
      return id;
    };
    renderZones({ counts: {}, crowded: [] });
  }).catch(function () { /* keep defaults */ });
}

// ── SSE ──
function connectSSE() {
  var es = new EventSource(appUrl("/api/events"));
  var conn = $("conn-state");
  es.onopen = function () { conn.textContent = "SSE connected"; conn.className = "live"; };
  es.onerror = function () { conn.textContent = "SSE disconnected, reconnecting..."; conn.className = "lost"; };
  es.onmessage = function (ev) {
    if (state.mode === "sync") return;
    try { onSnapshot(JSON.parse(ev.data)); } catch (e) {}
  };
}

// ── synchronized H.264 preview + pose WebSocket metadata ──
function wsUrl(path) {
  var proto = location.protocol === "https:" ? "wss:" : "ws:";
  return proto + "//" + location.host + appUrl(path);
}

function scheduleSyncReconnect() {
  if (!state.sync.active || state.sync.retryTimer) return;
  var empty = $("video-empty");
  empty.textContent = "Sync preview reconnecting...";
  empty.classList.remove("hide");
  state.sync.retryTimer = setTimeout(function () {
    state.sync.retryTimer = null;
    if (state.sync.active) connectSyncWs();
  }, 1000);
}

function handleSyncPreview(data) {
  var text = "";
  if (typeof data === "string") {
    text = data;
  } else if (data instanceof ArrayBuffer) {
    text = state.sync.decoder.decode(new Uint8Array(data));
  } else {
    return;
  }
  var meta;
  try {
    meta = JSON.parse(text);
  } catch (e) {
    return;
  }
  var snap = meta.snapshot || {};
  var seq = meta.seq || 0;
  state.sync.lastSeq = seq;
  snap.sync_seq = seq;
  snap.frame_sequence = meta.frame_sequence;
  snap.frame_timestamp_ns = meta.frame_timestamp_ns;
  snap.video_pts90 = meta.video_pts90;
  onSnapshot(snap);
  pushSyncPose(snap);
  updateSyncVideoDelay(meta);

  var cv = $("pose-canvas");
  if (meta.width && meta.height) {
    if (cv.width !== meta.width || cv.height !== meta.height) {
      cv.width = meta.width;
      cv.height = meta.height;
    }
  }
  $("video-empty").classList.add("hide");
}

function pushSyncPose(snap) {
  if (snap.video_pts90 == null) return;
  state.sync.poseBuffer.push(snap);
  if (state.sync.poseBuffer.length > state.sync.maxPoseSamples) {
    state.sync.poseBuffer.splice(
      0, state.sync.poseBuffer.length - state.sync.maxPoseSamples);
  }
}

function updateSyncVideoDelay(meta) {
  if (!state.hd || !state.hd.getBufferedEndPts90 || !state.hd.setLiveDelaySeconds) return;
  if (meta.video_pts90 == null) return;
  var endPts90 = state.hd.getBufferedEndPts90();
  if (endPts90 == null) return;
  var diffSeconds = diffPts90(endPts90, meta.video_pts90) / 90000;
  if (!isFinite(diffSeconds) || diffSeconds < 0.05 || diffSeconds > 4.0) return;
  var target = Math.max(0.3, Math.min(4.0, diffSeconds + 0.15));
  state.sync.videoDelaySeconds = state.sync.videoDelaySeconds * 0.85 + target * 0.15;
  state.hd.setLiveDelaySeconds(state.sync.videoDelaySeconds);
}

function connectSyncWs() {
  if (!state.sync.active) return;
  var empty = $("video-empty");
  empty.textContent = "Connecting sync preview...";
  empty.classList.remove("hide");
  var ws;
  try {
    ws = new WebSocket(wsUrl("/ws/sync-preview"));
  } catch (e) {
    scheduleSyncReconnect();
    return;
  }
  state.sync.ws = ws;
  ws.binaryType = "arraybuffer";
  ws.onopen = function () {
    $("video-hint").textContent = "H.264 video + pose metadata WS is live";
  };
  ws.onmessage = function (ev) {
    handleSyncPreview(ev.data);
  };
  ws.onerror = function () {
    try { ws.close(); } catch (e) {}
  };
  ws.onclose = function () {
    if (state.sync.ws === ws) state.sync.ws = null;
    scheduleSyncReconnect();
  };
}

function enterSync() {
  if (state.sync.active) return;
  state.sync.active = true;
  state.sync.poseBuffer = [];
  state.sync.videoDelaySeconds = SYNC_VIDEO_DELAY_SECONDS;
  startHdVideo("Starting H.264 video + pose metadata WS...", {
    liveDelaySeconds: SYNC_VIDEO_DELAY_SECONDS
  });
  connectSyncWs();
}

function exitSync() {
  state.sync.active = false;
  if (state.sync.retryTimer) {
    clearTimeout(state.sync.retryTimer);
    state.sync.retryTimer = null;
  }
  if (state.sync.ws) {
    var ws = state.sync.ws;
    state.sync.ws = null;
    ws.onclose = null;
    ws.onerror = null;
    ws.onmessage = null;
    try { ws.close(); } catch (e) {}
  }
}

function npuPct(du) {
  if (du == null) return "—";
  var v = du <= 1 ? du * 100 : du;   // backend may store 0-1 fraction
  return (v < 10 ? v.toFixed(1) : Math.round(v)) + "%";
}

function onSnapshot(s) {
  // Server ts is wall-clock time; animation uses performance.now().
  // Stamp the local receive time so interpolation lives on one clock.
  s.client_ts = performance.now() / 1000;
  state.prevSnapshot = state.lastSnapshot;
  state.lastSnapshot = s;
  state.latest = s;
  state.videoMode = !!s.video_mode;
  if (state.mode === "mjpeg") {
    var cv = $("pose-canvas");
    cv.style.display = state.videoMode ? "none" : "";
    if (state.videoMode) {
      cv.getContext("2d").clearRect(0, 0, cv.width || 1, cv.height || 1);
    }
  }
  $("stat-total").textContent = (s.zones && s.zones.total) || 0;
  var reps = 0; (s.persons || []).forEach(function (p) { reps += (p.snap && p.snap.reps) || 0; });
  $("stat-reps").textContent = reps;
  var vf = state.hd && state.hd.getFps ? state.hd.getFps() : null;
  var pf = s.pose_fps != null ? s.pose_fps : null;
  $("stat-fps").textContent =
    (vf != null && vf > 0 ? Math.round(vf) : "—") + "/" + (pf != null ? pf : "—");
  // Smooth the instantaneous device_utilization: a single raw sample flickers
  // between 0% and ~2% at preview rate, which reads as "broken". The EMA
  // stabilizes it; npuPct() still maps the 0-1 fraction. Low % here is
  // expected — pose inference does not saturate the NPU (see HW infer ms).
  var du = s.device_utilization;
  if (du != null) state.npuEma = state.npuEma == null ? du : state.npuEma * 0.8 + du * 0.2;
  $("stat-npu").textContent = npuPct(state.npuEma);

  if ((s.persons || []).length) state.lastPersonsTs = Date.now();
  renderPersons(s.persons || []);
  renderZones(s.zones || { counts: {}, crowded: [] });
  renderEquipment((s.zones && s.zones.equipment) || []);
  renderAlerts(s.alerts || []);
  // overlay is now drawn by animateOverlay() rAF loop (no direct drawPose here)

  var hb = $("badge-health");
  if (s.hw_infer_time_us != null) {
    var ms = Math.round(s.hw_infer_time_us / 1000);
    hb.textContent = ms + "ms";
    hb.className = "badge " + (ms > 100 ? "warn" : "ok");
  }
}

// ── render: persons ──
function renderPersons(persons) {
  var el = $("persons");
  if (!persons.length) {
    var lst = state.latest || {};
    var det = lst.detect_persons || 0;
    var msg = det > 0 ? "No pose tracks · detection-only: " + det : "No detections";
    var age = state.lastPersonsTs ? Math.max(0, Math.round((Date.now() - state.lastPersonsTs) / 1000)) : null;
    if (age != null) msg += " · last pose " + age + "s ago";
    el.innerHTML = '<p class="empty">' + msg + "</p>";
    return;
  }
  el.innerHTML = persons.map(function (p) {
    var sn = p.snap || {};
    var ph = sn.phase || "unknown";
    var flags = (sn.quality_flags || []).map(function (f) { return '<span class="flag">' + f + "</span>"; }).join("");
    var ang = sn.angle != null ? sn.angle + "°" : "—";
    return '<div class="person-card">' +
      '<div class="pc-head"><span class="pc-id">' + p.id + " - " + (sn.exercise || "") + "</span>" +
      '<span class="pc-reps">' + (sn.reps || 0) + "</span></div>" +
      '<div class="pc-meta">' +
      "<span>phase <b class=\"pc-phase " + ph + '">' + ph + "</b></span>" +
      "<span>angle <b>" + ang + "</b></span>" +
      "<span>side <b>" + (sn.side || "—") + "</b></span>" +
      "<span>bad <b>" + (sn.bad_reps || 0) + "</b></span>" +
      "</div>" + (flags ? '<div class="pc-flags">' + flags + "</div>" : "") + "</div>";
  }).join("");
}

// ── render: zones ──
function renderZones(z) {
  var el = $("zones");
  var counts = z.counts || {};
  var crowd = {}; (z.crowded || []).forEach(function (id) { crowd[id] = true; });
  var ids = Object.keys(counts);
  if (!ids.length) ids = state.zones.map(function (zz) { return zz.id; });
  if (!ids.length) { el.innerHTML = '<p class="empty">No zones configured</p>'; return; }
  el.innerHTML = ids.map(function (id) {
    var cap = 0; for (var i = 0; i < state.zones.length; i++) if (state.zones[i].id === id) { cap = state.zones[i].capacity || 0; break; }
    var n = counts[id] || 0;
    var isCrowd = !!crowd[id];
    var cls = isCrowd ? "zone-row crowd" : "zone-row";
    var bar = cap
      ? '<div class="z-bar"><i style="width:' + Math.min(100, Math.round(n / cap * 100)) + '%"></i></div><span class="z-count"><b>' + n + "</b>/" + cap + "</span>"
      : '<span class="z-count"><b>' + n + "</b></span>";
    return '<div class="' + cls + '"><span class="z-name">' + state.zoneName(id) + "</span>" + bar + "</div>";
  }).join("");
}

// ── render: equipment ──
function renderEquipment(eq) {
  var el = $("equip");
  if (!eq.length) { el.innerHTML = '<p class="empty">No equipment configured</p>'; return; }
  el.innerHTML = eq.map(function (e) {
    var cls = e.long ? "eq-card long" : (e.occupied ? "eq-card occupied" : "eq-card");
    var st = e.occupied ? "occupied " + e.duration_s + "s" + (e.long ? " - overtime" : "") : "idle";
    return '<div class="' + cls + '"><div class="eq-name">' + e.name + '</div><div class="eq-state">' + st + "</div></div>";
  }).join("");
}

// ── render: alerts ──
function renderAlerts(alerts) {
  var el = $("alerts");
  if (!alerts.length) { el.innerHTML = '<li class="empty">No alerts</li>'; return; }
  el.innerHTML = alerts.slice(-50).reverse().map(function (a) {
    var t = a.type || "alert";
    var who = a.tracker_id || a.zone_id || a.equipment_id || "";
    var label = (t + " " + who).trim();
    var ts = new Date((a.ts || 0) * 1000).toLocaleTimeString();
    return '<li><span class="a-tag ' + t + '">' + t + "</span><span>" + label + '</span><span class="a-time">' + ts + "</span></li>";
  }).join("");
}

// ── pose canvas overlay (both HD and MJPEG modes) ──
// Interpolation constants — mirror backend _MAX_PREDICT_HORIZON / _MAX_EXTRAP_DISPL
var _MAX_PREDICT_HORIZON = 0.40;   // extrapolate up to 400ms (generous for frontend)
var _MAX_EXTRAP_DISPL = 0.08;      // max per-keypoint displacement (normalized)
var _MIN_INFER_DT = 0.01;          // min infer interval for velocity (div-by-zero guard)

function interpPersons(prev, last, now) {
  /* Linearly extrapolate keypoints by velocity so the skeleton tracks
     the person between inference frames. Mirrors backend _interp_persons. */
  if (!last || !last.persons) return [];
  var lastPersons = last.persons;
  if (!prev || !prev.persons) return lastPersons;
  var tsPrev = prev.client_ts != null ? prev.client_ts : prev.ts;
  var tsLast = last.client_ts != null ? last.client_ts : last.ts;
  var dtInfer = tsLast - tsPrev;
  var tpAge = now - tsLast;
  if (tpAge > _MAX_PREDICT_HORIZON || dtInfer < _MIN_INFER_DT) return lastPersons;
  // index prev persons by id for O(1) pairing
  var prevById = {};
  prev.persons.forEach(function (p) { prevById[p.id] = p; });
  return lastPersons.map(function (p) {
    var pp = prevById[p.id];
    if (!pp || !pp.landmarks) return p;
    var lmLast = p.landmarks || [];
    var lmPrev = pp.landmarks || [];
    var interp = lmLast.map(function (pt, i) {
      var ptPrev = lmPrev[i];
      if (!pt || !ptPrev) return pt;
      var vx = (pt.x - ptPrev.x) / dtInfer;
      var vy = (pt.y - ptPrev.y) / dtInfer;
      var xi = pt.x + vx * tpAge;
      var yi = pt.y + vy * tpAge;
      // clamp displacement to bound overshoot
      var dx = xi - pt.x;
      if (dx > _MAX_EXTRAP_DISPL) xi = pt.x + _MAX_EXTRAP_DISPL;
      else if (dx < -_MAX_EXTRAP_DISPL) xi = pt.x - _MAX_EXTRAP_DISPL;
      var dy = yi - pt.y;
      if (dy > _MAX_EXTRAP_DISPL) yi = pt.y + _MAX_EXTRAP_DISPL;
      else if (dy < -_MAX_EXTRAP_DISPL) yi = pt.y - _MAX_EXTRAP_DISPL;
      return { x: xi, y: yi, c: pt.c };
    });
    return { id: p.id, landmarks: interp, snap: p.snap, member_id: p.member_id };
  });
}

function lerpPosePersons(a, b, t) {
  if (!a || !b) return b ? (b.persons || []) : [];
  var prevById = {};
  (a.persons || []).forEach(function (p) { prevById[p.id] = p; });
  return (b.persons || []).map(function (p) {
    var pp = prevById[p.id];
    if (!pp || !pp.landmarks) return p;
    var lmPrev = pp.landmarks || [];
    var lmNext = p.landmarks || [];
    var interp = lmNext.map(function (pt, i) {
      var ptPrev = lmPrev[i];
      if (!pt || !ptPrev) return pt;
      return {
        x: ptPrev.x + (pt.x - ptPrev.x) * t,
        y: ptPrev.y + (pt.y - ptPrev.y) * t,
        c: pt.c
      };
    });
    return { id: p.id, landmarks: interp, snap: p.snap, member_id: p.member_id };
  });
}

function syncPoseForVideoPts(videoPts90) {
  var buf = state.sync.poseBuffer;
  if (!buf.length) return null;
  if (videoPts90 == null) return { snap: buf[buf.length - 1], persons: buf[buf.length - 1].persons || [] };

  var before = null;
  var after = null;
  for (var i = 0; i < buf.length; i++) {
    var p = buf[i];
    if (p.video_pts90 == null) continue;
    if (diffPts90(p.video_pts90, videoPts90) <= 0) before = p;
    else { after = p; break; }
  }

  if (!before) {
    var first = after || buf[0];
    return { snap: first, persons: first.persons || [] };
  }
  if (!after) {
    return { snap: before, persons: before.persons || [] };
  }

  var span = diffPts90(after.video_pts90, before.video_pts90);
  if (span <= 0) return { snap: before, persons: before.persons || [] };
  var t = diffPts90(videoPts90, before.video_pts90) / span;
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  return { snap: before, persons: lerpPosePersons(before, after, t) };
}

function drawOverlay(persons, zones) {
  var cv = $("pose-canvas");
  if (!cv) return;
  var ctx = cv.getContext("2d");
  var W = cv.width || 1, H = cv.height || 1;
  ctx.clearRect(0, 0, W, H);
  if (!persons.length && !zones) return;

  // ── zone polygons ──
  if (zones) {
    var crowded = {};
    (zones.crowded || []).forEach(function (id) { crowded[id] = true; });
    state.zones.forEach(function (z) {
      var poly = z.polygon;
      if (!poly || poly.length < 3) return;
      var isCrowd = !!crowded[z.id];
      // fill with alpha
      ctx.save();
      ctx.globalAlpha = 0.15;
      ctx.fillStyle = isCrowd ? "#ff5d5d" : "#00ff00";
      ctx.beginPath();
      poly.forEach(function (pt, i) {
        if (i === 0) ctx.moveTo(pt[0] * W, pt[1] * H);
        else ctx.lineTo(pt[0] * W, pt[1] * H);
      });
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      // stroke
      ctx.strokeStyle = isCrowd ? "#ff5d5d" : "#00ff00";
      ctx.lineWidth = 2;
      ctx.beginPath();
      poly.forEach(function (pt, i) {
        if (i === 0) ctx.moveTo(pt[0] * W, pt[1] * H);
        else ctx.lineTo(pt[0] * W, pt[1] * H);
      });
      ctx.closePath();
      ctx.stroke();
      // label
      if (poly.length > 0) {
        var lx = poly[0][0] * W + 4, ly = poly[0][1] * H + 4;
        var label = z.name;
        if (z.capacity) label += (isCrowd ? " !" : "") + " " + z.capacity;
        ctx.font = "12px sans-serif";
        ctx.fillStyle = "#fff";
        ctx.fillText(label, lx, ly);
      }
    });
  }

  // ── skeleton ──
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#6ea8ff";
  ctx.fillStyle = "#ff5d5d";
  persons.forEach(function (p) {
    var lm = p.landmarks || [];
    ctx.beginPath();
    SKELETON.forEach(function (e) {
      var pa = lm[e[0]], pb = lm[e[1]];
      if (pa && pb && pa.c > 0.15 && pb.c > 0.15) {
        ctx.moveTo(pa.x * W, pa.y * H);
        ctx.lineTo(pb.x * W, pb.y * H);
      }
    });
    ctx.stroke();
    lm.forEach(function (pt) {
      if (pt && pt.c > 0.15) { ctx.beginPath(); ctx.arc(pt.x * W, pt.y * H, 4, 0, Math.PI * 2); ctx.fill(); }
    });
  });

  // ── per-person exercise tags ──
  ctx.font = "11px monospace";
  var yOff = 18;
  persons.forEach(function (p) {
    var sn = p.snap || {};
    var prefix = "[" + p.id + "]";
    if (p.member_id) prefix += " " + p.member_id;
    var line = prefix + " " + (sn.exercise || "") + " reps=" + (sn.reps || 0) +
      " sets=" + (sn.sets || 0) + " bad=" + (sn.bad_reps || 0) +
      " phase=" + (sn.phase || "");
    if (sn.angle != null) line += " " + sn.angle + "°";
    if (sn.rest_seconds > 0) line += " rest=" + Math.round(sn.rest_seconds) + "s";
    // bg
    var tw = ctx.measureText(line).width;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(4, yOff - 12, tw + 8, 16);
    ctx.fillStyle = "#fff";
    ctx.fillText(line, 8, yOff);
    yOff += 20;
  });

  // ── status bar ──
  if (zones) {
    var parts = ["total=" + (zones.total || 0)];
    var counts = zones.counts || {};
    Object.keys(counts).forEach(function (id) { parts.push(id + "=" + counts[id]); });
    // Show the *measured* stream fps (from the MSE player's PTS timing) so the
    // bar reflects actual throughput, distinct from the header's Target FPS.
    var measured = state.hd && state.hd.getFps ? state.hd.getFps() : null;
    var target = state.latest && state.latest.fps;
    if (measured != null && measured > 0) parts.push(measured + "fps");
    else if (target != null) parts.push(target + "fps");
    var barText = "  " + parts.join("  ");
    var bw = ctx.measureText(barText).width;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(4, H - 28, bw + 8, 20);
    ctx.fillStyle = "#fff";
    ctx.fillText(barText, 8, H - 14);
  }
}

// ── requestAnimationFrame overlay loop ──
function animateOverlay() {
  requestAnimationFrame(animateOverlay);
  if (state.mode !== "hd" && state.mode !== "mjpeg" && state.mode !== "sync") return;
  if (state.mode === "mjpeg" && state.videoMode) return;
  if (state.mode === "sync") {
    var videoPts90 = state.hd && state.hd.getSourcePts90 ? state.hd.getSourcePts90() : null;
    var picked = syncPoseForVideoPts(videoPts90);
    if (!picked) return;
    drawOverlay(picked.persons || [], picked.snap.zones);
    return;
  }
  var last = state.lastSnapshot;
  if (!last) return;
  var now = performance.now() / 1000;
  var persons = interpPersons(state.prevSnapshot, last, now);
  drawOverlay(persons, last.zones);
}

// ── mode switching ──
function setMode(mode) {
  state.mode = mode;
  $("btn-hd").classList.toggle("active", mode === "hd");
  $("btn-mjpeg").classList.toggle("active", mode === "sync" || mode === "mjpeg");
  $("badge-mode").textContent = mode === "hd" ? "HD" : (mode === "sync" ? "SYNC" : "MJPEG");
  var video = $("hd-video"), img = $("mjpeg-img"), cv = $("pose-canvas"), empty = $("video-empty");
  if (mode === "hd") {
    exitSync();
    img.style.display = "none";
    video.style.display = "";
    cv.style.display = "";
    empty.classList.add("hide");
    enterHd();
  } else if (mode === "sync") {
    exitHd();
    img.style.display = "none";
    video.style.display = "";
    cv.style.display = "";
    cv.getContext("2d").clearRect(0, 0, cv.width || 1, cv.height || 1);
    enterSync();
  } else {
    exitHd();
    exitSync();
    video.style.display = "none";
    cv.style.display = state.videoMode ? "none" : "";
    var ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width || 1, cv.height || 1);
    img.style.display = "";
    empty.classList.add("hide");
    img.src = appUrl("/stream") + "?t=" + Date.now();
  }
}

function startHdVideo(startText, opts) {
  if (startText) $("video-hint").textContent = startText;
  return fetch(appUrl("/api/preview")).then(function (r) { return r.json(); }).then(function (d) {
    if (!d || !d.wsUrl) { $("video-empty").textContent = "HD preview is disabled; PLATFORM_API_TOKEN is required"; $("video-empty").classList.remove("hide"); return; }
    if (!window.MediaSource) { $("video-empty").textContent = "This browser does not support MSE"; $("video-empty").classList.remove("hide"); return; }
    if (state.hd) state.hd.stop();
    state.hd = createHdPlayer($("hd-video"), opts);
    state.hd.start(d.wsUrl);
    $("video-empty").classList.add("hide");
  }).catch(function () { $("video-empty").textContent = "Failed to load preview information"; $("video-empty").classList.remove("hide"); });
}

function enterHd() {
  startHdVideo("HD preview is live");
}

function exitHd() {
  if (state.hd) { state.hd.stop(); state.hd = null; }
}

// ── video upload ──
function uploadVideo(file) {
  var fd = new FormData();
  fd.append("file", file);
  $("video-hint").textContent = "Uploading...";
  fetch(appUrl("/api/video/upload"), { method: "POST", body: fd }).then(function (r) { return r.json(); }).then(function (d) {
    if (d.ok) {
      $("video-hint").textContent = "Playing: " + (d.path || "");
      $("btn-stop-video").disabled = false;
      state.videoMode = true;
      if (state.mode !== "mjpeg") setMode("mjpeg");   // uploaded clip is MJPEG-only
      else $("pose-canvas").style.display = "none";
    } else { $("video-hint").textContent = "Upload failed: " + (d.error || ""); }
  }).catch(function () { $("video-hint").textContent = "Upload failed"; });
}

function stopVideo() {
  fetch(appUrl("/api/video/control"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "stop" }) })
    .then(function () {
      state.videoMode = false;
      if (state.mode === "mjpeg") setMode("sync");
      $("btn-stop-video").disabled = true;
      $("video-hint").textContent = "Playback stopped";
    })
    .catch(function () { $("video-hint").textContent = "Failed to stop playback"; });
}

// ── boot ──
function boot() {
  loadConfig();
  connectSSE();

  $("btn-hd").addEventListener("click", function () { setMode("hd"); });
  $("btn-mjpeg").addEventListener("click", function () { setMode("sync"); });

  var video = $("hd-video"), cv = $("pose-canvas"), img = $("mjpeg-img");
  // size pose canvas to the video's intrinsic resolution once known
  video.addEventListener("loadedmetadata", function () {
    var vw = video.videoWidth || 1920, vh = video.videoHeight || 1080;
    if (cv.width !== vw || cv.height !== vh) { cv.width = vw; cv.height = vh; }
  });
  // size pose canvas when MJPEG image loads
  img.addEventListener("load", function () {
    var iw = img.naturalWidth || img.width || 1280;
    var ih = img.naturalHeight || img.height || 720;
    if (cv.width !== iw || cv.height !== ih) { cv.width = iw; cv.height = ih; }
  });

  $("video-file").addEventListener("change", function () {
    if (this.files && this.files.length) uploadVideo(this.files[0]);
    this.value = "";
  });
  $("btn-stop-video").addEventListener("click", stopVideo);

  setMode("hd");
  // start the requestAnimationFrame overlay loop
  requestAnimationFrame(animateOverlay);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
