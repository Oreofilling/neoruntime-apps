/* ShelfOps frontend. No framework — a polling + SSE ops console.
   Endpoints are relative so the app renders equally at the LAN root
   (host network, inbound 8891) and under the app-manager /apps/shelf-ops/ path. */

const API = {
  config:  "api/config",
  state:   "api/state",
  heatmap: "api/heatmap",
  events:  "api/events",
  stream:  "stream",
  preview: "api/preview",
  importImage: "api/import/image",
  importVideo: "api/import/video",
  importVideoStatus: "api/import/video/status",
  demo: "api/demo",
};

const $ = (id) => document.getElementById(id);

/* ---------- state ---------- */
let CONFIG = null;              // slots + vocabulary + thresholds
let LAST = null;                // latest /api/state snapshot
let HEATDATA = [];              // rows from /api/heatmap
let HEATSPAN = "6h";

/* live preview: HD = platform hardware H.264 -> browser MSE, with slot
   polygons + code chips drawn on a canvas overlay; Smooth = server
   MJPEG /stream (chips baked in by overlay.py). */
let HD = null;                  // createHdPlayer instance while HD runs
let PREVIEW_MODE = "smooth";    // "hd" | "smooth"

/* ---------- formatting ---------- */
const CODE_COLOR = { A: "#4ec94c", B: "#eea947", C: "#fd7b52", D: "#e8b067", E: "#4c9cd1" };
/* per-item detector categories (detector.py vocab sidecar "categories") —
   import-result chips. Distinct from the A-E slot codes: these count real
   detected items, not cell-position joins. */
const CATEGORY_COLOR = {
  bottle: "#4c9cd1", can: "#eea947", carton: "#fd7b52",
  bag: "#e8b067", jar: "#9c6cd1", box: "#4ec94c", other: "#8b95a7",
};
const STATE_COLOR = { EMPTY: "#dc3030", PARTIAL: "#fabe3c", FULL: "#56c94c" };
const STATE_EN = { EMPTY: "Empty", PARTIAL: "Partial", FULL: "Full" };
/* grid mode (path A): big-frame region + dim vacant cells — canvas twins of
   overlay.py _GRID_REGION_COLOR / _GRID_EMPTY_COLOR */
const GRID_REGION_COLOR = "#ebebeb";
const GRID_EMPTY_COLOR = "#696969";

function fmtTime(ts) {
  // device epoch seconds or ISO "2026-…" — keep readable, not necessarily local
  const t = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(t.getHours())}:${p(t.getMinutes())}:${p(t.getSeconds())}`;
}
function fmtBucket(bucket) {
  // bucket is bucket-start epoch seconds; group by hour-of-day
  const t = new Date(bucket * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(t.getMonth() + 1)}-${p(t.getDate())} ${p(t.getHours())}h`;
}
function hexToRgba(hex, alpha) {
  // "#rrggbb" + alpha -> css rgba() (heatmap cell fills)
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha.toFixed(2)})`;
}

/* ---------- HD preview: H264 -> fMP4 muxer + MSE player -------------------
   Ported from gym-ops (itself from model-showcase), which has run on the
   same NE503 platform-api stream. Wire format per WS message: 18-byte
   FrameHeaderV2 (magic 0xFFD8FF00, frameType, flags[bit0=hasDts], PTS@90kHz,
   DTS@90kHz, reserved) + AVCC NAL units; cached SPS/PPS arrive first, each
   in its own frame. */
// minimal H264->fMP4 muxer: init (ftyp+moov) + media (moof+mdat) segments.
// The platform broadcasts AVCC-format NALs (4-byte length prefix), so each
// access unit is fed as one mdat sample — no Annex-B conversion needed.
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

// compact MSE player: live-edge management, frame watchdog (5s silence ->
// force ws.close), exponential-backoff restarts (800ms->8s), auto-degrade to
// the caller's fallback after 3 restarts/30s or 3 MSE errors.
function createHdPlayer(video, opts) {
  opts = opts || {};
  var liveDelay = opts.liveDelaySeconds != null ? opts.liveDelaySeconds : 0.08;
  var ws = null, ms = null, sb = null, muxer = null;
  var sps = null, pps = null, initialized = false, hasFirstI = false;
  var spsW = 1920, spsH = 1080;
  var firstPts, lastPts, fps = 30, nextTs = 0;
  var queue = [], url = '', active = false, destroyed = false;
  var mseOpen = false, retryTimer = null, mseErrs = 0, restartTimer = null;
  var watchdogTimer = null, lastDataTs = 0, stableMs = 0;
  var restartBackoffMs = 800, restartHistory = [], degraded = false;
  var adaptiveDelay = opts.adaptiveDelay !== false;

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
        if (target > video.currentTime + 0.3) {
          video.currentTime = target;
        } else if (target < video.currentTime - 0.05 && lead < 0.5) {
          if (adaptiveDelay && liveDelay < 4.0) liveDelay = Math.min(4.0, liveDelay + 0.5);
          if (lead < 0.15) video.currentTime = target;
        }
        if (sb.buffered.length && video.currentTime > sb.buffered.start(0) + Math.max(4, liveDelay + 3)) { sb.remove(0, video.currentTime - Math.max(3, liveDelay + 2)); return; }
      }
    } catch (e) {}
  }
  function onMseError() { if (destroyed || !active) return; mseErrs++; scheduleRestart(); }
  function scheduleRestart() {
    if (restartTimer || destroyed || !active) return;
    var now = Date.now();
    restartHistory.push(now);
    while (restartHistory.length && now - restartHistory[0] > 30000) restartHistory.shift();
    if (maybeDegrade()) { teardownMse(); return; }
    var delay = restartBackoffMs;
    restartBackoffMs = Math.min(8000, restartBackoffMs * 2);
    restartTimer = setTimeout(function () {
      restartTimer = null;
      if (active && !destroyed) { teardownMse(); openMse(); if (url) connectWs(url); }
    }, delay);
  }
  function maybeDegrade() {
    if (degraded || destroyed || !active || typeof opts.onDegrade !== "function") return false;
    if (restartHistory.length >= 3 || mseErrs >= 3) {
      degraded = true;
      try { opts.onDegrade(restartHistory.length >= 3 ? "restart storm" : "decode errors"); } catch (e) {}
      return true;
    }
    return false;
  }
  function tickWatchdog() {
    if (destroyed || !active) return;
    var idleMs = lastDataTs ? Date.now() - lastDataTs : 0;
    if (idleMs > 5000 && ws && ws.readyState === 1) { try { ws.close(); } catch (e) {} return; }
    if (idleMs < 2000) { stableMs += 1000; if (stableMs >= 30000) { restartBackoffMs = 800; stableMs = 0; } }
    else stableMs = 0;
  }
  function resetDecoder() { initialized = false; muxer = null; hasFirstI = false; nextTs = 0; firstPts = undefined; lastPts = undefined; pps = null; queue = []; }
  function teardownMse() {
    try { if (sb) { sb.removeEventListener('updateend', onUpd); sb.removeEventListener('error', onMseError); if (ms && ms.readyState === 'open') { try { sb.abort(); } catch (e) {} } } } catch (e) {}
    sb = null; resetDecoder(); sps = null; mseOpen = false;
    try { if (ms) { ms.removeEventListener('sourceopen', onSourceOpen); if (ms.readyState === 'open') { try { ms.endOfStream(); } catch (e) {} } } } catch (e) {}
    try { if (video.src) URL.revokeObjectURL(video.src); } catch (e) {}
    video.src = ''; ms = null;
  }
  function openMse() { if (!window.MediaSource) return false; ms = new MediaSource(); video.src = URL.createObjectURL(ms); ms.addEventListener('sourceopen', onSourceOpen); return true; }
  function onSourceOpen() { mseOpen = true; video.play().catch(function () {}); }

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
    ws.onmessage = function (ev) { lastDataTs = Date.now(); handleMsg(ev.data); };
    ws.onopen = function () { mseErrs = 0; };
    ws.onclose = function () { if (active && !destroyed) { retryTimer = setTimeout(function () { retryTimer = null; if (active && !destroyed && url) connectWs(url); }, 1200); } };
    ws.onerror = function () { try { ws.close(); } catch (e) {} };
  }

  return {
    start: function (u) {
      url = u; active = true; destroyed = false; mseErrs = 0;
      lastDataTs = Date.now(); stableMs = 0; restartBackoffMs = 800; restartHistory = []; degraded = false;
      resetDecoder(); sps = null; pps = null; openMse(); connectWs(u);
      if (!watchdogTimer) watchdogTimer = setInterval(tickWatchdog, 1000);
    },
    stop: function () {
      active = false;
      if (watchdogTimer) { clearInterval(watchdogTimer); watchdogTimer = null; }
      if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
      if (restartTimer) { clearTimeout(restartTimer); restartTimer = null; }
      if (ws) { ws.onclose = null; ws.onerror = null; ws.onmessage = null; try { ws.close(); } catch (e) {} ws = null; }
      teardownMse();
    },
    // (PTS90 getters from gym-ops omitted: shelf-ops has no pose-sync consumer.)
    setLiveDelaySeconds: function (seconds) {
      if (isFinite(seconds)) liveDelay = Math.max(0.08, Math.min(4.0, seconds));
    },
    getFps: function () { return fps; }   // measured stream fps from PTS deltas
  };
}

/* ---------- preview mode + canvas overlay --------------------------------- */
function setNote(text) {
  const note = $("preview-note");
  if (!note) return;
  note.textContent = text || "";
  note.classList.toggle("show", Boolean(text));
}

// True while the built-in demo video runs through the live pipeline. The
// backend swaps /api/config to the demo view (d-prefixed slots + demo grid),
// so every CONFIG-driven surface just re-fetches config after a toggle.
function demoOn() {
  return Boolean(CONFIG && CONFIG.demo && CONFIG.demo.enabled);
}

// Demo-video mode forces Smooth preview (HD = hardware camera H.264, and the
// camera is not the demo source) — checked before every switch to "hd".
function demoBlocksHd() { return demoOn(); }

/* Toasts (top-center of the preview): stock-out / restock notifications while
   the events feed scrolls below. textContent only — no server data as HTML. */
const TOAST_MS = 5000;
const TOAST_MAX = 4;
function toast(text, tone) {
  const host = $("toasts");
  if (!host) return;
  const el = document.createElement("div");
  el.className = "toast" + (tone ? " " + tone : "");
  el.textContent = text || "";
  host.appendChild(el);
  while (host.children.length > TOAST_MAX) host.removeChild(host.firstChild);
  requestAnimationFrame(() => el.classList.add("show"));
  window.setTimeout(() => {
    el.classList.remove("show");
    window.setTimeout(() => el.remove(), 350);
  }, TOAST_MS);
}

// Canvas twin of overlay.py: slot polygons + code chips in the same palette,
// drawn over the MSE video from the last SSE snapshot. Slot polygons are
// normalized 0..1 and the canvas bitmap matches the video's intrinsic size,
// so object-fit:contain letterboxes identically.
function paintChip(ctx, x, yTop, text, color, fs) {
  const ch = Math.round(fs * 1.45);
  ctx.font = `${fs}px monospace`;
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(18,18,18,.85)";
  ctx.fillRect(x, yTop, tw + 12, ch + 4);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 6, yTop + ch);
}

// Grid mode (path A): one thick region rect, dim vacant cell borders, a code
// chip only on occupied cells, and a GOODS n/N summary — mirrors overlay.py
// _draw_grid so HD canvas and Smooth MJPEG tell the same story.
// show_cells=false (grid.show_cells in config.yaml) is the whole-frame
// one-shelf view: region rect + summary + per-item boxes, no cell lines.
function drawGridOverlay(ctx, W, H, lw, fs, snaps) {
  const showCells = !(CONFIG.grid && CONFIG.grid.show_cells === false);
  const [rx1, ry1, rx2, ry2] = CONFIG.grid.region;
  const X1 = rx1 * W, Y1 = ry1 * H, X2 = rx2 * W, Y2 = ry2 * H;
  ctx.lineWidth = Math.max(3, lw * 1.5);
  ctx.strokeStyle = GRID_REGION_COLOR;
  ctx.strokeRect(X1, Y1, X2 - X1, Y2 - Y1);

  const goods = LAST.goods || {};
  const occupied_n = goods.occupied != null
    ? goods.occupied
    : (LAST.slots || []).filter((s) => s.count > 0).length;
  const total = goods.cells || (CONFIG.slots || []).length;
  const summaryFs = Math.round(fs * 1.2);
  // ITEMS m (chain A per-item boxes) rides along when the detector runs;
  // undefined (detector off) keeps the plain GOODS n/N text
  const summaryText = LAST.items != null
    ? `GOODS ${occupied_n}/${total} · ITEMS ${LAST.items}`
    : `GOODS ${occupied_n}/${total}`;
  // summary chip rect first: when the region touches the top edge there is
  // no room above, and top-left cell chips must dodge below it (overlay.py twin)
  ctx.font = `${summaryFs}px monospace`;
  const sumW = Math.round(ctx.measureText(summaryText).width) + 12;
  const sumH = Math.round(summaryFs * 1.45) + 4;
  const sx = X1 + 2;
  const sy = Math.max(0, Y1 - sumH - 6);
  const sumRight = sx + sumW, sumBottom = sy + sumH;

  for (const slot of showCells ? (CONFIG.slots || []) : []) {
    const poly = slot.polygon;
    if (!poly || poly.length < 3) continue;
    const snap = snaps[slot.id];
    const occupied = snap && snap.count > 0;
    const held = occupied ? Object.keys(snap.by_code || {})[0] || "" : "";
    // demo mode: CLIP-CONFIRMED empty (state EMPTY, count 0) gets the thick
    // red border + EMPTY chip — twin of overlay.py highlight_empty; cells not
    // yet confirmed stay dim gray so confidence reads at a glance
    const confirmedEmpty = !occupied && demoOn() && snap
      && snap.state === "EMPTY" && snap.count === 0;
    ctx.beginPath();
    poly.forEach((pt, i) => (i ? ctx.lineTo(pt[0] * W, pt[1] * H) : ctx.moveTo(pt[0] * W, pt[1] * H)));
    ctx.closePath();
    ctx.lineWidth = confirmedEmpty ? Math.max(4, lw * 2)
      : occupied ? Math.max(2, lw) : 1;
    ctx.strokeStyle = confirmedEmpty ? STATE_COLOR.EMPTY
      : occupied ? (CODE_COLOR[held] || "#c8c8c8") : GRID_EMPTY_COLOR;
    ctx.stroke();
    if (occupied) {
      // chip shows the *held* code (from by_code) so a missed read this tick
      // doesn't blank the label; cos joins only when confident right now
      const score = snap.code && snap.code !== "EMPTY" ? " " + snap.score.toFixed(2) : "";
      const cellFs = Math.round(fs * 0.9);
      const cx = poly[0][0] * W + 2;
      let cy = poly[0][1] * H + 2;
      if (cx < sumRight && cy < sumBottom) cy = sumBottom + 2; // dodge summary
      paintChip(ctx, cx, cy, `${slot.id} ${held}${score}`,
                CODE_COLOR[held] || "#c8c8c8", cellFs);
    } else if (confirmedEmpty) {
      const cellFs = Math.round(fs * 0.9);
      const cx = poly[0][0] * W + 2;
      let cy = poly[0][1] * H + 2;
      if (cx < sumRight && cy < sumBottom) cy = sumBottom + 2; // dodge summary
      paintChip(ctx, cx, cy, `${slot.id} EMPTY`, STATE_COLOR.EMPTY, cellFs);
    }
  }

  paintChip(ctx, sx, sy, summaryText, GRID_REGION_COLOR, summaryFs);
}

// Per-item detection boxes (chain A) painted over the grid/slot layer —
// canvas twin of overlay.py draw_detections: goods bright green with a
// `label score [cell]` chip, negative hits (person/hand) thin gray.
const DET_GOODS_COLOR = "#4ec94c";
const DET_NEG_COLOR = "#a0a0a0";

function drawDetections(ctx, W, H, fs) {
  for (const d of (LAST && LAST.detections) || []) {
    const b = d.box || [];
    if (b.length !== 4) continue;
    const px1 = b[0] * W, py1 = b[1] * H, px2 = b[2] * W, py2 = b[3] * H;
    const goods = !!d.goods;
    const color = goods ? DET_GOODS_COLOR : DET_NEG_COLOR;
    ctx.lineWidth = goods ? 2 : 1;
    ctx.strokeStyle = color;
    ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
    let label = `${d.label || "?"} ${(d.score || 0).toFixed(2)}`;
    if (d.cell) label += ` [${d.cell}]`;
    // chip above the box (inside it when the box touches the top edge)
    const ch = Math.round(fs * 1.45);
    ctx.font = `${fs}px monospace`;
    const tw = ctx.measureText(label).width;
    let cy = py1 - ch - 6;
    if (cy < 0) cy = py1 + 2;
    const cx = Math.min(px1, W - tw - 12);
    paintChip(ctx, Math.max(0, cx), cy, label, color, fs);
  }
}

function drawOverlay() {
  if (PREVIEW_MODE !== "hd") return;
  const cv = $("overlay-canvas");
  if (!cv) return;
  const W = cv.width, H = cv.height;
  if (!W || !H) return;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  if (!CONFIG || !LAST) return;

  const lw = Math.max(2, Math.round(W / 640));       // 2px @640 stream scale
  const fs = Math.max(11, Math.round(W / 78));       // label font
  const snaps = (LAST.slots || []).reduce((m, s) => (m[s.slot_id] = s, m), {});

  if (CONFIG.mode === "grid" && CONFIG.grid) {
    drawGridOverlay(ctx, W, H, lw, fs, snaps);
    drawDetections(ctx, W, H, fs);
    return;
  }

  for (const slot of CONFIG.slots || []) {
    const poly = slot.polygon;
    if (!poly || poly.length < 3) continue;
    const snap = snaps[slot.id];
    const state = snap ? snap.state : "PARTIAL";
    const color = STATE_COLOR[state] || "#c8c8c8";
    ctx.beginPath();
    poly.forEach((pt, i) => (i ? ctx.lineTo(pt[0] * W, pt[1] * H) : ctx.moveTo(pt[0] * W, pt[1] * H)));
    ctx.closePath();
    ctx.lineWidth = state === "EMPTY" ? lw * 2.5 : lw;
    ctx.strokeStyle = color;
    ctx.stroke();

    // chip: recognized vocabulary code (or EMPTY) + cosine confidence,
    // tinted by the code color — mirrors overlay.py draw_slots
    const code = (snap && snap.code) || "";
    const shown = code && code !== "EMPTY" ? code : "EMPTY";
    const chip = `${slot.id} ${shown}${snap && snap.score ? " " + snap.score.toFixed(2) : ""}`;
    const chipColor = code && code !== "EMPTY" ? (CODE_COLOR[code] || "#c8c8c8") : color;
    const x0 = poly[0][0] * W, y0 = poly[0][1] * H;
    const cy = Math.max(0, y0 - Math.round(fs * 1.45) - 2);
    paintChip(ctx, x0, cy, chip, chipColor, fs);
  }

  drawDetections(ctx, W, H, fs);
}

function enterHd() {
  $("preview").style.display = "none";
  $("preview").removeAttribute("src");   // stop any MJPEG client (server idles)
  $("hd-video").style.display = "block";
  $("overlay-canvas").style.display = "block";
  setNote("HD · connecting hardware H.264…");
  fetch(API.preview)
    .then((r) => r.json())
    .then((d) => {
      if (PREVIEW_MODE !== "hd") return;                    // user switched away
      if (!d || !d.enabled || !d.wsUrl) { degradeToSmooth("HD unavailable (no platform token)"); return; }
      if (!window.MediaSource) { degradeToSmooth("browser lacks MSE"); return; }
      if (HD) HD.stop();
      HD = createHdPlayer($("hd-video"), { onDegrade: degradeToSmooth });
      HD.start(d.wsUrl);
      setNote("HD · hardware H.264 + MSE (chips drawn client-side)");
    })
    .catch(() => { if (PREVIEW_MODE === "hd") degradeToSmooth("preview info fetch failed"); });
}

function exitHd() {
  if (HD) { HD.stop(); HD = null; }
  const cv = $("overlay-canvas");
  if (cv && cv.width) cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
}

function enterSmooth() {
  exitHd();
  $("hd-video").style.display = "none";
  $("overlay-canvas").style.display = "none";
  const img = $("preview");
  img.style.display = "block";
  img.src = API.stream;                  // overlay.py bakes chips into MJPEG
  setNote("Smooth · server MJPEG (chips baked in)");
}

function exitSmooth() {
  const img = $("preview");
  img.removeAttribute("src");            // drop the /stream client
}

function degradeToSmooth(reason) {
  if (PREVIEW_MODE !== "hd") return;
  setPreviewMode("smooth");
  setNote(`Smooth · ${reason}`);
}

function setPreviewMode(mode) {
  // demo video runs through the MJPEG /stream (chips baked by overlay.py);
  // the HD path is the hardware camera encoder, which has no demo source
  if (mode === "hd" && demoBlocksHd() && PREVIEW_MODE === "smooth") {
    // neutral wording: a customer clicking HD mid-demo must not learn why
    setNote("Smooth · HD unavailable right now");
    return;
  }
  PREVIEW_MODE = mode;
  const isHd = mode === "hd";
  $("btn-hd").classList.toggle("active", isHd);
  $("btn-smooth").classList.toggle("active", !isHd);
  if (isHd) { exitSmooth(); enterHd(); }
  else { exitHd(); enterSmooth(); }
}

/* ---------- topbar status ---------- */
const STATUS_PILL = { live: ["live", "LIVE"], simulation: ["sim", "SIM"], degraded: ["degraded", "DEGRADED"], error: ["degraded", "ERROR"] };
function paintStatus(status) {
  const s = STATUS_PILL[status] || ["off", status || "?"];
  const statEl = $("stat-status");
  statEl.className = "stat " + s[0];
  statEl.querySelector("b").textContent = s[1];
}

/* topbar goods counter — grid mode shows physical ITEM count (件数) whenever
   the per-item detector runs; occupied areas only as a fallback while it is
   off (title says so — the number's meaning must not silently change).
   slots mode: count of currently stocked slots */
function paintGoods() {
  const el = $("stat-goods");
  if (!el || !LAST) return;
  if (CONFIG && CONFIG.mode === "grid") {
    if (LAST.items != null) {
      el.textContent = String(LAST.items);
      el.title = "physical items counted by the per-item detector";
    } else {
      const goods = LAST.goods;
      const n = goods && goods.occupied != null
        ? goods.occupied
        : (LAST.slots || []).filter((s) => s.count > 0).length;
      el.textContent = String(n);
      el.title = "item detector off — showing occupied areas";
    }
    return;
  }
  const n = (LAST.slots || []).filter((s) => s.count > 0).length;
  el.textContent = String(n);
}

/* ---------- legends ---------- */
/* English goods label for a vocabulary code (falls back to the code). */
function codeLabel(code) {
  if (!code || code === "EMPTY" || !CONFIG) return "";
  const v = (CONFIG.vocabulary || []).find((x) => x.code === code);
  return (v && v.label) || code;
}

function renderLegends() {
  if (!CONFIG) return;
  const cont = $("legends");
  cont.innerHTML = "";
  for (const v of CONFIG.vocabulary) {
    if (v.code === "EMPTY") continue;   // no color of its own; noise in the legend
    const chip = document.createElement("span");
    chip.className = "code-chip";
    chip.dataset.code = v.code;
    chip.innerHTML = `<b>${v.code}</b><span>${v.label || v.code}</span>`;
    cont.appendChild(chip);
  }
}

/* ---------- rack / planogram ---------- */
function renderRack() {
  const rack = $("rack");
  if (!CONFIG || !LAST) {
    rack.innerHTML = `<p class="empty">Waiting for first scan…</p>`;
    return;
  }
  const snaps = (LAST.slots || []).reduce((m, s) => (m[s.slot_id] = s, m), {});
  rack.innerHTML = "";

  // grid mode: headline ITEM counts (件数 — each per-item detection joined to
  // its cell's goods category) + a mini map laid out exactly like the camera
  // view (row 1 = top of the picture) so users can locate goods. Area counts
  // only appear as a labeled fallback while the item detector is off.
  if (CONFIG.mode === "grid" && CONFIG.grid) {
    const goods = LAST.goods || {};
    const el = document.createElement("div");
    const chipsFrom = (entries) => entries.length
      ? entries.map(([c, m]) => {
          const other = c === "other";
          const label = other ? "other goods" : codeLabel(c);
          const dataAttr = other ? "" : ` data-code="${c}"`;
          return `<span class="mini-code"${dataAttr}><b></b><span class="mini-n">${label} × ${m}</span></span>`;
        }).join("")
      : `<span class="mini-code"><span class="mini-n">no goods recognized yet</span></span>`;

    if (LAST.items_by_code != null) {
      // item mode: N items + per-category item counts ("other" last)
      const entries = Object.entries(LAST.items_by_code)
        .filter(([c]) => CODE_COLOR[c] || c === "other")
        .sort(([a], [b]) => (a === "other") - (b === "other"));
      el.className = "slot s-" + (LAST.items > 0 ? "FULL" : "EMPTY") + " goods-summary";
      el.innerHTML = `
        <div class="slot-top">
          <span class="slot-name">Goods on the shelf</span>
          <span class="slot-state">${LAST.items} item${LAST.items === 1 ? "" : "s"}</span>
        </div>
        <div class="slot-codes">${chipsFrom(entries)}</div>`;
    } else {
      // detector off: occupied areas, explicitly labeled as areas
      const n = goods.occupied != null ? goods.occupied
        : (LAST.slots || []).filter((s) => s.count > 0).length;
      const total = goods.cells || CONFIG.slots.length;
      const pct = total ? Math.round((n / total) * 100) : 0;
      const byCode = Object.entries(goods.by_code || {}).filter(([c]) => CODE_COLOR[c]);
      el.className = "slot s-" + (n > 0 ? "FULL" : "EMPTY") + " goods-summary";
      el.innerHTML = `
        <div class="slot-top">
          <span class="slot-name">Goods on the shelf</span>
          <span class="slot-state">item detector off</span>
        </div>
        <div class="slot-meter">
          <div class="slot-bar"><i style="width:${pct}%"></i></div>
          <span class="slot-count">${n}<span class="cap"> / ${total} areas</span></span>
        </div>
        <div class="slot-codes">${chipsFrom(byCode)}</div>`;
    }
    rack.appendChild(el);

    // one tile per grid cell, same row-major order as build_grid_slots();
    // tile tint = recognized goods color, dashed = empty area
    const cols = CONFIG.grid.cols || 1;
    const map = document.createElement("div");
    map.className = "rack-grid";
    map.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    for (const slot of CONFIG.slots) {
      const s = snaps[slot.id] || { state: "EMPTY", count: 0, by_code: {} };
      const label = codeLabel(s.code);
      const match = label && s.score ? ` · ${Math.round(s.score * 100)}% match` : "";
      const pos = /^g(\d+)-(\d+)$/.exec(slot.id);
      const tile = document.createElement("div");
      tile.className = "rack-tile s-" + s.state;
      if (s.code && s.code !== "EMPTY") tile.dataset.code = s.code;
      tile.title = pos
        ? `Row ${+pos[1]} · Column ${+pos[2]}${label ? " — " + label : ""}${match}`
        : `${slot.id}${label ? " — " + label : ""}${match}`;
      tile.innerHTML = `<span class="tile-name${label ? "" : " dim"}">${label || "empty"}</span>`;
      map.appendChild(tile);
    }
    rack.appendChild(map);
    // the area mini-map is opt-in (hidden by default) — item counts made it
    // secondary; "Show grid map" in the panel head brings it back
    map.hidden = !gridMapOn();
    return;
  }

  for (const slot of CONFIG.slots) {
    const s = snaps[slot.id] || { state: "EMPTY", count: 0, by_code: {} };
    const cap = slot.capacity || 0;
    const pct = cap ? Math.min(100, Math.round((s.count / cap) * 100)) : 0;
    const el = document.createElement("div");
    el.className = "slot s-" + s.state;

    const byCode = Object.entries(s.by_code || {}).filter(([c]) => CODE_COLOR[c]);
    // expected_code = planogram guidance; the recognized chip shows the
    // goods label (code + match % live in its tooltip) so reality vs plan
    // reads at a glance
    const expectLabel = codeLabel(slot.expected_code) || slot.expected_code;
    const expected = slot.expected_code ? `<span class="expect">plan: ${expectLabel}</span>` : "";
    const seen = s.code && s.code !== "EMPTY"
      ? `<span class="mini-code" data-code="${s.code}" title="${s.code} · ${Math.round((s.score || 0) * 100)}% match"><b></b><span class="mini-n">${codeLabel(s.code)}</span></span>`
      : s.code === "EMPTY"
        ? `<span class="mini-code"><span class="mini-n">empty</span></span>`
        : `<span class="mini-code"><span class="mini-n">—</span></span>`;
    el.innerHTML = `
      <div class="slot-top">
        <span class="slot-id">${slot.id}</span>
        <span class="slot-name">${slot.name || slot.id}${expected}</span>
        <span class="slot-state">${STATE_EN[s.state] || s.state}</span>
      </div>
      <div class="slot-meter">
        <div class="slot-bar"><i style="width:${pct}%"></i></div>
        <span class="slot-count">${s.count}<span class="cap"> / ${cap}</span></span>
      </div>
      <div class="slot-codes">${seen}${
        byCode.length
          ? byCode.map(([c, n]) => `
              <span class="mini-code" data-code="${c}"><b></b><span class="mini-n">${codeLabel(c)} × ${n}</span></span>`).join("")
          : ""
      }</div>`;
    rack.appendChild(el);
  }
}

/* ---------- heatmap ---------- */
async function fetchHeatmap() {
  const res = await fetch(`${API.heatmap}?span=${HEATSPAN}`);
  const data = await res.json();
  HEATDATA = data.rows || [];
  renderHeatmap();
}

function renderHeatmap() {
  const grid = $("heat-grid");
  if (!CONFIG) return;
  if (!HEATDATA.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Waiting for data…";
    grid.replaceChildren(empty);
    return;
  }

  // rows: slots; columns: time buckets (asc). Each cell aggregates the
  // per-code peak counts into one tile: color = dominant goods code,
  // stronger fill = closer to that slot's capacity, number = peak count.
  const slots = CONFIG.slots;
  const buckets = [...new Set(HEATDATA.map((r) => r.bucket))].sort((a, b) => a - b);
  const byKey = new Map(HEATDATA.map((r) => [`${r.bucket}:${r.slot_id}:${r.code}`, r]));
  const capOf = (id) => (CONFIG.slots.find((s) => s.id === id) || {}).capacity || 1;

  // dominant code per bucket:slot; "all" rows only carry the peak count
  const domOf = new Map();
  for (const r of HEATDATA) {
    if (r.code === "all") continue;
    const k = `${r.bucket}:${r.slot_id}`;
    const cur = domOf.get(k);
    if (!cur || r.max_count > cur.max_count) domOf.set(k, r);
  }

  renderHeatLegend(new Set([...domOf.values()].map((r) => r.code)));

  const gridEl = document.createElement("div");
  gridEl.className = "hg";
  gridEl.style.gridTemplateRows = `34px repeat(${slots.length}, 28px)`;

  // label column
  const lblCol = document.createElement("div");
  lblCol.className = "hg-col hg-labels";
  lblCol.style.gridAutoFlow = "row";
  const head = document.createElement("time");
  head.style.height = "34px";
  head.textContent = "slot / time";
  lblCol.appendChild(head);
  for (const s of slots) {
    const rl = document.createElement("span");
    rl.className = "row-label";
    rl.style.height = "28px";
    rl.textContent = s.id;
    rl.title = `${s.id} · ${s.name || ""}`;
    lblCol.appendChild(rl);
  }
  gridEl.appendChild(lblCol);

  for (const b of buckets) {
    const col = document.createElement("div");
    col.className = "hg-col";
    const time = document.createElement("time");
    time.textContent = fmtBucket(b);
    time.style.height = "34px";
    col.appendChild(time);
    for (const s of slots) {
      const r = byKey.get(`${b}:${s.id}:all`);
      const best = domOf.get(`${b}:${s.id}`);
      const cap = capOf(s.id);
      const cell = document.createElement("div");
      cell.className = "cell";
      if (r && r.max_count) {
        const ratio = Math.min(1, r.max_count / cap);
        const hex = (best && CODE_COLOR[best.code]) || "#6ea8ff";
        const alpha = 0.25 + ratio * 0.7;
        cell.style.background = hexToRgba(hex, alpha);
        const n = document.createElement("span");
        n.className = "cell-n";
        n.textContent = String(r.max_count);
        // dark text only when the fill ends up bright (dense + light hue)
        const v = parseInt(hex.slice(1), 16);
        const lum = 0.299 * ((v >> 16) & 255) + 0.587 * ((v >> 8) & 255) + 0.114 * (v & 255);
        n.style.color = alpha * lum + (1 - alpha) * 15 > 110 ? "#0a0e14" : "#eef3fa";
        cell.appendChild(n);
        const what = best ? (codeLabel(best.code) || best.code) : "-";
        cell.title = `${s.id} · ${fmtBucket(b)} — ${what}, peak ${r.max_count} of capacity ${cap}`;
      } else {
        cell.classList.add("nil");
        cell.title = `${s.id} · ${fmtBucket(b)} — no data`;
      }
      col.appendChild(cell);
    }
    gridEl.appendChild(col);
  }
  grid.replaceChildren(gridEl);
}

/* legend rebuilt per render: dominant-code colors labeled with the live
   vocabulary, then the fill-ratio scale. createElement/textContent only. */
function renderHeatLegend(codes) {
  const legend = $("heat-legend");
  const known = ["A", "B", "C", "D", "E"].filter((c) => codes.has(c));
  const extra = [...codes].filter((c) => !/^[A-E]$/.test(c)).sort();
  legend.replaceChildren();
  for (const c of [...known, ...extra]) {
    const sw = document.createElement("span");
    sw.className = "sw";
    sw.style.background = CODE_COLOR[c] || "#6ea8ff";
    const lab = document.createElement("span");
    lab.className = "lab";
    lab.textContent = `${c} · ${codeLabel(c) || c}`;
    legend.append(sw, lab);
  }
  const sep = document.createElement("span");
  sep.className = "sep";
  sep.textContent = "|";
  const lo = document.createElement("span");
  lo.className = "lab";
  lo.textContent = "fill low";
  const hi = document.createElement("span");
  hi.className = "lab";
  hi.textContent = "high";
  legend.append(sep, lo);
  for (const ratio of [0.2, 0.6, 1]) {
    const sw = document.createElement("span");
    sw.className = "sw";
    sw.style.background = hexToRgba("#6ea8ff", 0.25 + ratio * 0.7);
    legend.appendChild(sw);
  }
  legend.appendChild(hi);
}

/* ---------- events ---------- */
function paintEvent(ev) {
  const ul = $("events");
  const first = ul.querySelector(".empty");
  if (first) first.remove();
  const li = document.createElement("li");
  const kind = ev.kind || "occupancy";
  const when = ev.ts ? fmtTime(ev.ts) : "";
  const slotName = CONFIG && CONFIG.slots.find((s) => s.id === ev.slot_id);
  const en = { stockout: "STOCK-OUT", restock: "RESTOCK" }[kind] || kind;
  li.innerHTML = `
    <span class="ev-time">${when}</span>
    <span class="ev-kind ev-${kind}">${en}</span>
    <span class="ev-msg">${slotName ? slotName.id + " · " + (slotName.name || "") : ev.slot_id || ""}</span>`;
  if (kind === "stockout" && ev.detail && ev.detail.empty_seconds != null) {
    li.querySelector(".ev-msg").textContent += ` · empty ${ev.detail.empty_seconds}s`;
  }
  ul.prepend(li);
  // cap list length
  while (ul.children.length > 60) ul.lastChild.remove();
  // 通知: surface stock-out / restock as a toast while the feed scrolls —
  // the customer-demo moment ("this slot needs refilling")
  if (kind === "stockout" || kind === "restock") {
    const who = slotName ? slotName.id : (ev.slot_id || "?");
    toast(`${en} · ${who}`, kind);
  }
}

/* ---------- polling loop ---------- */
async function tick() {
  try {
    const res = await fetch(API.state);
    const data = await res.json();
    LAST = data.scan || data;
    paintStatus(data.status || LAST.status);
    paintGoods();
    renderRack();
  } catch { /* server restarting — keep last snapshot */ }
}

// one-shot: pull model ids from /api/health for the topbar tag — two chains
// run since 0.3.0 (CLIP cell counting + yolo_world per-item detection), so
// the tag lists both when the detector is live.
async function paintModelTag() {
  try {
    const res = await fetch("api/health");
    const h = await res.json();
    let models = h.model || "-";
    if (h.detector_model && h.detector !== "unavailable" && h.detector !== "off") {
      const state = h.detector === "degraded" ? " (degraded)" : "";
      models += ` + ${h.detector_model}${state}`;
    }
    const tag = $("model-tag");
    if (tag) tag.textContent = `model: ${models} · ${h.mode || ""}`.trim();
  } catch { /* offline */ }
}

/* ---------- offline photo import + grid-map toggle ---------- */

function gridMapOn() {
  try { return localStorage.getItem("shelfops.gridmap") === "1"; }
  catch { return false; }   // storage blocked — stay hidden (the default)
}

let IMPORT_DROP_TPL = null;   // pristine dropzone markup for re-showing

function showDrop() {
  const body = $("import-body");
  const drop = $("import-drop");
  if (!IMPORT_DROP_TPL && drop) IMPORT_DROP_TPL = drop.cloneNode(true);
  const node = IMPORT_DROP_TPL ? IMPORT_DROP_TPL.cloneNode(true) : drop;
  body.replaceChildren(node);
  if (!node) return;
  node.addEventListener("click", () => $("import-file").click());
  node.addEventListener("dragover", (e) => e.preventDefault());
  node.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) doImport(file);
  });
}

function closeImport() { $("import-modal").hidden = true; }

async function doImport(file) {
  // videos go to the sampled job pipeline (POST + status polling)
  if (file.type.startsWith("video/") || /\.(mp4|mov|m4v)$/i.test(file.name)) {
    doImportVideo(file);
    return;
  }
  const body = $("import-body");
  const busy = document.createElement("div");
  busy.className = "import-busy";
  const name = document.createElement("p");
  name.textContent = file.name;
  const spin = document.createElement("p");
  spin.textContent = "Analyzing photo — counting items…";
  busy.append(name, spin);
  body.replaceChildren(busy);

  const fd = new FormData();
  fd.append("file", file, file.name);
  try {
    const res = await fetch(API.importImage, { method: "POST", body: fd });
    const reply = await res.json().catch(() => ({}));
    if (!res.ok || !reply.ok) throw new Error(reply.error || `HTTP ${res.status}`);
    renderImportResult(reply.result, `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`);
  } catch (err) {
    renderImportError(err && err.message ? err.message : "import failed");
  }
}

async function doImportVideo(file) {
  const body = $("import-body");
  const busy = document.createElement("div");
  busy.className = "import-busy";
  const name = document.createElement("p");
  name.textContent = file.name;
  const spin = document.createElement("p");
  spin.textContent = "Uploading video…";
  busy.append(name, spin);
  body.replaceChildren(busy);

  const fd = new FormData();
  fd.append("file", file, file.name);
  try {
    const res = await fetch(API.importVideo, { method: "POST", body: fd });
    const reply = await res.json().catch(() => ({}));
    if (!res.ok || !reply.ok || !reply.job) {
      throw new Error(reply.error || `HTTP ${res.status}`);
    }
    const done = await pollVideoJob(reply.job, spin);
    renderVideoResult(done.result,
      `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`);
  } catch (err) {
    renderImportError(err && err.message ? err.message : "video import failed");
  }
}

async function pollVideoJob(job, spin) {
  // worker-side progress: "frame k / N" until state flips to done/error
  for (;;) {
    if (job.state === "error") throw new Error(job.error || "video analysis failed");
    if (job.state === "done" && job.result) return job;
    spin.textContent = job.total
      ? `Analyzing video — frame ${Math.min(job.done, job.total)} / ${job.total}…`
      : "Analyzing video…";
    await new Promise((resolve) => setTimeout(resolve, 1500));
    const res = await fetch(API.importVideoStatus);
    const reply = await res.json().catch(() => ({}));
    if (!res.ok || !reply.ok || !reply.job) throw new Error("lost the analysis job");
    job = reply.job;
  }
}

function renderImportError(message) {
  const body = $("import-body");
  const box = document.createElement("div");
  box.className = "import-error";
  const msg = document.createElement("p");
  msg.textContent = message;
  const retry = document.createElement("button");
  retry.type = "button";
  retry.textContent = "Choose another file";
  retry.addEventListener("click", () => $("import-file").click());
  box.append(msg, retry);
  body.replaceChildren(box);
}

function renderImportResult(r, meta) {
  // server data goes through textContent only — nothing into innerHTML
  const body = $("import-body");
  const wrap = document.createElement("div");
  wrap.className = "import-result";

  const img = document.createElement("img");
  img.alt = "analyzed shelf photo";
  img.src = "data:image/jpeg;base64," + (r.annotated || "");
  wrap.appendChild(img);

  const stats = document.createElement("div");
  stats.className = "import-stats";
  const count = document.createElement("p");
  count.className = "import-count";
  const items = r.items != null ? r.items : 0;
  count.textContent = `${items} item${items === 1 ? "" : "s"} detected`;
  stats.appendChild(count);

  const chips = document.createElement("div");
  chips.className = "slot-codes";
  const byCategory = r.items_by_category || {};
  const catKeys = Object.keys(byCategory).filter((c) => CATEGORY_COLOR[c]);
  if (catKeys.length) {
    // true per-item identity from the tiled detector — count desc, other last
    catKeys.sort((a, b) =>
      (a === "other") - (b === "other") || byCategory[b] - byCategory[a]);
    for (const cat of catKeys) {
      const chip = document.createElement("span");
      chip.className = "mini-code";
      chip.dataset.category = cat;
      const dot = document.createElement("b");
      const txt = document.createElement("span");
      txt.className = "mini-n";
      txt.textContent = `${cat === "other" ? "other goods" : cat} × ${byCategory[cat]}`;
      chip.append(dot, txt);
      chips.appendChild(chip);
    }
  } else {
    // fallback: cell-position join (no detector / older server payload)
    const byCode = r.items_by_code || {};
    const codes = Object.keys(byCode)
      .filter((c) => CODE_COLOR[c] || c === "other")
      .sort((a, b) => (a === "other") - (b === "other"));
    for (const code of codes) {
      const chip = document.createElement("span");
      chip.className = "mini-code";
      if (CODE_COLOR[code]) chip.dataset.code = code;
      const dot = document.createElement("b");
      const txt = document.createElement("span");
      txt.className = "mini-n";
      txt.textContent = `${code === "other" ? "other goods" : codeLabel(code)} × ${byCode[code]}`;
      chip.append(dot, txt);
      chips.appendChild(chip);
    }
    if (!codes.length) {
      const none = document.createElement("span");
      none.className = "mini-code";
      none.textContent = "no goods recognized";
      chips.appendChild(none);
    }
  }
  stats.appendChild(chips);

  // what the boxes on the annotated photo mean — mirrors the on-frame legend
  const legend = document.createElement("p");
  legend.className = "import-legend";
  legend.textContent =
    "colored boxes = counted items (color = category) · gray boxes = recognized but not counted";
  stats.appendChild(legend);

  const goods = r.goods || {};
  const sub = document.createElement("p");
  sub.className = "import-sub";
  sub.textContent =
    `${goods.occupied != null ? goods.occupied : 0} / ${goods.cells != null ? goods.cells : 0} shelf areas with goods` +
    ` · ${r.width}×${r.height}px${meta ? " · " + meta : ""}`;
  stats.appendChild(sub);

  wrap.appendChild(stats);
  body.replaceChildren(wrap);
}

function renderVideoResult(r, meta) {
  // server data goes through textContent/src only — nothing into innerHTML
  const body = $("import-body");
  const wrap = document.createElement("div");
  wrap.className = "import-result";

  const stats = document.createElement("div");
  stats.className = "import-stats";
  const count = document.createElement("p");
  count.className = "import-count";
  const peak = r.items_peak != null ? r.items_peak : 0;
  const mean = r.items_mean != null ? r.items_mean : 0;
  count.textContent = `peak ${peak} item${peak === 1 ? "" : "s"} at ${r.peak_t}s · average ${mean}`;
  stats.appendChild(count);

  // category chips from the peak frame — same palette as the photo import
  const chips = document.createElement("div");
  chips.className = "slot-codes";
  const byCategory = r.items_by_category || {};
  const catKeys = Object.keys(byCategory).filter((c) => CATEGORY_COLOR[c])
    .sort((a, b) =>
      (a === "other") - (b === "other") || byCategory[b] - byCategory[a]);
  if (catKeys.length) {
    for (const cat of catKeys) {
      const chip = document.createElement("span");
      chip.className = "mini-code";
      chip.dataset.category = cat;
      const dot = document.createElement("b");
      const txt = document.createElement("span");
      txt.className = "mini-n";
      txt.textContent = `${cat === "other" ? "other goods" : cat} × ${byCategory[cat]}`;
      chip.append(dot, txt);
      chips.appendChild(chip);
    }
  } else {
    const none = document.createElement("span");
    none.className = "mini-code";
    none.textContent = "no goods recognized";
    chips.appendChild(none);
  }
  stats.appendChild(chips);

  const legend = document.createElement("p");
  legend.className = "import-legend";
  legend.textContent =
    "categories read at the peak frame · colored boxes = counted items (color = category) · gray = recognized but not counted";
  stats.appendChild(legend);

  const v = r.video || {};
  const sub = document.createElement("p");
  sub.className = "import-sub";
  sub.textContent =
    `${r.sampled} frames sampled over ${v.duration_seconds != null ? v.duration_seconds : 0}s` +
    ` · ${v.width}×${v.height}px @ ${v.fps}fps${meta ? " · " + meta : ""}`;
  stats.appendChild(sub);

  wrap.appendChild(stats);

  const strip = document.createElement("div");
  strip.className = "kf-strip";
  for (const kf of r.keyframes || []) {
    const tile = document.createElement("figure");
    tile.className = "kf-tile";
    const img = document.createElement("img");
    img.alt = `keyframe at ${kf.t}s`;
    img.loading = "lazy";
    img.src = "data:image/jpeg;base64," + (kf.image || "");
    const cap = document.createElement("figcaption");
    cap.textContent = `${kf.t}s · ${kf.items} item${kf.items === 1 ? "" : "s"}`;
    tile.append(img, cap);
    strip.appendChild(tile);
  }
  if (strip.childElementCount) wrap.appendChild(strip);

  // run this upload through the live pipeline (the demo machinery with its
  // source switched to the persisted import). The modal is operator UI; the
  // main view keeps the standard Smooth wording — nothing reveals the
  // "camera" is a file.
  const play = document.createElement("button");
  play.type = "button";
  play.className = "import-play";
  play.textContent = "Play as live feed";
  play.addEventListener("click", async () => {
    play.disabled = true;
    try {
      const res = await fetch(API.demo, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: true, source: "imported" }),
      });
      const reply = await res.json().catch(() => ({}));
      if (!res.ok || !reply.ok) {
        toast(`Playback failed — ${reply.message || "HTTP " + res.status}`,
              "error");
        play.disabled = false;
        return;
      }
      await loadConfig();
      setPreviewMode("smooth");   // same enter-Smooth path as the demo toggle
      closeImport();
      toast("Stock-out + restock alerts armed", "demo");
    } catch {
      toast("Playback failed — server unreachable", "error");
      play.disabled = false;
    }
  });
  wrap.appendChild(play);

  body.replaceChildren(wrap);
}

/* ---------- config fetch + demo-video toggle ---------- */

// (Re-)fetch /api/config and repaint everything derived from it: legends,
// slot/cell count, rack note, grid-map button, demo button. Called once at
// init and again after every demo toggle — the backend swaps this endpoint
// to the demo view (d-prefixed slots + demo grid) while the demo runs.
async function loadConfig() {
  try {
    const res = await fetch(API.config);
    CONFIG = await res.json();
  } catch { return false; }             // offline — keep the previous view
  renderLegends();
  $("stat-slots").textContent = CONFIG.slots.length;
  if (CONFIG.mode === "grid") {
    const lbl = $("stat-slots").parentElement.querySelector("i");
    if (lbl) lbl.textContent = "Cells";   // slots are virtual grid cells
    const note = $("rack-note");
    if (note && CONFIG.grid) {
      note.textContent = `${CONFIG.grid.rows}×${CONFIG.grid.cols} areas · row 1 = top of view`;
    }
    const gm = $("btn-gridmap");
    if (gm) {
      gm.hidden = false;
      gm.textContent = gridMapOn() ? "Hide grid map" : "Show grid map";
      if (!gm.dataset.wired) {           // re-fetches must not double-bind
        gm.dataset.wired = "1";
        gm.addEventListener("click", () => {
          localStorage.setItem("shelfops.gridmap", gridMapOn() ? "0" : "1");
          gm.textContent = gridMapOn() ? "Hide grid map" : "Show grid map";
          renderRack();
        });
      }
    }
  }
  paintDemoButton();
  return true;
}

function paintDemoButton() {
  const btn = $("btn-demo");
  if (!btn) return;
  const d = CONFIG && CONFIG.demo;
  // operator-only control: hidden in the normal UI so a customer facing the
  // screen sees a live shelf, never a "demo" button. Append #demo to the URL
  // to reveal it (hashchange re-checks, so it can be added/removed live).
  // Shown when EITHER the bundled video or a persisted import can play.
  btn.hidden = !(d && (d.available || d.imported_available))
    || location.hash !== "#demo";
  const on = !!(d && d.enabled);
  btn.classList.toggle("active", on);
  btn.textContent = on ? "Stop demo" : "Demo video";
}

// Toggle the built-in demo video through the live pipeline: POST the new
// state, re-fetch the demo view of /api/config (d-prefixed slots + demo
// grid so rack/overlay repaint themselves), and force Smooth preview — the
// MJPEG /stream is where the demo renders (HD is the camera encoder).
async function toggleDemo() {
  const btn = $("btn-demo");
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  try {
    const want = !demoOn();
    const res = await fetch(API.demo, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: want }),
    });
    const reply = await res.json().catch(() => ({}));
    if (!res.ok || !reply.ok) {
      toast(`Demo ${want ? "start" : "stop"} failed — ${reply.message || "HTTP " + res.status}`,
            "error");
      return;
    }
    await loadConfig();
    if (demoOn()) {
      setPreviewMode("smooth");        // enterSmooth sets the standard note —
      // demo must read as live: no "demo" wording anywhere the customer can
      // see (note/toast stay indistinguishable from the normal Smooth view)
      toast("Stock-out + restock alerts armed", "demo");
    } else {
      setNote("Live camera restored");
      toast("Live camera restored", "demo");
    }
  } catch {
    toast("Demo toggle failed — server unreachable", "error");
  } finally {
    btn.disabled = false;
  }
}

/* ---------- init ---------- */
async function init() {
  document.documentElement.classList.add("shelf-ops");

  // heatmap span switcher
  $("heat-span").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    HEATSPAN = btn.dataset.span;
    document.querySelectorAll("#heat-span button").forEach((b) => b.classList.toggle("active", b === btn));
    fetchHeatmap();
  });

  await loadConfig();
  fetchHeatmap(); // paint 6h default grid immediately

  // live preview: prefer HD (hardware H.264 via MSE); the mode buttons stay
  // available for manual switching, and HD auto-degrades to Smooth on flaky
  // links or when no platform token exists (sim mode -> straight to Smooth).
  const pill = document.createElement("div");
  pill.className = "live-pill sim";
  pill.id = "live-pill";
  pill.innerHTML = `<i class="pulse"></i><span>SIM / awaiting device</span>`;
  $("preview-wrap").appendChild(pill);
  $("preview-modes").style.display = "inline-flex";
  $("btn-hd").addEventListener("click", () => setPreviewMode("hd"));
  $("btn-smooth").addEventListener("click", () => setPreviewMode("smooth"));

  // canvas bitmap must match the video's intrinsic size so normalized
  // overlay coords land exactly (object-fit:contain letterboxes both alike)
  const hdVideo = $("hd-video");
  hdVideo.addEventListener("loadedmetadata", () => {
    const cv = $("overlay-canvas");
    const vw = hdVideo.videoWidth || 1920, vh = hdVideo.videoHeight || 1080;
    if (cv.width !== vw || cv.height !== vh) { cv.width = vw; cv.height = vh; }
  });
  (function overlayLoop() { requestAnimationFrame(overlayLoop); drawOverlay(); })();
  setPreviewMode("hd");

  // status pill tracks the scanned status; topbar fps shows the live video
  // rate (measured HD stream fps, or the configured MJPEG fps in Smooth)
  setInterval(() => {
    const status = (LAST && LAST.status) || "simulation";
    const s = STATUS_PILL[status] || ["off", "OFF"];
    const p = $("live-pill");
    p.className = "live-pill " + s[0];
    p.querySelector("span").textContent = (s[1] === "LIVE" ? "LIVE" : s[1]) + " / shelf scan";
    const vf = $("stat-fps");
    if (vf) vf.textContent = PREVIEW_MODE === "hd" && HD
      ? String(Math.round(HD.getFps()))
      : (CONFIG && CONFIG.preview_fps) || "-";
  }, 2000);

  // event stream
  const es = new EventSource(API.events);
  es.onmessage = (e) => {
    if (e.data === "connected") return;
    try {
      const snap = JSON.parse(e.data);
      if (snap.status !== undefined && snap.slots) {
        // scan snapshot — refreshes rack + topbar + HD canvas overlay (LAST)
        LAST = snap;
        paintStatus(snap.status);
        paintGoods();
        renderRack();
        const tf = $("stat-frame");
        if (tf) tf.textContent = snap.frame_seq != null ? snap.frame_seq : "-";
        return;
      }
      if (snap.kind) paintEvent(snap);
    } catch { /* ignore malformed frames */ }
  };
  es.onerror = () => { /* SSE auto-reconnects */ };

  tick();
  setInterval(tick, 1500);
  paintModelTag();
  renderRack();

  // offline photo import: topbar button opens the modal, the dropzone (or a
  // re-run after an error) opens the OS file picker
  $("btn-demo").addEventListener("click", toggleDemo);
  window.addEventListener("hashchange", paintDemoButton);  // #demo reveals it
  $("btn-import").addEventListener("click", () => {
    const modal = $("import-modal");
    modal.hidden = false;
    if (!$("import-drop")) showDrop();   // previous result/error still shown
  });
  $("import-file").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) doImport(file);
  });
  const importModal = $("import-modal");
  $("import-close").addEventListener("click", closeImport);
  importModal.addEventListener("click", (e) => {
    if (e.target === importModal) closeImport();   // backdrop click
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !importModal.hidden) closeImport();
  });
  showDrop();
}

document.addEventListener("DOMContentLoaded", init);