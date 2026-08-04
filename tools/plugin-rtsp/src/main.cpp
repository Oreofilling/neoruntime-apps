/**
 * @file main.cpp
 * @brief RTSP Plugin - Standalone RTSP server as a container plugin
 *
 * Reads encoded H.264/H.265 packets from camera-daemon's encoded stream
 * socket and serves them via RTSP to external clients.
 *
 * Environment variables:
 *   RTSP_PORT    - RTSP listen port (default: 8554)
 *   ENCODED_SOCK - Path to encoded stream Unix socket (default: /run/aipc/encoded/main.sock)
 *   STREAM_NAME  - RTSP stream name in URL (default: "main")
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

/* ---- Reuse RtspServer from camera-daemon (linked as library or copy) ---- */
/* For the plugin container, we compile rtsp_server.cpp directly */
#include "rtsp_server.h"

extern "C" {
    #include "hal_log.h"
    #include "hal_codec.h"
}

/* ================================================================
 * Encoded stream protocol (matches camera-daemon EncodedPublisher)
 * ================================================================
 * [4 bytes] total_size (uint32 LE)
 * [1 byte]  codec      (0=h264, 1=h265)
 * [1 byte]  flags      (bit0=keyframe)
 * [8 bytes] timestamp_ns (uint64 LE)
 * [4 bytes] width      (uint32 LE)
 * [4 bytes] height     (uint32 LE)
 * [N bytes] data       (raw Annex-B bitstream)
 */
static constexpr size_t ENC_HEADER_SIZE = 22;
static constexpr uint8_t CTRL_REQUEST_KEYFRAME = 0x4B;  // 'K' — matches EncodedPublisher

static std::atomic<bool> g_running{true};
static RtspServer* g_rtsp = nullptr;
static std::atomic<int> g_encoded_fd{-1};  // Socket fd for sending keyframe requests

static void signal_handler(int) {
    g_running.store(false);
    if (g_rtsp) g_rtsp->stop();
}

static std::string get_env(const char* name, const char* def) {
    const char* val = getenv(name);
    return val ? val : def;
}

/**
 * Read exactly `len` bytes from fd. Returns false on error/EOF.
 */
static bool read_exact(int fd, uint8_t* buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = ::read(fd, buf + got, len - got);
        if (n <= 0) {
            if (n < 0 && (errno == EINTR)) continue;
            return false;
        }
        got += (size_t)n;
    }
    return true;
}

static uint32_t read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)p[i]) << (i * 8);
    return v;
}

/**
 * Connect to the camera-daemon encoded stream socket.
 * Retries with exponential backoff.
 */
static int connect_encoded_socket(const std::string& sock_path) {
    int backoff_ms = 500;
    constexpr int MAX_BACKOFF = 10000;

    while (g_running.load()) {
        int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            HAL_LOG_ERROR("RTSP-Plugin: socket() failed: %s", strerror(errno));
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            backoff_ms = std::min(backoff_ms * 2, MAX_BACKOFF);
            continue;
        }

        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);

        if (::connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            HAL_LOG_WARNING("RTSP-Plugin: connect(%s) failed: %s (retrying in %dms)",
                           sock_path.c_str(), strerror(errno), backoff_ms);
            ::close(fd);
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            backoff_ms = std::min(backoff_ms * 2, MAX_BACKOFF);
            continue;
        }

        HAL_LOG_INFO("RTSP-Plugin: Connected to encoded stream: %s", sock_path.c_str());

        // Set receive buffer
        int rcvbuf = 2 * 1024 * 1024;
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

        return fd;
    }

    return -1;
}

/**
 * Main receive loop: read encoded packets from socket, feed to RtspServer.
 */
static void receive_loop(const std::string& sock_path,
                          RtspServer& rtsp,
                          const std::string& stream_name) {
    std::vector<uint8_t> pkt_buf;

    while (g_running.load()) {
        int fd = connect_encoded_socket(sock_path);
        if (fd < 0) break;

        g_encoded_fd.store(fd);
        HAL_LOG_INFO("RTSP-Plugin: Starting receive loop for stream '%s'", stream_name.c_str());

        while (g_running.load()) {
            // Read header (first 4 bytes = total_size)
            uint8_t hdr[ENC_HEADER_SIZE];
            if (!read_exact(fd, hdr, 4)) {
                HAL_LOG_WARNING("RTSP-Plugin: Connection lost, reconnecting...");
                break;
            }

            uint32_t total_size = read_u32_le(hdr);
            if (total_size < ENC_HEADER_SIZE || total_size > 4 * 1024 * 1024) {
                HAL_LOG_ERROR("RTSP-Plugin: Invalid packet size: %u", total_size);
                break;
            }

            // Read rest of header
            if (!read_exact(fd, hdr + 4, ENC_HEADER_SIZE - 4)) {
                break;
            }

            uint8_t codec = hdr[4];
            uint8_t flags = hdr[5];
            uint64_t timestamp_ns = read_u64_le(hdr + 6);
            // uint32_t width = read_u32_le(hdr + 14);  // Available if needed
            // uint32_t height = read_u32_le(hdr + 18);

            // Read payload
            uint32_t payload_size = total_size - ENC_HEADER_SIZE;
            pkt_buf.resize(payload_size);
            if (!read_exact(fd, pkt_buf.data(), payload_size)) {
                break;
            }

            // Build HalPacket and feed to RTSP server
            HalPacket packet{};
            packet.data = pkt_buf.data();
            packet.size = payload_size;
            packet.is_keyframe = (flags & 0x01) != 0;
            packet.timestamp_ns = timestamp_ns;

            rtsp.on_packet(stream_name, &packet);
        }

        g_encoded_fd.store(-1);
        ::close(fd);

        if (g_running.load()) {
            HAL_LOG_INFO("RTSP-Plugin: Disconnected, will reconnect...");
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

int main(int argc, char** argv) {
    // Setup logging
    hal_log_set_level(HAL_LOG_LEVEL_INFO);
    hal_log_set_color(1);
    hal_log_set_timestamp(1);

    HAL_LOG_INFO("====================================");
    HAL_LOG_INFO("AIPC RTSP Plugin v1.0.0");
    HAL_LOG_INFO("====================================");

    // Read config from environment
    uint16_t rtsp_port = (uint16_t)std::stoul(get_env("RTSP_PORT", "8554"));
    std::string encoded_sock = get_env("ENCODED_SOCK", "/run/aipc/encoded/main.sock");
    std::string stream_name = get_env("STREAM_NAME", "main");
    std::string codec = get_env("CODEC", "h264");
    uint32_t width = std::stoul(get_env("WIDTH", "1920"));
    uint32_t height = std::stoul(get_env("HEIGHT", "1080"));
    uint32_t fps = std::stoul(get_env("FPS", "30"));

    HAL_LOG_INFO("Config: port=%d, socket=%s, stream=%s, codec=%s, %ux%u@%u",
                rtsp_port, encoded_sock.c_str(), stream_name.c_str(),
                codec.c_str(), width, height, fps);

    // Signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create RTSP server
    RtspServer rtsp;
    g_rtsp = &rtsp;

    RtspServer::StreamInfo si;
    si.name = stream_name;
    si.codec = codec;
    si.width = width;
    si.height = height;
    si.fps = fps;
    rtsp.add_stream(si);

    // Wire keyframe request: when PLAY starts, send control byte to camera-daemon
    rtsp.set_keyframe_request_cb(
        [](const std::string& sname) {
            int fd = g_encoded_fd.load();
            if (fd >= 0) {
                uint8_t ctrl = CTRL_REQUEST_KEYFRAME;
                ssize_t n = ::send(fd, &ctrl, 1, MSG_NOSIGNAL | MSG_DONTWAIT);
                if (n == 1) {
                    HAL_LOG_INFO("RTSP-Plugin: Sent keyframe request for stream '%s'",
                                sname.c_str());
                }
            }
        });

    if (!rtsp.start(rtsp_port)) {
        HAL_LOG_ERROR("Failed to start RTSP server on port %d", rtsp_port);
        return 1;
    }

    HAL_LOG_INFO("RTSP server started: rtsp://<device>:%d/%s", rtsp_port, stream_name.c_str());

    // Start receive loop (blocking until shutdown)
    receive_loop(encoded_sock, rtsp, stream_name);

    // Cleanup
    rtsp.stop();
    g_rtsp = nullptr;

    HAL_LOG_INFO("RTSP Plugin exited");
    return 0;
}
