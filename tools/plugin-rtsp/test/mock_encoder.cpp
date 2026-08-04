/**
 * @file mock_encoder.cpp
 * @brief Mock encoded stream publisher for testing RTSP plugin
 *
 * Generates minimal valid H.264 Annex-B NAL units (SPS/PPS/IDR)
 * and publishes them via the same Unix socket protocol as camera-daemon's
 * EncodedPublisher. This allows testing the RTSP plugin without hardware.
 *
 * Usage: ./mock_encoder [socket_path] [fps]
 *   Default: /run/aipc/encoded/main.sock  30
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <csignal>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

static std::atomic<bool> g_running{true};

static void sig_handler(int) { g_running.store(false); }

/* ---- Minimal H.264 bitstream generation ---- */

// Baseline Profile SPS for 320x240 (minimal valid)
static const uint8_t FAKE_SPS[] = {
    0x00, 0x00, 0x00, 0x01,  // start code
    0x67,                     // NAL type 7 = SPS
    0x42, 0xc0, 0x1e,        // profile_idc=66(Baseline), constraint, level=30
    0xd9, 0x00, 0xa0, 0x47,  // seq_parameter (width=320, height=240 encoded)
    0xfe, 0xc8,
};

// Minimal PPS
static const uint8_t FAKE_PPS[] = {
    0x00, 0x00, 0x00, 0x01,  // start code
    0x68,                     // NAL type 8 = PPS
    0xce, 0x38, 0x80,
};

// Generate a fake IDR slice NAL (random-ish data of given size)
static std::vector<uint8_t> make_fake_idr(size_t payload_size, uint32_t frame_num) {
    std::vector<uint8_t> buf;
    // Start code
    buf.push_back(0x00);
    buf.push_back(0x00);
    buf.push_back(0x00);
    buf.push_back(0x01);
    // NAL header: type=5 (IDR), NRI=3
    buf.push_back(0x65);
    // Fake slice data
    for (size_t i = 0; i < payload_size; i++) {
        buf.push_back((uint8_t)((frame_num * 7 + i * 13) & 0xFF));
    }
    return buf;
}

// Generate a fake non-IDR P-frame
static std::vector<uint8_t> make_fake_pframe(size_t payload_size, uint32_t frame_num) {
    std::vector<uint8_t> buf;
    buf.push_back(0x00);
    buf.push_back(0x00);
    buf.push_back(0x00);
    buf.push_back(0x01);
    // NAL header: type=1 (non-IDR), NRI=2
    buf.push_back(0x41);
    for (size_t i = 0; i < payload_size; i++) {
        buf.push_back((uint8_t)((frame_num * 11 + i * 17) & 0xFF));
    }
    return buf;
}

/* ---- Encoded stream protocol (matches EncodedPublisher) ---- */
static constexpr size_t ENC_HEADER_SIZE = 22;

static std::vector<uint8_t> build_packet(const uint8_t* data, size_t size,
                                          bool keyframe, uint64_t ts_ns,
                                          uint32_t width, uint32_t height) {
    uint32_t total = (uint32_t)(ENC_HEADER_SIZE + size);
    std::vector<uint8_t> buf(total);
    uint8_t* p = buf.data();

    // total_size LE
    p[0] = total & 0xFF;
    p[1] = (total >> 8) & 0xFF;
    p[2] = (total >> 16) & 0xFF;
    p[3] = (total >> 24) & 0xFF;

    p[4] = 0;  // codec: h264
    p[5] = keyframe ? 0x01 : 0x00;

    // timestamp LE
    for (int i = 0; i < 8; i++)
        p[6 + i] = (ts_ns >> (i * 8)) & 0xFF;

    // width LE
    p[14] = width & 0xFF;
    p[15] = (width >> 8) & 0xFF;
    p[16] = (width >> 16) & 0xFF;
    p[17] = (width >> 24) & 0xFF;

    // height LE
    p[18] = height & 0xFF;
    p[19] = (height >> 8) & 0xFF;
    p[20] = (height >> 16) & 0xFF;
    p[21] = (height >> 24) & 0xFF;

    memcpy(p + ENC_HEADER_SIZE, data, size);
    return buf;
}

int main(int argc, char** argv) {
    const char* sock_path = argc > 1 ? argv[1] : "/run/aipc/encoded/main.sock";
    int fps = argc > 2 ? atoi(argv[2]) : 30;
    int gop = 30;  // IDR every 30 frames
    uint32_t width = 320, height = 240;

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    printf("Mock Encoder: socket=%s fps=%d gop=%d resolution=%ux%u\n",
           sock_path, fps, gop, width, height);

    // Create directory
    {
        std::string dir(sock_path);
        size_t slash = dir.rfind('/');
        if (slash != std::string::npos) {
            dir = dir.substr(0, slash);
            mkdir(dir.c_str(), 0755);
        }
    }

    // Remove stale socket
    unlink(sock_path);

    // Create listen socket
    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);

    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(listen_fd);
        return 1;
    }
    chmod(sock_path, 0666);

    if (listen(listen_fd, 4) < 0) {
        perror("listen");
        close(listen_fd);
        return 1;
    }

    printf("Mock Encoder: Listening on %s, waiting for RTSP plugin...\n", sock_path);

    while (g_running.load()) {
        // Accept one client
        int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            break;
        }

        printf("Mock Encoder: Client connected (fd=%d)\n", client_fd);

        // Set send buffer
        int sndbuf = 1024 * 1024;
        setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

        auto frame_interval = std::chrono::microseconds(1000000 / fps);
        uint32_t frame_num = 0;
        auto start_time = std::chrono::steady_clock::now();

        while (g_running.load()) {
            auto now = std::chrono::steady_clock::now();
            uint64_t ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - start_time).count();

            bool is_keyframe = (frame_num % gop == 0);

            // Build Annex-B frame
            std::vector<uint8_t> frame_data;

            if (is_keyframe) {
                // SPS + PPS + IDR
                frame_data.insert(frame_data.end(), FAKE_SPS,
                                  FAKE_SPS + sizeof(FAKE_SPS));
                frame_data.insert(frame_data.end(), FAKE_PPS,
                                  FAKE_PPS + sizeof(FAKE_PPS));
                auto idr = make_fake_idr(2000, frame_num);
                frame_data.insert(frame_data.end(), idr.begin(), idr.end());
            } else {
                auto pf = make_fake_pframe(800, frame_num);
                frame_data.insert(frame_data.end(), pf.begin(), pf.end());
            }

            // Build protocol packet
            auto pkt = build_packet(frame_data.data(), frame_data.size(),
                                     is_keyframe, ts_ns, width, height);

            // Send
            ssize_t sent = send(client_fd, pkt.data(), pkt.size(), MSG_NOSIGNAL);
            if (sent < 0) {
                printf("Mock Encoder: Client disconnected\n");
                break;
            }

            if (frame_num % (fps * 5) == 0) {
                printf("Mock Encoder: Frame %u, ts=%.2fs, %s, %zu bytes\n",
                       frame_num, ts_ns / 1e9,
                       is_keyframe ? "IDR" : "P",
                       frame_data.size());
            }

            frame_num++;
            std::this_thread::sleep_for(frame_interval);
        }

        close(client_fd);
    }

    close(listen_fd);
    unlink(sock_path);
    printf("Mock Encoder: Done\n");
    return 0;
}
