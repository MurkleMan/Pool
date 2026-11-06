// pool-rpc-server: lends this PC's graphics card to stable-diffusion.cpp on
// another PC. Built from the same ggml as sd-cli, so the two always match.
// Flags follow llama.cpp's ggml-rpc-server so the same commands work for either:
//   -H host (default 127.0.0.1)  -p port (default 50052)
//   -c  keep received model parts on disk (folder from GGML_RPC_CACHE or LLAMA_CACHE)
//   -d  dev1,dev2  devices to lend (default: every graphics card, else the CPU)
//   -t  threads for the CPU device
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) end = s.size();
        if (end > start) out.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string host = "127.0.0.1", port = "50052", devs;
    bool cache = false;
    size_t threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", name); exit(1); }
            return argv[++i];
        };
        if (a == "-H" || a == "--host") host = next("-H");
        else if (a == "-p" || a == "--port") port = next("-p");
        else if (a == "-d" || a == "--device") devs = next("-d");
        else if (a == "-t" || a == "--threads") threads = (size_t) atoi(next("-t").c_str());
        else if (a == "-c" || a == "--cache") cache = true;
        else if (a == "-h" || a == "--help") {
            printf("usage: %s [-H host] [-p port] [-c] [-d dev1,dev2] [-t threads]\n", argv[0]);
            return 0;
        } else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }

    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> devices;
    if (!devs.empty()) {
        for (auto & name : split(devs)) {
            ggml_backend_dev_t d = ggml_backend_dev_by_name(name.c_str());
            if (!d) { fprintf(stderr, "unknown device %s\n", name.c_str()); return 1; }
            devices.push_back(d);
        }
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) devices.push_back(d);
        }
        if (devices.empty()) {
            ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu) devices.push_back(cpu);
        }
    }
    if (devices.empty()) { fprintf(stderr, "no device to lend\n"); return 1; }

    std::string cache_dir;
    if (cache) {
        const char * base = getenv("GGML_RPC_CACHE");
        if (!base) base = getenv("LLAMA_CACHE");
        if (base) {
            cache_dir = (std::filesystem::path(base) / "rpc-sd").string();
            std::error_code ec;
            std::filesystem::create_directories(cache_dir, ec);
        }
    }

    std::string endpoint = host + ":" + port;
    printf("pool-rpc-server (RPC %d.%d.%d) on %s\n", RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION, RPC_PROTO_PATCH_VERSION, endpoint.c_str());
    for (auto d : devices) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(d, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d), total >> 20, free >> 20);
    }
    if (!cache_dir.empty()) printf("  cache: %s\n", cache_dir.c_str());
    fflush(stdout);
    ggml_backend_rpc_start_server(endpoint.c_str(), cache_dir.empty() ? nullptr : cache_dir.c_str(), threads, devices.size(), devices.data());
    return 0;
}
