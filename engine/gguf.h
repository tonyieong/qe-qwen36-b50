// minimal GGUF v3 reader: metadata is skipped except general.alignment, tensors are mmapped
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

struct GTensor {
    std::string name;
    int64_t ne[4] = {1, 1, 1, 1};
    uint32_t type = 0;   // ggml type id: 0 F32, 1 F16, 8 Q8_0
    uint64_t off = 0;
    const uint8_t * data = nullptr;
};

struct GGUF {
    std::vector<GTensor> tensors;
    std::unordered_map<std::string, size_t> idx;
    uint8_t * base = nullptr;
    size_t size = 0;

    explicit GGUF(const std::string & path) {
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("open " + path);
        struct stat st;
        fstat(fd, &st);
        size = st.st_size;
        base = (uint8_t *) mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (base == MAP_FAILED) throw std::runtime_error("mmap " + path);
        const uint8_t * p = base;
        auto rd = [&](void * dst, size_t n) { memcpy(dst, p, n); p += n; };
        auto u32 = [&] { uint32_t v; rd(&v, 4); return v; };
        auto u64 = [&] { uint64_t v; rd(&v, 8); return v; };
        auto str = [&] { const uint64_t n = u64(); std::string s((const char *) p, n); p += n; return s; };
        if (memcmp(p, "GGUF", 4) != 0) throw std::runtime_error("not a GGUF file");
        p += 4;
        u32();  // version
        const uint64_t n_t = u64(), n_kv = u64();
        uint32_t alignment = 32;
        const size_t tsize[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        std::function<void(uint32_t)> skip = [&](uint32_t t) {
            if (t == 8) { str(); return; }
            if (t == 9) { const uint32_t at = u32(); const uint64_t n = u64(); for (uint64_t i = 0; i < n; ++i) skip(at); return; }
            p += tsize[t];
        };
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = str();
            const uint32_t t = u32();
            if (key == "general.alignment" && t == 4) { alignment = u32(); continue; }
            skip(t);
        }
        for (uint64_t i = 0; i < n_t; ++i) {
            GTensor gt;
            gt.name = str();
            const uint32_t nd = u32();
            for (uint32_t d = 0; d < nd; ++d) gt.ne[d] = (int64_t) u64();
            gt.type = u32();
            gt.off = u64();
            idx[gt.name] = tensors.size();
            tensors.push_back(gt);
        }
        size_t data0 = (size_t) (p - base);
        data0 = (data0 + alignment - 1) / alignment * alignment;
        for (auto & t : tensors) t.data = base + data0 + t.off;
    }
    const GTensor * find(const std::string & n) const {
        auto it = idx.find(n);
        return it == idx.end() ? nullptr : &tensors[it->second];
    }
    const GTensor & get(const std::string & n) const {
        const GTensor * t = find(n);
        if (!t) throw std::runtime_error("missing tensor " + n);
        return *t;
    }
};
