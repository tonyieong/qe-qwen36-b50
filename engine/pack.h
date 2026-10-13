// engine weight pack: formats, model constants, file layout
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace qe {

// ---- Qwen3.6-35B-A3B (qwen35moe) ----
constexpr int D = 2048, N_LAYER = 40, VOCAB = 248320;
constexpr int N_EXPERT = 256, K_USED = 8, FF_EXP = 512, FF_SH = 512;
constexpr int DK = 128, NK = 16, NV = 32;                // gated delta net: head dim, k heads, v heads
constexpr int QKV_DIM = DK * NK * 2 + DK * NV;           // 8192 (q | k | v)
constexpr int Z_DIM = DK * NV;                           // 4096
constexpr int CONV_K = 4;
constexpr int HD = 256, NQ = 16, NKV = 2, N_ROT = 64;    // full attention
constexpr int Q_ROWS = HD * NQ * 2;                      // per head [q | gate]
constexpr float RMS_EPS = 1e-6f, ROPE_BASE = 10000000.f;
inline bool is_recurrent(int il) { return (il + 1) % 4 != 0; }

// ---- weight formats; every quantized row = [quant data][fp16 scale per 32 weights] ----
enum Fmt : uint32_t { F_F32 = 0, F_Q8 = 1, F_Q6 = 2, F_NL4 = 3 };
//  Q8 : int8, element k at byte k
//  Q6 : (q+32) in 6 bits; low-nibble plane cols/2 bytes (element 2k low / 2k+1 high nibble of byte k),
//       then 2-bit high plane cols/4 bytes (element k at bits 2k..2k+1 of the block's 64-bit word)
//  NL4: IQ4_NL indices; per 32-block 16 bytes, element j<16 low nibble of byte j, element j+16 high nibble
inline size_t qbytes(Fmt f, int cols) {
    switch (f) {
        case F_Q8: return (size_t) cols;
        case F_Q6: return (size_t) cols / 2 + cols / 4;
        case F_NL4: return (size_t) cols / 2;
        default: return (size_t) cols * 4;
    }
}
inline size_t row_bytes(Fmt f, int cols) { return f == F_F32 ? (size_t) cols * 4 : qbytes(f, cols) + (size_t) (cols / 32) * 2; }
static const int8_t KVALUES_NL4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// one expert = gate rows (FF_EXP x D), up rows (FF_EXP x D), down rows (D x FF_EXP), all NL4
inline size_t exp_gu_rb() { return row_bytes(F_NL4, D); }
inline size_t exp_d_rb() { return row_bytes(F_NL4, FF_EXP); }
inline size_t exp_bytes() { return 2 * FF_EXP * exp_gu_rb() + (size_t) D * exp_d_rb(); }

// ---- file layout ----
struct PackEntry {
    char name[96];
    uint32_t fmt, rows, cols, n;   // n = number of matrices stacked (experts per layer), else 1
    uint64_t off, bytes;
};
struct PackHeader {
    char magic[8];                 // "QE36PK01"
    uint32_t n_entries, pad;
    uint64_t table_off;
};

}  // namespace qe
