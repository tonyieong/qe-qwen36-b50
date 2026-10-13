// qconv: convert Qwen3.6-35B-A3B Q8_0 GGUF into the engine pack
//   attention / shared experts: Q8 (exact repack of Q8_0), gated-delta-net projections + lm head: Q6,
//   routed experts: IQ4_NL (ggml's quantizer), norms / router / small tensors: F32
// usage: qconv <model-Q8_0.gguf> <out.pack> [threads]
//        qconv --mtp <mtp-Q8_0.gguf> <out.pack> [threads]   (MTP / nextn block only, names prefixed "mtp.")
#include "gguf.h"
#include "pack.h"
#include "ggml.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <map>

using namespace qe;

static std::vector<float> dequant_rows(const GTensor & t, int64_t row0, int64_t nrows) {
    const int64_t cols = t.ne[0];
    std::vector<float> out((size_t) nrows * cols);
    if (t.type == 0) {
        memcpy(out.data(), (const float *) t.data + row0 * cols, out.size() * 4);
    } else if (t.type == 1) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) t.data + row0 * cols;
        for (size_t i = 0; i < out.size(); ++i) out[i] = ggml_fp16_to_fp32(s[i]);
    } else if (t.type == 8) {
        const size_t rb = (size_t) cols / 32 * 34;
        for (int64_t r = 0; r < nrows; ++r) {
            const uint8_t * row = t.data + (size_t) (row0 + r) * rb;
            for (int64_t b = 0; b < cols / 32; ++b) {
                const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) (row + b * 34));
                const int8_t * q = (const int8_t *) (row + b * 34 + 2);
                for (int i = 0; i < 32; ++i) out[(size_t) r * cols + b * 32 + i] = d * q[i];
            }
        }
    } else {
        throw std::runtime_error("unsupported source type for " + t.name);
    }
    return out;
}

// one row of floats -> engine format
static void quant_row(Fmt f, const float * x, int cols, uint8_t * dst) {
    const int nb = cols / 32;
    ggml_fp16_t * sc = (ggml_fp16_t *) (dst + qbytes(f, cols));
    if (f == F_F32) { memcpy(dst, x, (size_t) cols * 4); return; }
    if (f == F_Q8) {
        for (int b = 0; b < nb; ++b) {
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
            const ggml_fp16_t dh = ggml_fp32_to_fp16(amax / 127.f);
            const float d = ggml_fp16_to_fp32(dh), id = d > 0 ? 1.f / d : 0.f;
            sc[b] = dh;
            for (int i = 0; i < 32; ++i) ((int8_t *) dst)[b * 32 + i] = (int8_t) std::max(-127.f, std::min(127.f, std::round(x[b * 32 + i] * id)));
        }
        return;
    }
    if (f == F_Q6) {
        uint8_t * lo = dst, * hi = dst + cols / 2;
        for (int b = 0; b < nb; ++b) {
            const float * xb = x + b * 32;
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
            float best_err = INFINITY; ggml_fp16_t best_d = ggml_fp32_to_fp16(0.f);
            for (int s = -8; s <= 8 && amax > 0; ++s) {
                const ggml_fp16_t dh = ggml_fp32_to_fp16(amax / (31.f + 0.25f * s));
                const float d = ggml_fp16_to_fp32(dh), id = 1.f / d;
                float err = 0.f;
                for (int i = 0; i < 32; ++i) {
                    const float q = std::max(-32.f, std::min(31.f, std::round(xb[i] * id)));
                    err += (xb[i] - q * d) * (xb[i] - q * d);
                }
                if (err < best_err) { best_err = err; best_d = dh; }
            }
            sc[b] = best_d;
            const float d = ggml_fp16_to_fp32(best_d), id = d > 0 ? 1.f / d : 0.f;
            uint64_t hb = 0;
            for (int i = 0; i < 32; ++i) {
                const int q = (int) std::max(-32.f, std::min(31.f, std::round(xb[i] * id))) + 32;
                uint8_t & L = lo[b * 16 + i / 2];
                if (i % 2 == 0) L = (uint8_t) (q & 0xf); else L |= (uint8_t) ((q & 0xf) << 4);
                hb |= (uint64_t) ((q >> 4) & 0x3) << (2 * i);
            }
            memcpy(hi + b * 8, &hb, 8);
        }
        return;
    }
    // NL4 via ggml IQ4_NL (block = fp16 d + 16 bytes)
    std::vector<uint8_t> tmp((size_t) nb * 18);
    ggml_quantize_chunk(GGML_TYPE_IQ4_NL, x, tmp.data(), 0, 1, cols, nullptr);
    for (int b = 0; b < nb; ++b) {
        memcpy(&sc[b], tmp.data() + b * 18, 2);
        memcpy(dst + b * 16, tmp.data() + b * 18 + 2, 16);
    }
}

// exact repack of Q8_0 rows into Q8 SoA
static void repack_q8(const GTensor & t, int64_t row0, int64_t nrows, uint8_t * dst) {
    const int cols = (int) t.ne[0];
    const size_t srb = (size_t) cols / 32 * 34, drb = row_bytes(F_Q8, cols);
    for (int64_t r = 0; r < nrows; ++r) {
        const uint8_t * s = t.data + (size_t) (row0 + r) * srb;
        uint8_t * d = dst + (size_t) r * drb;
        for (int b = 0; b < cols / 32; ++b) {
            memcpy(d + qbytes(F_Q8, cols) + b * 2, s + b * 34, 2);
            memcpy(d + b * 32, s + b * 34 + 2, 32);
        }
    }
}

struct Writer {
    FILE * f;
    std::vector<PackEntry> entries;
    uint64_t pos = 0;
    explicit Writer(const char * path) {
        f = fopen(path, "wb");
        PackHeader h{};
        fwrite(&h, sizeof(h), 1, f);
        pos = sizeof(h);
        align();
    }
    void align() {
        static const uint8_t z[4096] = {0};
        const uint64_t a = (pos + 4095) / 4096 * 4096;
        fwrite(z, 1, a - pos, f);
        pos = a;
    }
    void add(const std::string & name, Fmt fmt, int rows, int cols, int n, const std::vector<uint8_t> & data) {
        PackEntry e{};
        snprintf(e.name, sizeof(e.name), "%s", name.c_str());
        e.fmt = fmt; e.rows = rows; e.cols = cols; e.n = n; e.off = pos; e.bytes = data.size();
        fwrite(data.data(), 1, data.size(), f);
        pos += data.size();
        align();
        entries.push_back(e);
    }
    void finish() {
        PackHeader h{};
        memcpy(h.magic, "QE36PK01", 8);
        h.n_entries = (uint32_t) entries.size();
        h.table_off = pos;
        fwrite(entries.data(), sizeof(PackEntry), entries.size(), f);
        fseek(f, 0, SEEK_SET);
        fwrite(&h, sizeof(h), 1, f);
        fclose(f);
    }
};

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s [--mtp] model-Q8_0.gguf out.pack [threads]\n", argv[0]); return 1; }
    const bool mtp = std::string(argv[1]) == "--mtp";
    if (mtp) { ++argv; --argc; }
    const int nthr = argc > 3 ? atoi(argv[3]) : 16;
    GGUF g(argv[1]);
    Writer w(argv[2]);
    // dense format overrides: QE_DN (delta-net qkv/z/out), QE_ATT (attention q/k/v/o), QE_SH (shared expert), QE_LM (lm head)
    auto fmt_env = [](const char * name, Fmt def) {
        const char * v = getenv(name);
        if (!v) return def;
        const std::string sv = v;
        return sv == "q8" ? F_Q8 : sv == "q6" ? F_Q6 : sv == "nl4" ? F_NL4 : def;
    };
    const Fmt f_dn = fmt_env("QE_DN", F_Q6), f_att = fmt_env("QE_ATT", F_Q8), f_sh = fmt_env("QE_SH", F_Q8), f_lm = fmt_env("QE_LM", F_Q6);
    // QE_EXPERTS_FROM=<pack>: copy the routed-expert blobs instead of re-quantizing them
    const char * ef = getenv("QE_EXPERTS_FROM");
    uint8_t * ebase = nullptr; std::map<std::string, PackEntry> eentries;
    if (ef) {
        const int fd = open(ef, O_RDONLY); struct stat st; fstat(fd, &st);
        ebase = (uint8_t *) mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0); close(fd);
        const PackHeader * h = (const PackHeader *) ebase;
        const PackEntry * t = (const PackEntry *) (ebase + h->table_off);
        for (uint32_t i = 0; i < h->n_entries; ++i) eentries[t[i].name] = t[i];
    }
    ggml_quantize_init(GGML_TYPE_IQ4_NL);

    // generic: whole 2-D tensor -> format (row-parallel)
    auto conv = [&](const std::string & src, const std::string & dst, Fmt f) {
        const GTensor & t = g.get(src);
        const int cols = (int) t.ne[0], rows = (int) (t.ne[1] * t.ne[2]);
        std::vector<uint8_t> out((size_t) rows * row_bytes(f, cols));
        if (f == F_Q8 && t.type == 8) {
            repack_q8(t, 0, rows, out.data());
        } else {
            std::atomic<int> next{0};
            std::vector<std::thread> th;
            for (int k = 0; k < nthr; ++k) th.emplace_back([&] {
                for (int r0; (r0 = next.fetch_add(256)) < rows;) {
                    const int nr = std::min(256, rows - r0);
                    const std::vector<float> x = dequant_rows(t, r0, nr);
                    for (int r = 0; r < nr; ++r) quant_row(f, x.data() + (size_t) r * cols, cols, out.data() + (size_t) (r0 + r) * row_bytes(f, cols));
                }
            });
            for (auto & t2 : th) t2.join();
        }
        w.add(dst, f, rows, cols, 1, out);
    };

    // routed experts -> one blob per layer, expert-major, each expert = gate | up | down (NL4)
    auto convert_experts = [&](const std::string & p, const std::string & dst_name) {
        if (ebase && eentries.count(dst_name)) {
            const PackEntry & e = eentries[dst_name];
            w.add(dst_name, F_NL4, 0, 0, N_EXPERT, std::vector<uint8_t>(ebase + e.off, ebase + e.off + e.bytes));
            return;
        }
        const GTensor & tg = g.get(p + "ffn_gate_exps.weight");
        const GTensor & tu = g.get(p + "ffn_up_exps.weight");
        const GTensor & td = g.get(p + "ffn_down_exps.weight");
        std::vector<uint8_t> blob((size_t) N_EXPERT * exp_bytes());
        std::atomic<int> next{0};
        std::vector<std::thread> th;
        for (int k = 0; k < nthr; ++k) th.emplace_back([&] {
            for (int e; (e = next.fetch_add(1)) < N_EXPERT;) {
                uint8_t * dst = blob.data() + (size_t) e * exp_bytes();
                const std::vector<float> xg = dequant_rows(tg, (int64_t) e * FF_EXP, FF_EXP);
                const std::vector<float> xu = dequant_rows(tu, (int64_t) e * FF_EXP, FF_EXP);
                const std::vector<float> xd = dequant_rows(td, (int64_t) e * D, D);
                for (int r = 0; r < FF_EXP; ++r) quant_row(F_NL4, xg.data() + (size_t) r * D, D, dst + (size_t) r * exp_gu_rb());
                for (int r = 0; r < FF_EXP; ++r) quant_row(F_NL4, xu.data() + (size_t) r * D, D, dst + (size_t) (FF_EXP + r) * exp_gu_rb());
                uint8_t * dd = dst + 2 * (size_t) FF_EXP * exp_gu_rb();
                for (int r = 0; r < D; ++r) quant_row(F_NL4, xd.data() + (size_t) r * FF_EXP, FF_EXP, dd + (size_t) r * exp_d_rb());
            }
        });
        for (auto & t2 : th) t2.join();
        w.add(dst_name, F_NL4, 0, 0, N_EXPERT, blob);
    };

    if (mtp) {
        const std::string p = "blk.40.";
        conv(p + "nextn.eh_proj.weight", "mtp.eh_proj", F_Q8);
        conv(p + "nextn.enorm.weight", "mtp.enorm", F_F32);
        conv(p + "nextn.hnorm.weight", "mtp.hnorm", F_F32);
        conv(p + "nextn.shared_head_norm.weight", "mtp.head_norm", F_F32);
        conv(p + "attn_norm.weight", "mtp.attn_norm", F_F32);
        conv(p + "post_attention_norm.weight", "mtp.post_norm", F_F32);
        conv(p + "attn_q.weight", "mtp.wq", F_Q8);
        conv(p + "attn_k.weight", "mtp.wk", F_Q8);
        conv(p + "attn_v.weight", "mtp.wv", F_Q8);
        conv(p + "attn_output.weight", "mtp.wo", F_Q8);
        conv(p + "attn_q_norm.weight", "mtp.q_norm", F_F32);
        conv(p + "attn_k_norm.weight", "mtp.k_norm", F_F32);
        conv(p + "ffn_gate_inp.weight", "mtp.router", F_F32);
        conv(p + "ffn_gate_inp_shexp.weight", "mtp.sh_gate", F_F32);
        conv(p + "ffn_gate_shexp.weight", "mtp.sh_gate_proj", F_Q8);
        conv(p + "ffn_up_shexp.weight", "mtp.sh_up", F_Q8);
        conv(p + "ffn_down_shexp.weight", "mtp.sh_down", F_Q8);
        convert_experts(p, "mtp.experts");
        w.finish();
        fprintf(stderr, "mtp pack written: %s\n", argv[2]);
        return 0;
    }

    conv("token_embd.weight", "token_embd", F_Q8);
    conv("output_norm.weight", "output_norm", F_F32);
    conv("output.weight", "output", f_lm);
    fprintf(stderr, "global tensors done\n");

    for (int il = 0; il < N_LAYER; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        conv(p + "attn_norm.weight", p + "attn_norm", F_F32);
        conv(p + "post_attention_norm.weight", p + "post_norm", F_F32);
        conv(p + "ffn_gate_inp.weight", p + "router", F_F32);
        conv(p + "ffn_gate_inp_shexp.weight", p + "sh_gate", F_F32);
        conv(p + "ffn_gate_shexp.weight", p + "sh_gate_proj", f_sh);
        conv(p + "ffn_up_shexp.weight", p + "sh_up", f_sh);
        conv(p + "ffn_down_shexp.weight", p + "sh_down", f_sh);
        if (is_recurrent(il)) {
            conv(p + "attn_qkv.weight", p + "qkv", f_dn);
            conv(p + "attn_gate.weight", p + "z", f_dn);
            conv(p + "ssm_conv1d.weight", p + "conv", F_F32);
            conv(p + "ssm_dt.bias", p + "dt_bias", F_F32);
            conv(p + "ssm_a", p + "ssm_a", F_F32);
            conv(p + "ssm_alpha.weight", p + "alpha", F_F32);
            conv(p + "ssm_beta.weight", p + "beta", F_F32);
            conv(p + "ssm_norm.weight", p + "ssm_norm", F_F32);
            conv(p + "ssm_out.weight", p + "ssm_out", f_dn);
        } else {
            conv(p + "attn_q.weight", p + "wq", f_att);
            conv(p + "attn_k.weight", p + "wk", f_att);
            conv(p + "attn_v.weight", p + "wv", f_att);
            conv(p + "attn_output.weight", p + "wo", f_att);
            conv(p + "attn_q_norm.weight", p + "q_norm", F_F32);
            conv(p + "attn_k_norm.weight", p + "k_norm", F_F32);
        }
        convert_experts(p, p + "experts");
        fprintf(stderr, "layer %d done\n", il);
    }
    w.finish();
    fprintf(stderr, "pack written: %s\n", argv[2]);
    return 0;
}
#if 0
    {
        const GTensor & tg = g.get(p + "ffn_gate_exps.weight");
        const GTensor & tu = g.get(p + "ffn_up_exps.weight");
        const GTensor & td = g.get(p + "ffn_down_exps.weight");
        std::vector<uint8_t> blob((size_t) N_EXPERT * exp_bytes());
        std::atomic<int> next{0};
        std::vector<std::thread> th;
        for (int k = 0; k < nthr; ++k) th.emplace_back([&] {
            for (int e; (e = next.fetch_add(1)) < N_EXPERT;) {
                uint8_t * dst = blob.data() + (size_t) e * exp_bytes();
                const std::vector<float> xg = dequant_rows(tg, (int64_t) e * FF_EXP, FF_EXP);
                const std::vector<float> xu = dequant_rows(tu, (int64_t) e * FF_EXP, FF_EXP);
                const std::vector<float> xd = dequant_rows(td, (int64_t) e * D, D);
                for (int r = 0; r < FF_EXP; ++r) quant_row(F_NL4, xg.data() + (size_t) r * D, D, dst + (size_t) r * exp_gu_rb());
                for (int r = 0; r < FF_EXP; ++r) quant_row(F_NL4, xu.data() + (size_t) r * D, D, dst + (size_t) (FF_EXP + r) * exp_gu_rb());
                uint8_t * dd = dst + 2 * (size_t) FF_EXP * exp_gu_rb();
                for (int r = 0; r < D; ++r) quant_row(F_NL4, xd.data() + (size_t) r * FF_EXP, FF_EXP, dd + (size_t) r * exp_d_rb());
            }
        });
        for (auto & t2 : th) t2.join();
    }
#endif
