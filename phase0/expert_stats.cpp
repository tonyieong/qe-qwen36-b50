// expert_stats: run text files through a MoE model and record which experts the router picks.
// usage: expert_stats <model.gguf> <out.bin> <file1.txt> [file2.txt ...]
// out.bin: header {int32 n_layer, int32 n_expert, int32 n_used}, then per token per layer n_used uint8 expert ids
//          (records are appended in eval order: for each ubatch, layer by layer, token-major inside a layer).
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct stats_ctx {
    FILE * out = nullptr;
    int n_used = 0;
    std::vector<int32_t> buf;
    long long n_records = 0;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * sc = (stats_ctx *) user_data;
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) {
        return false;
    }
    if (ask) {
        return true;
    }
    const int il = atoi(t->name + 13);
    const int64_t n_used = t->ne[0];
    const int64_t n_tok = t->ne[1];
    sc->buf.resize(n_used * n_tok);
    // topk is a view into argsort; copy row by row honoring strides
    std::vector<int32_t> row(t->ne[0]);
    for (int64_t j = 0; j < n_tok; ++j) {
        ggml_backend_tensor_get(t, row.data(), j * t->nb[1], n_used * sizeof(int32_t));
        memcpy(sc->buf.data() + j * n_used, row.data(), n_used * sizeof(int32_t));
    }
    // record: int16 layer, int32 n_tok, then n_tok * n_used uint8
    int16_t l16 = (int16_t) il;
    int32_t nt = (int32_t) n_tok;
    fwrite(&l16, sizeof(l16), 1, sc->out);
    fwrite(&nt, sizeof(nt), 1, sc->out);
    std::vector<uint8_t> ids(sc->buf.size());
    for (size_t k = 0; k < ids.size(); ++k) {
        ids[k] = (uint8_t) sc->buf[k];
    }
    fwrite(ids.data(), 1, ids.size(), sc->out);
    sc->n_records++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf out.bin file1.txt [...]\n", argv[0]);
        return 1;
    }
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    stats_ctx sc;
    sc.out = fopen(argv[2], "wb");

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 8192;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = &sc;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        return 1;
    }

    long long total_tok = 0;
    for (int fi = 3; fi < argc; ++fi) {
        std::ifstream f(argv[fi]);
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string text = ss.str();
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), false, true);
        if (n < 0) {
            continue;
        }
        toks.resize(std::min(n, 8000));
        llama_memory_clear(llama_get_memory(ctx), true);
        for (size_t i = 0; i < toks.size(); i += 512) {
            const int nb = std::min<size_t>(512, toks.size() - i);
            llama_batch b = llama_batch_get_one(toks.data() + i, nb);
            if (llama_decode(ctx, b) != 0) {
                fprintf(stderr, "decode failed on %s\n", argv[fi]);
                break;
            }
        }
        total_tok += toks.size();
        fprintf(stderr, "%s: %zu tokens (total %lld)\n", argv[fi], toks.size(), total_tok);
    }
    fclose(sc.out);
    fprintf(stderr, "done: %lld tokens, %lld layer records\n", total_tok, sc.n_records);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
