// ref_dump: llama.cpp reference for the engine — logits of every prompt position + greedy continuation
// usage: ref_dump <model.gguf> <out.bin> <prompt-file> [n_gen]
// out.bin: int32 n_tok, int32 ids[n_tok], float logits[n_tok][VOCAB], int32 n_gen, int32 gen[n_gen]
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf out.bin prompt.txt [n_gen]\n", argv[0]); return 1; }
    const int n_gen = argc > 4 ? atoi(argv[4]) : 32;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::ifstream fin(argv[3]);
    std::stringstream ss; ss << fin.rdbuf();
    const std::string text = ss.str();
    std::vector<llama_token> toks(text.size() + 16);
    const int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), false, true);
    toks.resize(n);

    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 4096; cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = true;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }

    FILE * f = fopen(argv[2], "wb");
    fwrite(&n, 4, 1, f);
    fwrite(toks.data(), 4, n, f);
    for (int i = 0; i < n; ++i) fwrite(llama_get_logits_ith(ctx, i), 4, n_vocab, f);

    std::vector<int> gen;
    llama_token last = 0;
    {
        const float * lg = llama_get_logits_ith(ctx, n - 1);
        int best = 0; for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[best]) best = v;
        last = best;
    }
    for (int k = 0; k < n_gen; ++k) {
        gen.push_back(last);
        llama_batch one = llama_batch_get_one(&last, 1);
        if (llama_decode(ctx, one)) break;
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0; for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[best]) best = v;
        last = best;
    }
    const int ng = (int) gen.size();
    fwrite(&ng, 4, 1, f);
    fwrite(gen.data(), 4, ng, f);
    fclose(f);
    std::string out;
    for (int t : gen) { char buf[256]; const int l = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true); out.append(buf, std::max(0, l)); }
    fprintf(stderr, "prompt %d tokens, greedy: %s\n", n, out.c_str());
    return 0;
}
