#ifndef __DIT_HPP__
#define __DIT_HPP__

#include "ggml_extend.hpp"
#include "model.h"

#define DIT_GRAPH_SIZE 10240

struct Mlp : public GGMLBlock {
public:
    Mlp(int64_t in_features,
        int64_t hidden_features = -1,
        int64_t out_features    = -1,
        bool bias               = true) {
        // act_layer is always lambda: nn.GELU(approximate="tanh")
        // norm_layer is always None
        // use_conv is always False
        if (hidden_features == -1) {
            hidden_features = in_features;
        }
        if (out_features == -1) {
            out_features = in_features;
        }
        blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(in_features, hidden_features, bias));
        blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_features, out_features, bias));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, in_features]
        auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

        x = fc1->forward(ctx, x);
        x = ggml_gelu_inplace(ctx, x);
        x = fc2->forward(ctx, x);
        return x;
    }
};

struct PatchEmbed : public GGMLBlock {
    // 2D Image to Patch Embedding
protected:
    bool flatten;
    bool dynamic_img_pad;
    int patch_size;

public:
    PatchEmbed(int64_t img_size     = 224,
               int patch_size       = 16,
               int64_t in_chans     = 3,
               int64_t embed_dim    = 1536,
               bool bias            = true,
               bool flatten         = true,
               bool dynamic_img_pad = true)
        : patch_size(patch_size),
          flatten(flatten),
          dynamic_img_pad(dynamic_img_pad) {
        // img_size is always None
        // patch_size is always 2
        // in_chans is always 16
        // norm_layer is always False
        // strict_img_size is always true, but not used

        blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_chans,
                                                               embed_dim,
                                                               {patch_size, patch_size},
                                                               {patch_size, patch_size},
                                                               {0, 0},
                                                               {1, 1},
                                                               bias));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, C, H, W]
        // return: [N, H*W, embed_dim]
        auto proj = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);

        if (dynamic_img_pad) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int pad_h = (patch_size - H % patch_size) % patch_size;
            int pad_w = (patch_size - W % patch_size) % patch_size;
            x         = ggml_pad(ctx, x, pad_w, pad_h, 0, 0);  // TODO: reflect pad mode
        }
        x = proj->forward(ctx, x);

        if (flatten) {
            x = ggml_reshape_3d(ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        }
        return x;
    }
};

struct TimestepEmbedder : public GGMLBlock {
    // Embeds scalar timesteps into vector representations.
protected:
    int64_t frequency_embedding_size;

public:
    TimestepEmbedder(int64_t hidden_size,
                     int64_t frequency_embedding_size = 256)
        : frequency_embedding_size(frequency_embedding_size) {
        blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(frequency_embedding_size, hidden_size, true, true));
        blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, true));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* t) {
        // t: [N, ]
        // return: [N, hidden_size]
        auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
        auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

        auto t_freq = ggml_nn_timestep_embedding(ctx, t, frequency_embedding_size);  // [N, frequency_embedding_size]

        auto t_emb = mlp_0->forward(ctx, t_freq);
        t_emb      = ggml_silu_inplace(ctx, t_emb);
        t_emb      = mlp_2->forward(ctx, t_emb);
        return t_emb;
    }
};

struct VectorEmbedder : public GGMLBlock {
    // Embeds a flat vector of dimension input_dim
public:
    VectorEmbedder(int64_t input_dim,
                   int64_t hidden_size) {
        blocks["mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(input_dim, hidden_size, true, true));
        blocks["mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true, true));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, input_dim]
        // return: [N, hidden_size]
        auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
        auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

        x = mlp_0->forward(ctx, x);
        x = ggml_silu_inplace(ctx, x);
        x = mlp_2->forward(ctx, x);
        return x;
    }
};

class RMSNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "") {
        enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
        params["weight"]     = ggml_new_tensor_1d(ctx, wtype, hidden_size);
    }

public:
    RMSNorm(int64_t hidden_size,
            float eps = 1e-06f)
        : hidden_size(hidden_size),
          eps(eps) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        x                     = ggml_rms_norm(ctx, x, eps);
        x                     = ggml_mul(ctx, x, w);
        return x;
    }
};

class SelfAttention : public GGMLBlock {
public:
    int64_t num_heads;
    bool pre_only;
    std::string qk_norm;

public:
    SelfAttention(int64_t dim,
                  int64_t num_heads   = 8,
                  std::string qk_norm = "",
                  bool qkv_bias       = false,
                  bool pre_only       = false)
        : num_heads(num_heads), pre_only(pre_only), qk_norm(qk_norm) {
        int64_t d_head = dim / num_heads;
        blocks["qkv"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
        if (!pre_only) {
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
        }
        if (qk_norm == "rms") {
            blocks["ln_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(d_head, 1.0e-6));
            blocks["ln_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(d_head, 1.0e-6));
        } else if (qk_norm == "ln") {
            blocks["ln_q"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_head, 1.0e-6));
            blocks["ln_k"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_head, 1.0e-6));
        }
    }

    std::vector<struct ggml_tensor*> pre_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
        auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);

        auto qkv         = qkv_proj->forward(ctx, x);
        auto qkv_vec     = split_qkv(ctx, qkv);
        int64_t head_dim = qkv_vec[0]->ne[0] / num_heads;
        auto q           = ggml_reshape_4d(ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);  // [N, n_token, n_head, d_head]
        auto k           = ggml_reshape_4d(ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);  // [N, n_token, n_head, d_head]
        auto v           = qkv_vec[2];                                                                                   // [N, n_token, n_head*d_head]

        if (qk_norm == "rms" || qk_norm == "ln") {
            auto ln_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_q"]);
            auto ln_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["ln_k"]);
            q         = ln_q->forward(ctx, q);
            k         = ln_k->forward(ctx, k);
        }

        q = ggml_reshape_3d(ctx, q, q->ne[0] * q->ne[1], q->ne[2], q->ne[3]);  // [N, n_token, n_head*d_head]
        k = ggml_reshape_3d(ctx, k, k->ne[0] * k->ne[1], k->ne[2], k->ne[3]);  // [N, n_token, n_head*d_head]

        return {q, k, v};
    }

    struct ggml_tensor* post_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
        GGML_ASSERT(!pre_only);

        auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

        x = proj->forward(ctx, x);  // [N, n_token, dim]
        return x;
    }

    // x: [N, n_token, dim]
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        auto qkv = pre_attention(ctx, x);
        x        = ggml_nn_attention_ext(ctx, qkv[0], qkv[1], qkv[2], num_heads);  // [N, n_token, dim]
        x        = post_attention(ctx, x);                                         // [N, n_token, dim]
        return x;
    }
};

__STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                               struct ggml_tensor* x,
                                               struct ggml_tensor* shift,
                                               struct ggml_tensor* scale) {
    // x: [N, L, C]
    // scale: [N, C]
    // shift: [N, C]
    scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
    shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C]
    x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    x     = ggml_add(ctx, x, shift);
    return x;
}

struct DiTBlock: public GGMLBlock {
public:
    int64_t num_heads;

public:
    DiTBlock(
        int64_t hidden_size,
        int64_t num_heads,
        float mlp_ratio     = 4.0,
        std::string qk_norm = "",
        bool qkv_bias       = true
    ): num_heads(num_heads) {
        // rmsnorm is always Flase
        // scale_mod_only is always Flase
        // swiglu is always Flase
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
        blocks["attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qk_norm, qkv_bias, false));

        blocks["norm2"]        = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
        int64_t mlp_hidden_dim = (int64_t)(hidden_size * mlp_ratio);
        blocks["mlp"]          = std::shared_ptr<GGMLBlock>(new Mlp(hidden_size, mlp_hidden_dim));

        int64_t n_mods = 6;
        blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, n_mods * hidden_size));
    }

    std::pair<std::vector<struct ggml_tensor*>, std::vector<struct ggml_tensor*>> pre_attention(struct ggml_context* ctx,
                                                                                                struct ggml_tensor* x,
                                                                                                struct ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        auto norm1              = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
        auto attn               = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        int64_t n_mods = 6;
        auto m = adaLN_modulation_1->forward(ctx, ggml_silu(ctx, c));  // [N, n_mods * hidden_size]
        m      = ggml_reshape_3d(ctx, m, c->ne[0], n_mods, c->ne[1]);  // [N, n_mods, hidden_size]
        m      = ggml_cont(ctx, ggml_permute(ctx, m, 0, 2, 1, 3));     // [n_mods, N, hidden_size]

        int64_t offset = m->nb[1] * m->ne[1];
        auto shift_msa = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 0);  // [N, hidden_size]
        auto scale_msa = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 1);  // [N, hidden_size]
        auto gate_msa  = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 2);  // [N, hidden_size]
        auto shift_mlp = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 3);  // [N, hidden_size]
        auto scale_mlp = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 4);  // [N, hidden_size]
        auto gate_mlp  = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 5);  // [N, hidden_size]

        auto attn_in = modulate(ctx, norm1->forward(ctx, x), shift_msa, scale_msa);

        auto qkv = attn->pre_attention(ctx, attn_in);

        return {qkv, {x, gate_msa, shift_mlp, scale_mlp, gate_mlp}};
    }

    struct ggml_tensor* post_attention(struct ggml_context* ctx,
                                       struct ggml_tensor* attn_out,
                                       struct ggml_tensor* x,
                                       struct ggml_tensor* gate_msa,
                                       struct ggml_tensor* shift_mlp,
                                       struct ggml_tensor* scale_mlp,
                                       struct ggml_tensor* gate_mlp) {
        // attn_out: [N, n_token, hidden_size]
        // x: [N, n_token, hidden_size]
        // gate_msa: [N, hidden_size]
        // shift_mlp: [N, hidden_size]
        // scale_mlp: [N, hidden_size]
        // gate_mlp: [N, hidden_size]
        // return: [N, n_token, hidden_size]
        auto attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);
        auto norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
        auto mlp   = std::dynamic_pointer_cast<Mlp>(blocks["mlp"]);

        gate_msa = ggml_reshape_3d(ctx, gate_msa, gate_msa->ne[0], 1, gate_msa->ne[1]);  // [N, 1, hidden_size]
        gate_mlp = ggml_reshape_3d(ctx, gate_mlp, gate_mlp->ne[0], 1, gate_mlp->ne[1]);  // [N, 1, hidden_size]

        attn_out = attn->post_attention(ctx, attn_out);

        x            = ggml_add(ctx, x, ggml_mul(ctx, attn_out, gate_msa));
        auto mlp_out = mlp->forward(ctx, modulate(ctx, norm2->forward(ctx, x), shift_mlp, scale_mlp));
        x            = ggml_add(ctx, x, ggml_mul(ctx, mlp_out, gate_mlp));

        return x;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        // return: [N, n_token, hidden_size]

        auto attn = std::dynamic_pointer_cast<SelfAttention>(blocks["attn"]);

        auto qkv_intermediates = pre_attention(ctx, x, c);
        auto qkv               = qkv_intermediates.first;
        auto intermediates     = qkv_intermediates.second;

        auto attn_out = ggml_nn_attention_ext(ctx, qkv[0], qkv[1], qkv[2], num_heads);  // [N, n_token, dim]
        x             = post_attention(ctx,
                                        attn_out,
                                        intermediates[0],
                                        intermediates[1],
                                        intermediates[2],
                                        intermediates[3],
                                        intermediates[4]);
        return x;  // [N, n_token, dim]
    }
};

struct FinalLayer : public GGMLBlock {
    // The final layer of DiT.
public:
    FinalLayer(int64_t hidden_size,
               int64_t patch_size,
               int64_t out_channels) {
        // total_out_channels is always None
        blocks["norm_final"]         = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
        blocks["linear"]             = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, patch_size * patch_size * out_channels, true, true));
        blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 2 * hidden_size));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* c) {
        // x: [N, n_token, hidden_size]
        // c: [N, hidden_size]
        // return: [N, n_token, patch_size * patch_size * out_channels]
        auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
        auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
        auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

        auto m = adaLN_modulation_1->forward(ctx, ggml_silu(ctx, c));  // [N, 2 * hidden_size]
        m      = ggml_reshape_3d(ctx, m, c->ne[0], 2, c->ne[1]);       // [N, 2, hidden_size]
        m      = ggml_cont(ctx, ggml_permute(ctx, m, 0, 2, 1, 3));     // [2, N, hidden_size]

        int64_t offset = m->nb[1] * m->ne[1];
        auto shift     = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 0);  // [N, hidden_size]
        auto scale     = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 1);  // [N, hidden_size]

        x = modulate(ctx, norm_final->forward(ctx, x), shift, scale);
        x = linear->forward(ctx, x);

        return x;
    }
};

struct DiT: public GGMLBlock {
    // Diffusion model with a Transformer backbone.
protected:
    int64_t input_size              = 32;
    int64_t patch_size              = 2;
    int64_t in_channels             = 4;
    int64_t hidden_size             = 1152;
    int64_t depth                   = 28;
    int64_t num_heads               = 16;
    float mlp_ratio                 = 4.0f;
    float class_dropout_prob        = 0.1f;
    int64_t num_classes             = 1000;
    bool learn_sigma                = true;
    bool record_inputs              = false;
    int64_t num_patches             = -1;
    int64_t out_channels            = -1;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        enum ggml_type wtype = GGML_TYPE_F32;
        num_patches = int(input_size / patch_size) * int(input_size / patch_size);
        params["pos_embed"]  = ggml_new_tensor_3d(ctx, wtype, hidden_size, num_patches, 1);
    }

public:
    DiT() {
        // read tensors from tensor_types
        LOG_INFO("DiT layers: %d", depth);
        if (learn_sigma)
        {
            out_channels = in_channels * 2;
        } else {
            out_channels = in_channels;
        }
        blocks["x_embedder"] = std::shared_ptr<GGMLBlock>(new PatchEmbed(input_size, patch_size, in_channels, hidden_size, true));
        blocks["t_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedder(hidden_size));
        // blocks["y_embedder"] = std::shared_ptr<GGMLBlock>(new VectorEmbedder(num_classes, hidden_size));
        int use_cfg_embedding = class_dropout_prob > 0;
        blocks["y_embedder"] = std::shared_ptr<GGMLBlock>(new Embedding(num_classes + use_cfg_embedding, hidden_size));

        for (int i = 0; i < depth; i++) {
            blocks["blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new DiTBlock(hidden_size,
                                                                                                    num_heads,
                                                                                                    mlp_ratio,
                                                                                                    "",
                                                                                                    true));
        }
        blocks["final_layer"] = std::shared_ptr<GGMLBlock>(new FinalLayer(hidden_size, patch_size, out_channels));
    }

    struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                                   struct ggml_tensor* x,
                                   int64_t h,
                                   int64_t w) {
        // x: [N, H*W, patch_size * patch_size * C]
        // return: [N, C, H, W]
        int64_t n = x->ne[2];
        int64_t c = out_channels;
        int64_t p = patch_size;
        h         = (h + 1) / p;
        w         = (w + 1) / p;

        GGML_ASSERT(h * w == x->ne[1]);

        x = ggml_reshape_4d(ctx, x, c, p * p, w * h, n);       // [N, H*W, P*P, C]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));  // [N, C, H*W, P*P]
        x = ggml_reshape_4d(ctx, x, p, p, w, h * c * n);       // [N*C*H, W, P, P]
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*H, P, W, P]
        x = ggml_reshape_4d(ctx, x, p * w, p * h, c, n);       // [N, C, H*P, W*P]
        return x;
    }

    struct ggml_tensor* forward_core_with_concat(struct ggml_context* ctx,
                                                 struct ggml_tensor* x,
                                                 struct ggml_tensor* c_mod) {
        // x: [N, H*W, hidden_size]
        // context: [N, n_context, d_context]
        // c: [N, hidden_size]
        // return: [N, N*W, patch_size * patch_size * out_channels]
        auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

        for (int i = 0; i < depth; i++) {
            auto block = std::dynamic_pointer_cast<DiTBlock>(blocks["joint_blocks." + std::to_string(i)]);
            x = block->forward(ctx, x, c_mod);
        }

        x = final_layer->forward(ctx, x, c_mod);  // (N, T, patch_size ** 2 * out_channels)

        return x;
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* t,
                                struct ggml_tensor* y) {
        // Forward pass of DiT.
        // x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        // t: (N,) tensor of diffusion timesteps
        // y: (N, adm_in_channels) tensor of class labels
        // context: (N, L, D)
        // return: (N, C, H, W)
        auto x_embedder = std::dynamic_pointer_cast<PatchEmbed>(blocks["x_embedder"]);
        auto t_embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);

        int64_t w = x->ne[0];
        int64_t h = x->ne[1];

        auto patch_embed = x_embedder->forward(ctx, x);            // [N, H*W, hidden_size]
        auto pos_embed   = params["pos_embed"];           // [1, H*W, hidden_size]
        x                = ggml_add(ctx, patch_embed, pos_embed);  // [N, H*W, hidden_size]

        auto c = t_embedder->forward(ctx, t);  // [N, hidden_size]
        auto y_embedder = std::dynamic_pointer_cast<VectorEmbedder>(blocks["y_embedder"]);

        y = y_embedder->forward(ctx, y);  // [N, hidden_size]
        c = ggml_add(ctx, c, y);

        x = forward_core_with_concat(ctx, x, c);  // (N, H*W, patch_size ** 2 * out_channels)

        x = unpatchify(ctx, x, h, w);  // [N, C, H, W]

        return x;
    }
};

struct DiTRunner: public GGMLRunner {
    DiT dit;
    static std::map<std::string, enum ggml_type> empty_tensor_types;
    DiTRunner(ggml_backend_t backend,
                std::map<std::string, enum ggml_type>& tensor_types = empty_tensor_types,
                const std::string prefix                            = "")
        : GGMLRunner(backend), dit() {
        dit.init(params_ctx, tensor_types, prefix);
    }

    std::string get_desc() {
        return "dit";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        dit.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* y) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, DIT_GRAPH_SIZE, false);

        x         = to_backend(x);
        y         = to_backend(y);
        timesteps = to_backend(timesteps);

        struct ggml_tensor* out = dit.forward(compute_ctx,
                                                x,
                                                timesteps,
                                                y);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* y,
                 struct ggml_tensor** output     = NULL,
                 struct ggml_context* output_ctx = NULL) {
        // x: [N, in_channels, h, w]
        // timesteps: [N, ]
        // context: [N, max_position, hidden_size]([N, 154, 4096]) or [1, max_position, hidden_size]
        // y: [N, adm_in_channels] or [1, adm_in_channels]
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, y);
        };

        GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }

    void test() {
        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        struct ggml_context* work_ctx = ggml_init(params);
        GGML_ASSERT(work_ctx != NULL);

        {
            // cpu f16: pass
            // cpu f32: pass
            // cuda f16: pass
            // cuda f32: pass
            auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 128, 128, 16, 1);
            std::vector<float> timesteps_vec(1, 999.f);
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            ggml_set_f32(x, 0.01f);
            // print_ggml_tensor(x);

            auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 4096, 154, 1);
            ggml_set_f32(context, 0.01f);
            // print_ggml_tensor(context);

            auto y = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, 2048, 1);
            ggml_set_f32(y, 0.01f);
            // print_ggml_tensor(y);

            struct ggml_tensor* out = NULL;

            int t0 = ggml_time_ms();
            compute(8, x, timesteps, y, &out, work_ctx);
            int t1 = ggml_time_ms();

            print_ggml_tensor(out);
            LOG_DEBUG("dit test done in %dms", t1 - t0);
        }
    }

    static void load_from_file_and_test(const std::string& file_path) {
        // ggml_backend_t backend    = ggml_backend_cuda_init(0);
        ggml_backend_t backend             = ggml_backend_cpu_init();
        ggml_type model_data_type          = GGML_TYPE_F16;
        std::shared_ptr<DiTRunner> dit = std::shared_ptr<DiTRunner>(new DiTRunner(backend));
        {
            LOG_INFO("loading from '%s'", file_path.c_str());

            dit->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            dit->get_param_tensors(tensors, "blocks.");

            ModelLoader model_loader;
            if (!model_loader.init_from_file(file_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            bool success = model_loader.load_tensors(tensors, backend);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("mmdit model loaded");
        }
        dit->test();
    }
};

#endif