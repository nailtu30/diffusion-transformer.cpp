#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include "dit.hpp"

struct DiffusionModel {
    virtual void compute(int n_threads,
                         struct ggml_tensor* x,
                         struct ggml_tensor* timesteps,
                         struct ggml_tensor* context,
                         struct ggml_tensor* c_concat,
                         struct ggml_tensor* y,
                         struct ggml_tensor* guidance,
                         int num_video_frames                      = -1,
                         std::vector<struct ggml_tensor*> controls = {},
                         float control_strength                    = 0.f,
                         struct ggml_tensor** output               = NULL,
                         struct ggml_context* output_ctx           = NULL,
                         std::vector<int> skip_layers              = std::vector<int>())             = 0;
    virtual void alloc_params_buffer()                                                  = 0;
    virtual void free_params_buffer()                                                   = 0;
    virtual void free_compute_buffer()                                                  = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                             = 0;
    virtual int64_t get_adm_in_channels()                                               = 0;
};

struct DiTModel : public DiffusionModel {
    MMDiTRunner mmdit;

    MMDiTModel(ggml_backend_t backend,
               std::map<std::string, enum ggml_type>& tensor_types)
        : mmdit(backend, tensor_types, "model.diffusion_model") {
    }

    void alloc_params_buffer() {
        mmdit.alloc_params_buffer();
    }

    void free_params_buffer() {
        mmdit.free_params_buffer();
    }

    void free_compute_buffer() {
        mmdit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return mmdit.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return 768 + 1280;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        return mmdit.compute(n_threads, x, timesteps, context, y, output, output_ctx, skip_layers);
    }
};

#endif
