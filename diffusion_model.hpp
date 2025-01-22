#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include "dit.hpp"

struct DiffusionModel {
    virtual void compute(int n_threads,
                         struct ggml_tensor* x,
                         struct ggml_tensor* timesteps,
                         struct ggml_tensor* y,
                         struct ggml_tensor** output               = NULL,
                         struct ggml_context* output_ctx           = NULL);
    virtual void alloc_params_buffer()                                                  = 0;
    virtual void free_params_buffer()                                                   = 0;
    virtual void free_compute_buffer()                                                  = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                             = 0;
    virtual int64_t get_adm_in_channels()                                               = 0;
};

struct DiTModel : public DiffusionModel {
    DiTRunner dit;

    DiTModel(ggml_backend_t backend,
               std::map<std::string, enum ggml_type>& tensor_types)
        : dit(backend, tensor_types, "") {
    }

    void alloc_params_buffer() {
        dit.alloc_params_buffer();
    }

    void free_params_buffer() {
        dit.free_params_buffer();
    }

    void free_compute_buffer() {
        dit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        dit.get_param_tensors(tensors, "");
    }

    size_t get_params_buffer_size() {
        return dit.get_params_buffer_size();
    }

    // int64_t get_adm_in_channels() {
    //     return 768 + 1280;
    // }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* y,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL) {
        return dit.compute(n_threads, x, timesteps, y, output, output_ctx);
    }
};

#endif
