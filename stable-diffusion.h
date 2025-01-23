#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef DIT_BUILD_SHARED_LIB
#define DIT_API
#else
#ifdef DIT_BUILD_DLL
#define DIT_API __declspec(dllexport)
#else
#define DIT_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define DIT_API __attribute__((visibility("default")))
#else
#define DIT_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG
};

enum sample_method_t {
    EULER_A,
    EULER,
    HEUN,
    DPM2,
    DPMPP2S_A,
    DPMPP2M,
    DPMPP2Mv2,
    IPNDM,
    IPNDM_V,
    LCM,
    N_SAMPLE_METHODS
};

enum schedule_t {
    DEFAULT,
    DISCRETE,
    KARRAS,
    EXPONENTIAL,
    AYS,
    GITS,
    N_SCHEDULES
};

// same as enum ggml_type
enum dit_type_t {
    DIT_TYPE_F32  = 0,
    DIT_TYPE_F16  = 1,
    DIT_TYPE_Q4_0 = 2,
    DIT_TYPE_Q4_1 = 3,
    // DIT_TYPE_Q4_2 = 4, support has been removed
    // DIT_TYPE_Q4_3 = 5, support has been removed
    DIT_TYPE_Q5_0     = 6,
    DIT_TYPE_Q5_1     = 7,
    DIT_TYPE_Q8_0     = 8,
    DIT_TYPE_Q8_1     = 9,
    DIT_TYPE_Q2_K     = 10,
    DIT_TYPE_Q3_K     = 11,
    DIT_TYPE_Q4_K     = 12,
    DIT_TYPE_Q5_K     = 13,
    DIT_TYPE_Q6_K     = 14,
    DIT_TYPE_Q8_K     = 15,
    DIT_TYPE_IQ2_XXS  = 16,
    DIT_TYPE_IQ2_XS   = 17,
    DIT_TYPE_IQ3_XXS  = 18,
    DIT_TYPE_IQ1_S    = 19,
    DIT_TYPE_IQ4_NL   = 20,
    DIT_TYPE_IQ3_S    = 21,
    DIT_TYPE_IQ2_S    = 22,
    DIT_TYPE_IQ4_XS   = 23,
    DIT_TYPE_I8       = 24,
    DIT_TYPE_I16      = 25,
    DIT_TYPE_I32      = 26,
    DIT_TYPE_I64      = 27,
    DIT_TYPE_F64      = 28,
    DIT_TYPE_IQ1_M    = 29,
    DIT_TYPE_BF16     = 30,
    DIT_TYPE_Q4_0_4_4 = 31,
    DIT_TYPE_Q4_0_4_8 = 32,
    DIT_TYPE_Q4_0_8_8 = 33,
    DIT_TYPE_TQ1_0    = 34,
    DIT_TYPE_TQ2_0    = 35,
    DIT_TYPE_COUNT,
};

DIT_API const char* dit_type_name(enum dit_type_t type);

enum dit_log_level_t {
    DIT_LOG_DEBUG,
    DIT_LOG_INFO,
    DIT_LOG_WARN,
    DIT_LOG_ERROR
};

typedef void (*dit_log_cb_t)(enum dit_log_level_t level, const char* text, void* data);
typedef void (*dit_progress_cb_t)(int step, int steps, float time, void* data);

DIT_API void dit_set_log_callback(dit_log_cb_t dit_log_cb, void* data);
DIT_API void dit_set_progress_callback(dit_progress_cb_t cb, void* data);
DIT_API int32_t get_num_physical_cores();
DIT_API const char* dit_get_system_info();

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} dit_image_t;

typedef struct sd_ctx_t sd_ctx_t;

DIT_API sd_ctx_t* new_sd_ctx(const char* model_path,
                            const char* vae_path,
                            bool free_params_immediately,
                            int n_threads,
                            enum dit_type_t wtype,
                            enum rng_type_t rng_type,
                            enum schedule_t s,
                            bool keep_vae_on_cpu);

DIT_API void free_sd_ctx(sd_ctx_t* sd_ctx);

DIT_API dit_image_t* class_label2img(sd_ctx_t* sd_ctx,
                           std::vector<int> class_label_prompt,
                           float cfg_scale,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           int sample_steps,
                           int64_t seed,
                           int batch_count);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__