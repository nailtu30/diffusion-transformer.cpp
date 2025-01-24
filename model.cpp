#include <stdarg.h>
#include <fstream>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "stable-diffusion.h"
#include "util.h"
#include "vocab.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "stable-diffusion.h"

#ifdef DIT_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef DIT_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#define ST_HEADER_SIZE_LEN 8

uint64_t read_u64(uint8_t* buffer) {
    // little endian
    uint64_t value = 0;
    value |= static_cast<int64_t>(buffer[7]) << 56;
    value |= static_cast<int64_t>(buffer[6]) << 48;
    value |= static_cast<int64_t>(buffer[5]) << 40;
    value |= static_cast<int64_t>(buffer[4]) << 32;
    value |= static_cast<int64_t>(buffer[3]) << 24;
    value |= static_cast<int64_t>(buffer[2]) << 16;
    value |= static_cast<int64_t>(buffer[1]) << 8;
    value |= static_cast<int64_t>(buffer[0]);
    return value;
}

int32_t read_int(uint8_t* buffer) {
    // little endian
    int value = 0;
    value |= buffer[3] << 24;
    value |= buffer[2] << 16;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

uint16_t read_short(uint8_t* buffer) {
    // little endian
    uint16_t value = 0;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

/*================================================= Preprocess ==================================================*/

bool is_unused_tensor(std::string name) {
    // if (contains("blocks", name) & 
    // (ends_with(name, "attn.qkv.weight") || 
    // ends_with(name, "attn.proj.weight") || 
    // ends_with(name, "mlp.fc1.weight") || 
    // ends_with(name, "mlp.fc2.weight")))
    // {
    //     return false;
    // }
    // return true;
    return false;
}

void preprocess_tensor(TensorStorage tensor_storage,
                       std::vector<TensorStorage>& processed_tensor_storages) {
    processed_tensor_storages.push_back(tensor_storage);
}

float bf16_to_f32(uint16_t bfloat16) {
    uint32_t val_bits = (static_cast<uint32_t>(bfloat16) << 16);
    return *reinterpret_cast<float*>(&val_bits);
}

uint16_t f8_e4m3_to_f16(uint8_t f8) {
    // do we need to support uz?

    const uint32_t exponent_bias = 7;
    if (f8 == 0xff) {
        return ggml_fp32_to_fp16(-NAN);
    } else if (f8 == 0x7f) {
        return ggml_fp32_to_fp16(NAN);
    }

    uint32_t sign     = f8 & 0x80;
    uint32_t exponent = (f8 & 0x78) >> 3;
    uint32_t mantissa = f8 & 0x07;
    uint32_t result   = sign << 24;
    if (exponent == 0) {
        if (mantissa > 0) {
            exponent = 0x7f - exponent_bias;

            // yes, 2 times
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }

            result |= (mantissa & 0x03) << 21;
            result |= exponent << 23;
        }
    } else {
        result |= mantissa << 20;
        exponent += 0x7f - exponent_bias;
        result |= exponent << 23;
    }

    return ggml_fp32_to_fp16(*reinterpret_cast<const float*>(&result));
}

uint16_t f8_e5m2_to_f16(uint8_t fp8) {
    uint8_t sign     = (fp8 >> 7) & 0x1;
    uint8_t exponent = (fp8 >> 2) & 0x1F;
    uint8_t mantissa = fp8 & 0x3;

    uint16_t fp16_sign = sign << 15;
    uint16_t fp16_exponent;
    uint16_t fp16_mantissa;

    if (exponent == 0 && mantissa == 0) {  // zero
        return fp16_sign;
    }

    if (exponent == 0x1F) {  // NAN and INF
        fp16_exponent = 0x1F;
        fp16_mantissa = mantissa ? (mantissa << 8) : 0;
        return fp16_sign | (fp16_exponent << 10) | fp16_mantissa;
    }

    if (exponent == 0) {  // subnormal numbers
        fp16_exponent = 0;
        fp16_mantissa = (mantissa << 8);
        return fp16_sign | fp16_mantissa;
    }

    // normal numbers
    int16_t true_exponent = (int16_t)exponent - 15 + 15;
    if (true_exponent <= 0) {
        fp16_exponent = 0;
        fp16_mantissa = (mantissa << 8);
    } else if (true_exponent >= 0x1F) {
        fp16_exponent = 0x1F;
        fp16_mantissa = 0;
    } else {
        fp16_exponent = (uint16_t)true_exponent;
        fp16_mantissa = mantissa << 8;
    }

    return fp16_sign | (fp16_exponent << 10) | fp16_mantissa;
}

void bf16_to_f32_vec(uint16_t* src, float* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = bf16_to_f32(src[i]);
    }
}

void f8_e4m3_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e4m3_to_f16(src[i]);
    }
}
void f8_e5m2_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e5m2_to_f16(src[i]);
    }
}

void convert_tensor(void* src,
                    ggml_type src_type,
                    void* dst,
                    ggml_type dst_type,
                    int nrows,
                    int n_per_row) {
    int n = nrows * n_per_row;
    if (src_type == dst_type) {
        size_t nbytes = n * ggml_type_size(src_type) / ggml_blck_size(src_type);
        memcpy(((char*)dst), ((char*)src), nbytes);
    } else if (src_type == GGML_TYPE_F32) {
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src, dst, 0, nrows, n_per_row, im);
        }
    } else if (dst_type == GGML_TYPE_F32) {
        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t*)src, (float*)dst, n);
        } else {
            auto qtype = ggml_get_type_traits(src_type);
            if (qtype->to_float == NULL) {
                throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available",
                                                ggml_type_name(src_type)));
            }
            qtype->to_float(src, (float*)dst, n);
        }
    } else {
        // src_type == GGML_TYPE_F16 => dst_type is quantized
        // src_type is quantized => dst_type == GGML_TYPE_F16 or dst_type is quantized
        auto qtype = ggml_get_type_traits(src_type);
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available",
                                            ggml_type_name(src_type)));
        }
        std::vector<char> buf;
        buf.resize(sizeof(float) * n);
        char* src_data_f32 = buf.data();
        qtype->to_float(src, (float*)src_data_f32, n);
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src_data_f32, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src_data_f32, dst, 0, nrows, n_per_row, im);
        }
    }
}

/*================================================= ModelLoader ==================================================*/

// ported from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L16
std::map<char, int> unicode_to_byte() {
    std::map<int, char> byte_to_unicode;

    // List of utf-8 byte ranges
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_to_unicode[b] = static_cast<char>(b);
    }

    for (int b = 49825; b <= 49836; ++b) {
        byte_to_unicode[b] = static_cast<char>(b);
    }

    for (int b = 49838; b <= 50111; ++b) {
        byte_to_unicode[b] = static_cast<char>(b);
    }
    // printf("%d %d %d %d\n", static_cast<int>('¡'), static_cast<int>('¬'), static_cast<int>('®'), static_cast<int>('ÿ'));
    // exit(1);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_to_unicode.find(b) == byte_to_unicode.end()) {
            byte_to_unicode[b] = static_cast<char>(256 + n);
            n++;
        }
    }

    // byte_encoder = bytes_to_unicode()
    // byte_decoder = {v: k for k, v in byte_encoder.items()}
    std::map<char, int> byte_decoder;

    for (const auto& entry : byte_to_unicode) {
        byte_decoder[entry.second] = entry.first;
    }

    byte_to_unicode.clear();

    return byte_decoder;
}

bool is_zip_file(const std::string& file_path) {
    struct zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == NULL) {
        return false;
    }
    zip_close(zip);
    return true;
}

bool is_gguf_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4];

    file.read(magic, sizeof(magic));
    if (!file) {
        return false;
    }
    for (uint32_t i = 0; i < sizeof(magic); i++) {
        if (magic[i] != GGUF_MAGIC[i]) {
            return false;
        }
    }

    return true;
}

bool is_safetensors_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        return false;
    }

    size_t header_size_ = read_u64(header_size_buf);
    if (header_size_ >= file_size_ || header_size_ <= 2) {
        return false;
    }

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        return false;
    }
    nlohmann::json header_ = nlohmann::json::parse(header_buf.data());
    if (header_.is_discarded()) {
        return false;
    }
    return true;
}

bool ModelLoader::init_from_file(const std::string& file_path, const std::string& prefix) {
    if (is_directory(file_path)) {
        LOG_INFO("load %s using diffusers format", file_path.c_str());
        return init_from_diffusers_file(file_path, prefix);
    } else if (is_gguf_file(file_path)) {
        LOG_INFO("load %s using gguf format", file_path.c_str());
        return init_from_gguf_file(file_path, prefix);
    } else if (is_safetensors_file(file_path)) {
        LOG_INFO("load %s using safetensors format", file_path.c_str());
        return init_from_safetensors_file(file_path, prefix);
    } else if (is_zip_file(file_path)) {
        LOG_INFO("load %s using checkpoint format", file_path.c_str());
        return init_from_ckpt_file(file_path, prefix);
    } else {
        LOG_WARN("unknown format %s", file_path.c_str());
        return false;
    }
}

/*================================================= GGUFModelLoader ==================================================*/

bool ModelLoader::init_from_gguf_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    gguf_context* ctx_gguf_ = NULL;
    ggml_context* ctx_meta_ = NULL;
    ctx_gguf_               = gguf_init_from_file(file_path.c_str(), {true, &ctx_meta_});
    if (!ctx_gguf_) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return false;
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf_);

    size_t total_size  = 0;
    size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name          = gguf_get_tensor_name(ctx_gguf_, i);
        struct ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset             = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        // LOG_DEBUG("%s", name.c_str());

        TensorStorage tensor_storage(prefix + name, dummy->type, dummy->ne, ggml_n_dims(dummy), file_index, offset);

        GGML_ASSERT(ggml_nbytes(dummy) == tensor_storage.nbytes());

        tensor_storages.push_back(tensor_storage);
        tensor_storages_types[tensor_storage.name] = tensor_storage.type;
    }

    gguf_free(ctx_gguf_);
    ggml_free(ctx_meta_);

    return true;
}

/*================================================= SafeTensorsModelLoader ==================================================*/

ggml_type str_to_ggml_type(const std::string& dtype) {
    ggml_type ttype = GGML_TYPE_COUNT;
    if (dtype == "F16") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "BF16") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F32") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F8_E4M3") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "F8_E5M2") {
        ttype = GGML_TYPE_F16;
    }
    return ttype;
}

// https://huggingface.co/docs/safetensors/index
bool ModelLoader::init_from_safetensors_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        LOG_ERROR("invalid safetensor file '%s'", file_path.c_str());
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        LOG_ERROR("read safetensors header size failed: '%s'", file_path.c_str());
        return false;
    }

    size_t header_size_ = read_u64(header_size_buf);
    if (header_size_ >= file_size_) {
        LOG_ERROR("invalid safetensor file '%s'", file_path.c_str());
        return false;
    }

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        LOG_ERROR("read safetensors header failed: '%s'", file_path.c_str());
        return false;
    }

    nlohmann::json header_ = nlohmann::json::parse(header_buf.data());

    for (auto& item : header_.items()) {
        std::string name           = item.key();
        nlohmann::json tensor_info = item.value();
        // LOG_DEBUG("%s %s\n", name.c_str(), tensor_info.dump().c_str());

        if (name == "__metadata__") {
            continue;
        }

        if (is_unused_tensor(name)) {
            continue;
        }

        std::string dtype    = tensor_info["dtype"];
        nlohmann::json shape = tensor_info["shape"];

        size_t begin = tensor_info["data_offsets"][0].get<size_t>();
        size_t end   = tensor_info["data_offsets"][1].get<size_t>();

        ggml_type type = str_to_ggml_type(dtype);
        if (type == GGML_TYPE_COUNT) {
            LOG_ERROR("unsupported dtype '%s' (tensor '%s')", dtype.c_str(), name.c_str());
            return false;
        }

        if (shape.size() > DIT_MAX_DIMS) {
            LOG_ERROR("invalid tensor '%s'", name.c_str());
            return false;
        }

        int n_dims              = (int)shape.size();
        int64_t ne[DIT_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            ne[i] = shape[i].get<int64_t>();
        }

        if (n_dims == 5) {
            if (ne[3] == 1 && ne[4] == 1) {
                n_dims = 4;
            } else {
                LOG_ERROR("invalid tensor '%s'", name.c_str());
                return false;
            }
        }

        // ggml_n_dims returns 1 for scalars
        if (n_dims == 0) {
            n_dims = 1;
        }

        TensorStorage tensor_storage(prefix + name, type, ne, n_dims, file_index, ST_HEADER_SIZE_LEN + header_size_ + begin);
        tensor_storage.reverse_ne();

        size_t tensor_data_size = end - begin;

        if (dtype == "BF16") {
            tensor_storage.is_bf16 = true;
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F8_E4M3") {
            tensor_storage.is_f8_e4m3 = true;
            // f8 -> f16
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F8_E5M2") {
            tensor_storage.is_f8_e5m2 = true;
            // f8 -> f16
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size * 2);
        } else {
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size);
        }

        tensor_storages.push_back(tensor_storage);
        tensor_storages_types[tensor_storage.name] = tensor_storage.type;

        // LOG_DEBUG("%s %s", tensor_storage.to_string().c_str(), dtype.c_str());
    }

    return true;
}

/*================================================= DiffusersModelLoader ==================================================*/

bool ModelLoader::init_from_diffusers_file(const std::string& file_path, const std::string& prefix) {
    std::string unet_path = path_join(file_path, "unet/diffusion_pytorch_model.safetensors");
    std::string vae_path  = path_join(file_path, "vae/diffusion_pytorch_model.safetensors");
    std::string clip_path = path_join(file_path, "text_encoder/model.safetensors");

    if (!init_from_safetensors_file(unet_path, "unet.")) {
        return false;
    }
    if (!init_from_safetensors_file(vae_path, "vae.")) {
        return false;
    }
    if (!init_from_safetensors_file(clip_path, "te.")) {
        return false;
    }
    return true;
}

/*================================================= CkptModelLoader ==================================================*/
struct PickleTensorReader {
    enum ReadPhase {
        READ_NAME,
        READ_DATA,
        CHECK_SIZE,
        READ_DIMENS
    };
    ReadPhase phase   = READ_NAME;
    size_t entry_size = 0;
    int32_t nelements = 0;

    TensorStorage tensor_storage;

    static ggml_type global_type;  // all pickle_tensors data type
    static bool read_global_type;

    bool read_int_value(uint32_t value) {
        if (phase == CHECK_SIZE) {
            if (entry_size == value * ggml_type_size(tensor_storage.type)) {
                nelements = value;
                phase     = READ_DIMENS;
                return true;
            } else {
                phase = READ_NAME;
            }
        } else if (phase == READ_DIMENS) {
            if (tensor_storage.n_dims + 1 > DIT_MAX_DIMS) {  // too many dimens
                phase                 = READ_NAME;
                tensor_storage.n_dims = 0;
            }
            if (nelements % value == 0) {
                tensor_storage.ne[tensor_storage.n_dims] = value;
                tensor_storage.n_dims++;
            }
        }
        return false;
    }

    void read_global(const std::string& str) {
        if (str == "FloatStorage") {
            if (read_global_type) {
                global_type      = GGML_TYPE_F32;
                read_global_type = false;
            }
            tensor_storage.type = GGML_TYPE_F32;
        } else if (str == "HalfStorage") {
            if (read_global_type) {
                global_type      = GGML_TYPE_F16;
                read_global_type = false;
            }
            tensor_storage.type = GGML_TYPE_F16;
        }
    }

    void read_string(const std::string& str, struct zip_t* zip, std::string dir) {
        if (str == "storage") {
            read_global_type = true;
        } else if (str != "state_dict") {
            if (phase == READ_DATA) {
                std::string entry_name = dir + "data/" + std::string(str);

                size_t i, n = zip_entries_total(zip);
                for (i = 0; i < n; ++i) {
                    zip_entry_openbyindex(zip, i);
                    {
                        std::string name = zip_entry_name(zip);
                        if (name == entry_name) {
                            tensor_storage.index_in_zip = (int)i;
                            entry_size                  = zip_entry_size(zip);
                            zip_entry_close(zip);
                            break;
                        }
                    }
                    zip_entry_close(zip);
                }

                phase = entry_size > 0 ? CHECK_SIZE : READ_NAME;
            }
            if (!read_global_type && phase == READ_NAME) {
                tensor_storage.name = str;
                phase               = READ_DATA;
                tensor_storage.type = global_type;
            }
        }
    }
};

ggml_type PickleTensorReader::global_type = GGML_TYPE_F32;  // all pickle_tensors data type
bool PickleTensorReader::read_global_type = false;

int find_char(uint8_t* buffer, int len, char c) {
    for (int pos = 0; pos < len; pos++) {
        if (buffer[pos] == c) {
            return pos;
        }
    }
    return -1;
}

#define MAX_STRING_BUFFER 512

bool ModelLoader::parse_data_pkl(uint8_t* buffer,
                                 size_t buffer_size,
                                 zip_t* zip,
                                 std::string dir,
                                 size_t file_index,
                                 const std::string prefix) {
    uint8_t* buffer_end = buffer + buffer_size;
    if (buffer[0] == 0x80) {  // proto
        if (buffer[1] != 2) {
            LOG_ERROR("Unsupported protocol\n");
            return false;
        }
        buffer += 2;  // 0x80 and version
        char string_buffer[MAX_STRING_BUFFER];
        bool finish = false;
        PickleTensorReader reader;
        // read pickle binary file
        while (!finish && buffer < buffer_end) {
            uint8_t opcode = *buffer;
            buffer++;
            // https://github.com/python/cpython/blob/3.7/Lib/pickletools.py#L1048
            // https://github.com/python/cpython/blob/main/Lib/pickle.py#L105
            switch (opcode) {
                case '}':  // EMPTY_DICT     = b'}'   # push empty dict
                    break;
                case ']':  // EMPTY_LIST     = b']'   # push empty list
                    break;
                // skip unused sections
                case 'h':  // BINGET         = b'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
                case 'q':  // BINPUT         = b'q'   #   "     "    "   "   " ;   "    " 1-byte arg
                case 'Q':  // BINPERSID      = b'Q'   #  "       "         "  ;  "  "   "     "  stack
                    buffer++;
                    break;
                case 'r':  // LONG_BINPUT    = b'r'   #   "     "    "   "   " ;   "    " 4-byte arg
                    buffer += 4;
                    break;
                case 0x95:  // FRAME            = b'\x95'  # indicate the beginning of a new frame
                    buffer += 8;
                    break;
                case 0x94:  // MEMOIZE          = b'\x94'  # store top of the stack in memo
                    break;
                case '(':  // MARK           = b'('   # push special markobject on stack
                    break;
                case 'K':  // BININT1        = b'K'   # push 1-byte unsigned int
                {
                    uint8_t value = *buffer;
                    if (reader.read_int_value(value)) {
                        buffer++;
                    }
                    buffer++;
                } break;
                case 'M':  // BININT2        = b'M'   # push 2-byte unsigned int
                {
                    uint16_t value = read_short(buffer);
                    if (reader.read_int_value(value)) {
                        buffer++;
                    }
                    buffer += 2;
                } break;
                case 'J':  // BININT         = b'J'   # push four-byte signed int
                {
                    const int32_t value = read_int(buffer);
                    if (reader.read_int_value(value)) {
                        buffer++;  // skip tuple after read num_elements
                    }
                    buffer += 4;
                } break;
                case 'X':  // BINUNICODE     = b'X'   #   "     "       "  ; counted UTF-8 string argument
                {
                    const int32_t len = read_int(buffer);
                    buffer += 4;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    if (len > MAX_STRING_BUFFER) {
                        LOG_WARN("tensor name very large");
                    }
                    memcpy(string_buffer, buffer, len < MAX_STRING_BUFFER ? len : (MAX_STRING_BUFFER - 1));
                    buffer += len;
                    reader.read_string(string_buffer, zip, dir);
                } break;
                case 0x8C:  // SHORT_BINUNICODE = b'\x8c'  # push short string; UTF-8 length < 256 bytes
                {
                    const int8_t len = *buffer;
                    buffer++;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len;
                    // printf("String: '%s'\n", string_buffer);
                } break;
                case 'c':  // GLOBAL         = b'c'   # push self.find_class(modname, name); 2 string args
                {
                    int len = find_char(buffer, MAX_STRING_BUFFER, '\n');

                    buffer += len + 1;
                    len = find_char(buffer, MAX_STRING_BUFFER, '\n');

                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len + 1;
                    reader.read_global(string_buffer);
                } break;
                case 0x86:  // TUPLE2         = b'\x86'  # build 2-tuple from two topmost stack items
                case 0x85:  // TUPLE1         = b'\x85'  # build 1-tuple from stack top
                case 't':   // TUPLE          = b't'   # build tuple from topmost stack items
                    if (reader.phase == PickleTensorReader::READ_DIMENS) {
                        reader.tensor_storage.reverse_ne();
                        reader.tensor_storage.file_index = file_index;
                        // if(strcmp(prefix.c_str(), "scarlett") == 0)
                        // printf(" ZIP got tensor %s \n ", reader.tensor_storage.name.c_str());
                        reader.tensor_storage.name = prefix + reader.tensor_storage.name;
                        tensor_storages.push_back(reader.tensor_storage);
                        tensor_storages_types[reader.tensor_storage.name] = reader.tensor_storage.type;

                        // LOG_DEBUG("%s", reader.tensor_storage.name.c_str());
                        // reset
                        reader = PickleTensorReader();
                    }
                    break;
                case '.':  // STOP           = b'.'   # every pickle ends with STOP
                    finish = true;
                    break;
                default:
                    break;
            }
        }
    }
    return true;
}

bool ModelLoader::init_from_ckpt_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    struct zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == NULL) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return false;
    }
    LOG_DEBUG("before zip_entries_total %p", zip);
    int n = (int)zip_entries_total(zip);
    LOG_DEBUG("zip_entries_total %n", n);
    for (int i = 0; i < n; ++i) {
        zip_entry_openbyindex(zip, i);
        {
            std::string name = zip_entry_name(zip);
            size_t pos       = name.find("data.pkl");
            if (pos != std::string::npos) {
                std::string dir = name.substr(0, pos);
                printf("ZIP %d, name = %s, dir = %s \n", i, name.c_str(), dir.c_str());
                void* pkl_data = NULL;
                size_t pkl_size;
                zip_entry_read(zip, &pkl_data, &pkl_size);

                // LOG_DEBUG("%lld", pkl_size);

                parse_data_pkl((uint8_t*)pkl_data, pkl_size, zip, dir, file_index, prefix);

                free(pkl_data);
            }
        }
        zip_entry_close(zip);
    }
    zip_close(zip);
    return true;
}

DITVersion ModelLoader::get_dit_version() {
    return VERSION_DIT1;
}

ggml_type ModelLoader::get_dit_wtype() {
    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (ggml_is_quantized(tensor_storage.type)) {
            return tensor_storage.type;
        }

        if (tensor_should_be_converted(tensor_storage, GGML_TYPE_Q4_K)) {
            return tensor_storage.type;
        }
    }
    return GGML_TYPE_COUNT;
}

void ModelLoader::set_wtype_override(ggml_type wtype, std::string prefix) {
    for (auto& pair : tensor_storages_types) {
        if (prefix.size() < 1 || pair.first.substr(0, prefix.size()) == prefix) {
            for (auto& tensor_storage : tensor_storages) {
                if (tensor_storage.name == pair.first) {
                    if (tensor_should_be_converted(tensor_storage, wtype)) {
                        pair.second = wtype;
                    }
                    break;
                }
            }
        }
    }
}

std::vector<TensorStorage> remove_duplicates(const std::vector<TensorStorage>& vec) {
    std::vector<TensorStorage> res;
    std::unordered_map<std::string, size_t> name_to_index_map;

    for (size_t i = 0; i < vec.size(); ++i) {
        const std::string& current_name = vec[i].name;
        auto it                         = name_to_index_map.find(current_name);

        if (it != name_to_index_map.end()) {
            res[it->second] = vec[i];
        } else {
            name_to_index_map[current_name] = i;
            res.push_back(vec[i]);
        }
    }

    // vec.resize(name_to_index_map.size());

    return res;
}

bool ModelLoader::load_tensors(on_new_tensor_cb_t on_new_tensor_cb, ggml_backend_t backend) {
    std::vector<TensorStorage> processed_tensor_storages;
    for (auto& tensor_storage : tensor_storages) {
        // LOG_DEBUG("%s", name.c_str());

        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        preprocess_tensor(tensor_storage, processed_tensor_storages);
    }
    std::vector<TensorStorage> dedup = remove_duplicates(processed_tensor_storages);
    processed_tensor_storages        = dedup;

    bool success = true;
    for (size_t file_index = 0; file_index < file_paths_.size(); file_index++) {
        std::string file_path = file_paths_[file_index];
        LOG_DEBUG("loading tensors from %s", file_path.c_str());

        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            LOG_ERROR("failed to open '%s'", file_path.c_str());
            return false;
        }

        bool is_zip = false;
        for (auto& tensor_storage : tensor_storages) {
            if (tensor_storage.file_index != file_index) {
                continue;
            }
            if (tensor_storage.index_in_zip >= 0) {
                is_zip = true;
                break;
            }
        }

        struct zip_t* zip = NULL;
        if (is_zip) {
            zip = zip_open(file_path.c_str(), 0, 'r');
            if (zip == NULL) {
                LOG_ERROR("failed to open zip '%s'", file_path.c_str());
                return false;
            }
        }

        std::vector<uint8_t> read_buffer;
        std::vector<uint8_t> convert_buffer;

        auto read_data = [&](const TensorStorage& tensor_storage, char* buf, size_t n) {
            if (zip != NULL) {
                zip_entry_openbyindex(zip, tensor_storage.index_in_zip);
                size_t entry_size = zip_entry_size(zip);
                if (entry_size != n) {
                    read_buffer.resize(entry_size);
                    zip_entry_noallocread(zip, (void*)read_buffer.data(), entry_size);
                    memcpy((void*)buf, (void*)(read_buffer.data() + tensor_storage.offset), n);
                } else {
                    zip_entry_noallocread(zip, (void*)buf, n);
                }
                zip_entry_close(zip);
            } else {
                file.seekg(tensor_storage.offset);
                file.read(buf, n);
                if (!file) {
                    LOG_ERROR("read tensor data failed: '%s'", file_path.c_str());
                    return false;
                }
            }
            return true;
        };
        int tensor_count = 0;
        int64_t t1       = ggml_time_ms();
        for (auto& tensor_storage : processed_tensor_storages) {
            if (tensor_storage.file_index != file_index) {
                ++tensor_count;
                continue;
            }
            ggml_tensor* dst_tensor = NULL;

            success = on_new_tensor_cb(tensor_storage, &dst_tensor);
            if (!success) {
                LOG_WARN("process tensor failed: '%s'", tensor_storage.name.c_str());
                break;
            }

            if (dst_tensor == NULL) {
                ++tensor_count;
                continue;
            }

            size_t nbytes_to_read = tensor_storage.nbytes_to_read();

            if (dst_tensor->buffer == NULL || ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                // for the CPU and Metal backend, we can copy directly into the tensor
                if (tensor_storage.type == dst_tensor->type) {
                    GGML_ASSERT(ggml_nbytes(dst_tensor) == tensor_storage.nbytes());
                    read_data(tensor_storage, (char*)dst_tensor->data, nbytes_to_read);

                    if (tensor_storage.is_bf16) {
                        // inplace op
                        bf16_to_f32_vec((uint16_t*)dst_tensor->data, (float*)dst_tensor->data, tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e4m3) {
                        // inplace op
                        f8_e4m3_to_f16_vec((uint8_t*)dst_tensor->data, (uint16_t*)dst_tensor->data, tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e5m2) {
                        // inplace op
                        f8_e5m2_to_f16_vec((uint8_t*)dst_tensor->data, (uint16_t*)dst_tensor->data, tensor_storage.nelements());
                    }
                } else {
                    read_buffer.resize(tensor_storage.nbytes());
                    read_data(tensor_storage, (char*)read_buffer.data(), nbytes_to_read);

                    if (tensor_storage.is_bf16) {
                        // inplace op
                        bf16_to_f32_vec((uint16_t*)read_buffer.data(), (float*)read_buffer.data(), tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e4m3) {
                        // inplace op
                        f8_e4m3_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e5m2) {
                        // inplace op
                        f8_e5m2_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                    }

                    convert_tensor((void*)read_buffer.data(), tensor_storage.type, dst_tensor->data,
                                   dst_tensor->type, (int)tensor_storage.nelements() / (int)tensor_storage.ne[0], (int)tensor_storage.ne[0]);
                }
            } else {
                read_buffer.resize(tensor_storage.nbytes());
                read_data(tensor_storage, (char*)read_buffer.data(), nbytes_to_read);

                if (tensor_storage.is_bf16) {
                    // inplace op
                    bf16_to_f32_vec((uint16_t*)read_buffer.data(), (float*)read_buffer.data(), tensor_storage.nelements());
                } else if (tensor_storage.is_f8_e4m3) {
                    // inplace op
                    f8_e4m3_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                } else if (tensor_storage.is_f8_e5m2) {
                    // inplace op
                    f8_e5m2_to_f16_vec((uint8_t*)read_buffer.data(), (uint16_t*)read_buffer.data(), tensor_storage.nelements());
                }

                if (tensor_storage.type == dst_tensor->type) {
                    // copy to device memory
                    ggml_backend_tensor_set(dst_tensor, read_buffer.data(), 0, ggml_nbytes(dst_tensor));
                } else {
                    // convert first, then copy to device memory
                    convert_buffer.resize(ggml_nbytes(dst_tensor));
                    convert_tensor((void*)read_buffer.data(), tensor_storage.type,
                                   (void*)convert_buffer.data(), dst_tensor->type,
                                   (int)tensor_storage.nelements() / (int)tensor_storage.ne[0], (int)tensor_storage.ne[0]);
                    ggml_backend_tensor_set(dst_tensor, convert_buffer.data(), 0, ggml_nbytes(dst_tensor));
                }
            }
            int64_t t2 = ggml_time_ms();
            pretty_progress(++tensor_count, processed_tensor_storages.size(), (t2 - t1) / 1000.0f);
            t1 = t2;
        }

        if (zip != NULL) {
            zip_close(zip);
        }

        if (!success) {
            break;
        }
    }
    return success;
}

bool ModelLoader::load_tensors(std::map<std::string, struct ggml_tensor*>& tensors,
                               ggml_backend_t backend,
                               std::set<std::string> ignore_tensors) {
    std::set<std::string> tensor_names_in_file;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        // LOG_DEBUG("%s", tensor_storage.to_string().c_str());
        tensor_names_in_file.insert(name);

        struct ggml_tensor* real;
        if (tensors.find(name) != tensors.end()) {
            real = tensors[name];
        } else {
            for (auto& ignore_tensor : ignore_tensors) {
                if (starts_with(name, ignore_tensor)) {
                    return true;
                }
            }
            LOG_INFO("unknown tensor '%s' in model file", tensor_storage.to_string().c_str());
            return true;
        }

        if (
            real->ne[0] != tensor_storage.ne[0] ||
            real->ne[1] != tensor_storage.ne[1] ||
            real->ne[2] != tensor_storage.ne[2] ||
            real->ne[3] != tensor_storage.ne[3]) {
            LOG_ERROR(
                "tensor '%s' has wrong shape in model file: "
                "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                name.c_str(),
                (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                (int)real->ne[0], (int)real->ne[1], (int)real->ne[2], (int)real->ne[3]);
            return false;
        }

        *dst_tensor = real;

        return true;
    };

    bool success = load_tensors(on_new_tensor_cb, backend);
    if (!success) {
        LOG_ERROR("load tensors from file failed");
        return false;
    }

    bool some_tensor_not_init = false;

    for (auto pair : tensors) {
        if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
            LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
            some_tensor_not_init = true;
        }
    }

    if (some_tensor_not_init) {
        return false;
    }
    return true;
}

bool ModelLoader::tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) {
    const std::string& name = tensor_storage.name;
    if (type != GGML_TYPE_COUNT) {
        if (ggml_is_quantized(type) && tensor_storage.ne[0] % ggml_blck_size(type) != 0) {
            // Pass, do not convert
            LOG_ERROR("tensor_storage.ne[0] is not a multiple of ggml_blck_size(type)");
        } else if (contains(name, "blocks") &
                    (ends_with(name, "attn.qkv.weight") ||
                    ends_with(name, "attn.proj.weight") ||
                    ends_with(name, "mlp.fc1.weight") ||
                    ends_with(name, "mlp.fc2.weight"))) {
            return true;
        }
    }
    return false;
}

bool ModelLoader::save_to_gguf_file(const std::string& file_path, ggml_type type) {
    auto backend    = ggml_backend_cpu_init();
    size_t mem_size = 1 * 1024 * 1024;  // for padding
    mem_size += tensor_storages.size() * ggml_tensor_overhead();
    mem_size += get_params_mem_size(backend, type);
    LOG_INFO("model tensors mem size: %.2fMB", mem_size / 1024.f / 1024.f);
    ggml_context* ggml_ctx = ggml_init({mem_size, NULL, false});

    gguf_context* gguf_ctx = gguf_init_empty();

    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;

        ggml_type tensor_type = tensor_storage.type;
        if (tensor_should_be_converted(tensor_storage, type)) {
            tensor_type = type;
        }

        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, tensor_type, tensor_storage.n_dims, tensor_storage.ne);
        if (tensor == NULL) {
            LOG_ERROR("ggml_new_tensor failed");
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        // LOG_DEBUG("%s %d %s %d[%d %d %d %d] %d[%d %d %d %d]", name.c_str(),
        // ggml_nbytes(tensor), ggml_type_name(tensor_type),
        // tensor_storage.n_dims,
        // tensor_storage.ne[0], tensor_storage.ne[1], tensor_storage.ne[2], tensor_storage.ne[3],
        // tensor->n_dims, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

        *dst_tensor = tensor;

        gguf_add_tensor(gguf_ctx, tensor);

        return true;
    };

    bool success = load_tensors(on_new_tensor_cb, backend);
    ggml_backend_free(backend);
    LOG_INFO("load tensors done");
    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    if (success) {
        gguf_write_to_file(gguf_ctx, file_path.c_str(), false);
    }
    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);
    return success;
}

int64_t ModelLoader::get_params_mem_size(ggml_backend_t backend, ggml_type type) {
    size_t alignment = 128;
    if (backend != NULL) {
        alignment = ggml_backend_get_alignment(backend);
    }
    int64_t mem_size = 0;
    std::vector<TensorStorage> processed_tensor_storages;
    for (auto& tensor_storage : tensor_storages) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }
        preprocess_tensor(tensor_storage, processed_tensor_storages);
    }

    for (auto& tensor_storage : processed_tensor_storages) {
        if (tensor_should_be_converted(tensor_storage, type)) {
            tensor_storage.type = type;
        }
        mem_size += tensor_storage.nbytes() + alignment;
    }

    return mem_size;
}
