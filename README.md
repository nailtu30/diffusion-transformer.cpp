# diffusion-transformer.cpp

Inference of Diffusion Transformer (DiT) in C++ based on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/tree/master) and [llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master).

## Usage
```
git clone --recursive https://github.com/nailtu30/diffusion-transformer.cpp
cd diffusion-transformer.cpp
```
build with CPU
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
build with CUDA
```
mkdir build
cd build
cmake .. -DDIT_CUDA=ON
cmake --build . --config Release
```
