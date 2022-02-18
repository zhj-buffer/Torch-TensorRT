# Torch-TensorRT

[![Documentation](https://img.shields.io/badge/docs-master-brightgreen)](https://nvidia.github.io/Torch-TensorRT/)

> Ahead of Time (AOT) compiling for PyTorch JIT

Torch-TensorRT is a compiler for PyTorch/TorchScript, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime. Unlike PyTorch's Just-In-Time (JIT) compiler, Torch-TensorRT is an Ahead-of-Time (AOT) compiler, meaning that before you deploy your TorchScript code, you go through an explicit compile step to convert a standard TorchScript program into an module targeting a TensorRT engine. Torch-TensorRT operates as a PyTorch extention and compiles modules that integrate into the JIT runtime seamlessly. After compilation using the optimized graph should feel no different than running a TorchScript module. You also have access to TensorRT's suite of configurations at compile time, so you are able to specify operating precision (FP32/FP16/INT8) and other settings for your module.

More Information / System Architecture:

- [GTC 2020 Talk](https://developer.nvidia.com/gtc/2020/video/s21671)




## Example Usage

### C++

```c++
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

...
// Set input datatypes. Allowerd options torch::{kFloat, kHalf, kChar, kInt32, kBool}
// Size of input_dtypes should match number of inputs to the network.
// If input_dtypes is not set, default precision follows traditional PyT / TRT rules
auto input = torch_tensorrt::Input(dims, torch::kHalf)
auto compile_settings = torch_tensorrt::ts::CompileSpec({input});
// FP16 execution
compile_settings.enabled_precisions = {torch::kHalf};
// Compile module
auto trt_mod = torch_tensorrt::ts::compile(ts_mod, compile_settings);
// Run like normal
auto results = trt_mod.forward({in_tensor});
// Save module for later
trt_mod.save("trt_torchscript_module.ts");
...
```

### Python

```py
import torch_tensorrt

...

trt_ts_module = torch_tensorrt.compile(torch_script_module,
    inputs = [example_tensor, # Provide example tensor for input shape or...
        torch_tensorrt.Input( # Specify input object with shape and dtype
            min_shape=[1, 3, 224, 224],
            opt_shape=[1, 3, 512, 512],
            max_shape=[1, 3, 1024, 1024],
            # For static size shape=[1, 3, 224, 224]
            dtype=torch.half) # Datatype of input tensor. Allowed options torch.(float|half|int8|int32|bool)
    ],
    enabled_precisions = {torch.half}, # Run with FP16)

result = trt_ts_module(input_data) # run inference
torch.jit.save(trt_ts_module, "trt_torchscript_module.ts") # save the TRT embedded Torchscript
```

> Notes on running in lower precisions:
>
> - Enabled lower precisions with compile_spec.enabled_precisions
> - The module should be left in FP32 before compilation (FP16 can support half tensor models)
> - Provided input tensors dtype should be the same as module before compilation, regardless of `enabled_precisions`. This can be overrided by setting `Input::dtype`

## Platform Support

| Platform            | Support                                          |
| ------------------- | ------------------------------------------------ |
| Linux AMD64 / GPU   | **Supported**                                    |
| Linux aarch64 / GPU | **Native Compilation Supported on JetPack-4.4+** |
| Linux aarch64 / DLA | **Native Compilation Supported on JetPack-4.4+** |
| Windows / GPU       | **Unofficial Support**                           |
| Linux ppc64le / GPU | -                                                |
| NGC Containers      | **Included in PyTorch NGC Containers 21.11+**   |

> Torch-TensorRT will be included in NVIDIA NGC containers (https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) starting in 21.11.

> Note: Refer NVIDIA NGC container(https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch) for PyTorch libraries on JetPack.

### Dependencies

These are the following dependencies used to verify the testcases. Torch-TensorRT can work with other versions, but the tests are not guaranteed to pass.

- Bazel 4.2.1
- Libtorch 1.10.0 (built with CUDA 10.2)
- CUDA 10.2 (10.2 on Jetson)
- cuDNN 8.2.1.32-1+cuda10.2
- TensorRT 8.0.1.6-1+cuda10.2 ( on Jetson)

## Compiling Torch-TensorRT

### Installing Dependencies

#### 0. Install Bazel or Using the precompiled binary under tools

If you don't have bazel installed, the easiest way is to install bazelisk using the method of you choosing https://github.com/bazelbuild/bazelisk

Otherwise you can use the following instructions to install binaries https://docs.bazel.build/versions/master/install.html

Finally if you need to compile from source (e.g. aarch64 until bazel distributes binaries for the architecture) you can use these instructions

```sh
export BAZEL_VERSION=<VERSION>
mkdir bazel
cd bazel
curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-dist.zip
unzip bazel-$BAZEL_VERSION-dist.zip
bash ./compile.sh
```

You need to start by having CUDA installed on the system, LibTorch will automatically be pulled for you by bazel,
then you have two options.

#### 1. Building using locally installed cuDNN & TensorRT

> If you find bugs and you compiled using this method please disclose you used this method in the issue
> (an `ldd` dump would be nice too)

1. Install TensorRT, CUDA and cuDNN on the system before starting to compile.
2. In `WORKSPACE` comment out

```py
# Downloaded distributions to use with --distdir
http_archive(
    name = "cudnn",
    urls = ["<URL>",],

    build_file = "@//third_party/cudnn/archive:BUILD",
    sha256 = "<TAR SHA256>",
    strip_prefix = "cuda"
)

http_archive(
    name = "tensorrt",
    urls = ["<URL>",],

    build_file = "@//third_party/tensorrt/archive:BUILD",
    sha256 = "<TAR SHA256>",
    strip_prefix = "TensorRT-<VERSION>"
)
```

and uncomment

```py
# Locally installed dependencies
new_local_repository(
    name = "cudnn",
    path = "/usr/",
    build_file = "@//third_party/cudnn/local:BUILD"
)

new_local_repository(
   name = "tensorrt",
   path = "/usr/",
   build_file = "@//third_party/tensorrt/local:BUILD"
)
```

3. Compile tools:
``` shell
sudo cp /tools/bazel /usr/bin/
sudo apt install openjdk-11-jdk -y
```
4. Dependency：
Refer to https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048
``` shell
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython numy
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision
export BUILD_VERSION=v0.11.1  # where 0.x.0 is the torchvision version  
python3 setup.py install --user
cd ../  # attempting to load torchvision from build dir will result in import error
pip install 'pillow<7' # always needed for Python 2.7, not needed torchvision v0.5.0+ with Python 3.6

```
### Native compilation on NVIDIA Jetson AGX
We performed end to end testing on Jetson platform using Jetpack SDK 4.6.

``` shell
bazel build //:libtorchtrt --platforms //toolchains:jetpack_4.6
```

### Debug build

``` shell
bazel build //:libtorchtrt --compilation_mode=dbg  --platforms //toolchains:jetpack_4.6
```

### Debug build

``` shell
cd py
python3 setup.py install --use-cxx11-abi
```

### Test with Python3

``` shell
cd ..
❯ torchtrtc -p f16 lenet_scripted.ts trt_lenet_scripted.ts "(1,1,32,32)"

❯ python3
Python 3.6.9 (default, Apr 18 2020, 01:56:04)
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> import torch_tensorrt
>>> ts_model = torch.jit.load(“trt_lenet_scripted.ts”)
>>> ts_model(torch.randn((1,1,32,32)).to(“cuda”).half())

python3 examples/test.py
```
> Note: Please refer [installation](docs/tutorials/installation.html) instructions for Pre-requisites

A tarball with the include files and library can then be found in bazel-bin

## How do I add support for a new op...

### In Torch-TensorRT?

Thanks for wanting to contribute! There are two main ways to handle supporting a new op. Either you can write a converter for the op from scratch and register it in the NodeConverterRegistry or if you can map the op to a set of ops that already have converters you can write a graph rewrite pass which will replace your new op with an equivalent subgraph of supported ops. Its preferred to use graph rewriting because then we do not need to maintain a large library of op converters. Also do look at the various op support trackers in the [issues](https://github.com/NVIDIA/Torch-TensorRT/issues) for information on the support status of various operators.

### In my application?

> The Node Converter Registry is not exposed in the top level API but in the internal headers shipped with the tarball.

You can register a converter for your op using the `NodeConverterRegistry` inside your application.

## Structure of the repo

| Component                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| [**core**](core)         | Main JIT ingest, lowering, conversion and runtime implementations |
| [**cpp**](cpp)           | C++ API and CLI source                                       |
| [**examples**](examples) | Example applications to show different features of Torch-TensorRT |
| [**py**](py)             | Python API for Torch-TensorRT                                |
| [**tests**](tests)       | Unit tests for Torch-TensorRT                                |

## Contributing

Take a look at the [CONTRIBUTING.md](CONTRIBUTING.md)


## License

The Torch-TensorRT license can be found in the LICENSE file. It is licensed with a BSD Style licence
