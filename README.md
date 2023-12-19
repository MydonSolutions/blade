# ⚔️ BLADE
### Breakthrough Listen Accelerated DSP Engine

<p align="center">
<img src="docs/IMG_7960.jpeg" />
</p>

- 🌌 Optimized for high-throughput signal processing for radio telescopes.
- 📡 Used in production in world-class radio telescopes like the Allen Telescope Array.
- 💻 Deep CUDA integration for superior performance and efficiency.
- 🐍 Complete Python bindings for seamless integration with scientific computing.
- 🧰 Comprehensive C++20 support for robust and maintainable codebase.
- 🔄 Modular DSP architecture for flexible and extensible signal processing solutions.
- 🚀 Just-in-time (JIT) compilation of CUDA kernels for maximum performance and runtime customizability.

BLADE (Breakthrough Listen Accelerated DSP Engine) is a modern, high-performance signal processing library for radio telescopes, such as the Allen Telescope Array. Written in modern C++20, it utilizes CUDA for accelerated processing, Python bindings for ease of use, and just-in-time (JIT) compilation of CUDA kernels for runtime customizability. BLADE is designed to be flexible and extensible to support other radio telescopes. 

The library structures into Modules, Pipelines, and Runners:

- **Modules**: Handle data manipulation and processing (e.g. Beamformer, Channelizer, Polarizer, etc).
- **Pipelines**: Integrate Modules to form a processing pipeline (e.g. Beamformer + Channelizer).
- **Runners**: Execute Pipelines asynchronously for optimal parallelization.

Currently, BLADE implements the following Modules:

- **Beamformer**: Performs beamforming on a set of antennas.
- **Channelizer**: Converts a time series into a frequency series.
- **Detector**: Integrates and calculates the Stokes-I.
- **Polarizer**: Converts a horizontal and vertical polarization into a left and right circular polarization.
- **Cast**: Converts the input data type to another data type.
- **Gather**: Tiles the input data into a larger Tensor axis.
- **Duplicate**: Copies the input data into a new Tensor.
- **Permutation**: Transposes the input data axis into a new order.

All frequency values are in Hertz and all angles are in radians!

## Installation
Don't worry, it is not difficult! Follow the instructions below to compile it on your system. Keep in mind that BLADE requires a Linux system with a NVIDIA Graphics Card. A Docker image is also available for building and testing BLADE. To build it, run `docker build -t blade .` and to run it, run `docker run -rm -it --gpus all blade bash`. The Docker image is based on Ubuntu 22.04 and contains all the dependencies required to build and test BLADE. The [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) is required to run this image.

### Step 1: Dependencies
BLADE requires a C++20 compiler (>GCC-10 or >Clang 13.0), the [Meson](https://mesonbuild.com) build system, [Ninja Build](https://ninja-build.org), and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Follow the instructions below to install the dependencies and build BLADE from source.

#### Ubuntu 22.04
Core dependencies (you probably already have them).
```bash
$ apt install git build-essential cmake pkg-config ninja-build meson git
```

Modules dependencies.
```bash
$ apt install liberfa-dev libhdf5-dev
```

Python dependencies (optional).
```bash
$ apt install python3-dev python3-pip
```

Test and benchmark dependencies (optional).
```bash
$ apt install libbenchmark-dev libgtest-dev 
```

```bash
$ python3 -m pip install numpy astropy pandas
```

Finally, to install the CUDA Toolkit, follow the official NVIDIA instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu). Make sure to install the CUDA Toolkit version 11.4 or higher.

### Step 2: Install
There are two ways to install BLADE. The first one will only install only the Python bindings and the second one will install the Python bindings and the C++ library. If you plan to use BLADE exclusively from Python, it's recommend installing the Python bindings. Otherwise, install the C++ library.

#### Python Library
```bash
$ pip install git+https://github.com/luigifcruz/blade.git
```

#### C++ and Python Library
Clone the repository from Github.
```bash
$ git clone https://github.com/luigifcruz/blade.git
$ cd blade
```

Build and install `release` version.
```bash
$ meson build && cd build
$ ninja install
```

Or build and install `debug` version.
```bash
$ meson -Dbuildtype=debugoptimized build && cd build
$ ninja install
```

## About
BLADE was created originally as the beamforming engine for the Allen Telescope Array (ATA). Since then, the library has grown to support other workloads like the High-Resolution Spectrometer (HRS). The library is designed to be flexible and extensible to support other radio telescopes with ongoing efforts to upstream the Very Large Array (VLA) COSMIC support. The library is written in modern C++20 and makes use of just-in-time (JIT) compilation of CUDA kernels to deliver accelerated processing with runtime customizability. Performant Python bindings are also available. Regular talks about BLADE were given at multiple conferences and are available [here](https://luigi.ltd/talks/).

## Contributing
Contributions are welcome! Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests and invite you to submit pull requests directly in this repository. The library follows the [Google C++ Code Style Guide](https://google.github.io/styleguide/cppguide.html). The default line length is 88. This can be overridden if necessary. Please, be sensible.

### License
BLADE is distributed under the [MIT license](./LICENSE.md). See [LICENSE.md](./LICENSE.md) for details. All contributions to the project are considered to be licensed under the same terms. If you have any questions, please contact [Luigi Cruz](https://luigi.ltd/contact).


```
                                     .-.
                    .-""`""-.      |(@ @)
                 _/`oOoOoOoOo`\_   \ \-/
                '.-=-=-=-=-=-=-.'   \/ \
                  `-=.=-.-=.=-'      \ /\
                     ^  ^  ^         _H_ \ art by jgs
```
