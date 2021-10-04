#include "blade/kernels/checker.hh"

#include "checker.jit.hh"

namespace Blade::Kernel {

Checker::Checker(const Config & config) : config(config), cache(100, *checker_kernel) {
    BL_DEBUG("Initilizating class.");

    if (config.block > 1024) {
        BL_FATAL("Maximum block length is 1024.");
        throw Result::ERROR;
    }

    block = dim3(config.block);
    grid = dim3((config.len + block.x - 1) / block.x);

    if (cudaMallocManaged(&counter, sizeof(unsigned long long int)) != cudaSuccess) {
        BL_FATAL("Can't allocate CUDA memory for counter.");
        throw Result::ERROR;
    }
    *counter = 0;
}

Checker::~Checker() {
    BL_DEBUG("Destroying class.");
    cudaFree(counter);
}

unsigned long long int Checker::run(const std::complex<float>* input, const std::complex<float>* output) {
    auto kernel = Template("checker_complex").instantiate(config.len);

    *counter = 0;
    cache
        .get_kernel(kernel)
        ->configure(grid, block)
        ->launch(
            reinterpret_cast<const cuFloatComplex*>(input),
            reinterpret_cast<const cuFloatComplex*>(output),
            counter
        );

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return -1;
    });

    cudaDeviceSynchronize();

    return *counter;
}

unsigned long long int Checker::run(const float* input, const float* output) {
    auto kernel = Template("checker").instantiate(Type<float>(), config.len);

    *counter = 0;
    cache
        .get_kernel(kernel)
        ->configure(grid, block)
        ->launch(input, output, counter);

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return -1;
    });

    cudaDeviceSynchronize();

    return *counter;
}

unsigned long long int Checker::run(const int8_t* input, const int8_t* output) {
    auto kernel = Template("checker").instantiate(Type<int8_t>(), config.len);

    *counter = 0;
    cache
        .get_kernel(kernel)
        ->configure(grid, block)
        ->launch(input, output, counter);

    BL_CUDA_CHECK_KERNEL([&]{
        BL_FATAL("Kernel failed to execute: {}", err);
        return -1;
    });

    cudaDeviceSynchronize();

    return *counter;
}

} // namespace Blade::Kernel
