#ifndef BLADE_MODULES_GATHERHOSTSIDE_GENERIC_HH
#define BLADE_MODULES_GATHERHOSTSIDE_GENERIC_HH

#include "blade/base.hh"
#include "blade/module.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
class BLADE_API GatherHostside : public Module {
 public:
    // Configuration

    struct Config {
        U64 axis = 0;
        U64 multiplier = 1;

        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    struct Input {
        const ArrayTensor<Device::CUDA, IT>& buf;
    };

    constexpr const ArrayTensor<Device::CUDA, IT>& getInputBuffer() const {
        return this->input.buf;
    }

    // Output 

    struct Output {
        ArrayTensor<Device::CPU, OT> buf;
    };

    constexpr const ArrayTensor<Device::CPU, OT>& getOutputBuffer() const {
        return this->output.buf;
    }

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::CONSUMER |
               Taint::PRODUCER |
               Taint::CHRONOUS;
    }

    constexpr U64 getComputeRatio() const {
        return computeRatio;
    }

    std::string name() const {
        return "GatherHostside";
    }

    // Constructor & Processing

    explicit GatherHostside(const Config& config, const Input& input, const Stream& stream = {});
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

 private:
    // Variables

    const Config config;
    const Input input;
    Output output;

    U64 computeRatio;
    U64 widthSize;
    U64 widthByteSize;
    U64 heightSize;

    // Expected Shape

    const ArrayShape getOutputBufferShape() {
        ArrayShape::Type shape = getInputBuffer().shape();
        shape[config.axis] *= config.multiplier;
        return shape;
    }
};

}  // namespace Blade::Modules

#endif
