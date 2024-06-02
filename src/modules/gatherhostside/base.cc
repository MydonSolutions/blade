#define BL_LOG_DOMAIN "M::GATHERHOSTSIDE"

#include <type_traits>
#include <typeindex>

#include "blade/modules/gatherhostside.hh"

#include "gatherhostside.jit.hh"

namespace Blade::Modules {

template<typename IT, typename OT>
GatherHostside<IT, OT>::GatherHostside(const Config& config,
                       const Input& input,
                       const Stream& stream)
        : Module(gatherhostside_program),
          config(config),
          input(input),
          computeRatio(config.multiplier) {
    if constexpr (!std::is_same<IT, OT>::value) {
        BL_FATAL("Input ({}) and output ({}) types aren't the same. Casting isn't supported by GatherHostside.",
                 TypeInfo<IT>::name, TypeInfo<OT>::name);
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.axis >= input.buf.shape().dimensions()) {
        BL_FATAL("Selected input axis ({}) is larger than input shape dimensions ({}).",
                 config.axis, input.buf.shape().dimensions());
        BL_CHECK_THROW(Result::ERROR);
    }

    if (config.multiplier <= 0) {
        BL_FATAL("Multiplier ({}) should be more than zero.", config.multiplier);
        BL_CHECK_THROW(Result::ERROR);
    }

    widthSize = 1;
    for (U64 i = config.axis; i < input.buf.shape().dimensions(); i++) {
        widthSize *= input.buf.shape()[i];
    }
    widthByteSize = widthSize * sizeof(IT);
    BL_DEBUG("Width size of {} elements.", widthSize);
    BL_DEBUG("Step copy size of {} bytes.", widthByteSize);

    heightSize = 1;
    for (U64 i = 0; i < config.axis; i++) {
        heightSize *= input.buf.shape()[i];
    }
    BL_DEBUG("Height size of {} elements.", heightSize);

    // Allocate output buffers.
    output.buf = ArrayTensor<Device::CPU, OT>(getOutputBufferShape());

    // Print configuration values.

    BL_INFO("Type: {} -> {}", TypeInfo<IT>::name, TypeInfo<OT>::name);
    BL_INFO("Shape: {} -> {}", getInputBuffer().shape(),
                               getOutputBuffer().shape());
    BL_INFO("Axis: {}", config.axis);
    BL_INFO("Multiplier: {}", computeRatio);
}

template<typename IT, typename OT>
Result GatherHostside<IT, OT>::process(const U64& currentStepCount, const Stream& stream) {
    BL_DEBUG("currentStepCount: {}...", currentStepCount);

    BL_CHECK(
        Copy2D(
            output.buf,
            widthByteSize * computeRatio,
            widthByteSize * currentStepCount,

            input.buf,
            widthByteSize,
            0,

            widthByteSize,
            heightSize,
            stream
        )
    );

    return Result::SUCCESS;
}

template class BLADE_API GatherHostside<CI8, CI8>;
template class BLADE_API GatherHostside<CF16, CF16>;
template class BLADE_API GatherHostside<CF32, CF32>;
template class BLADE_API GatherHostside<F32, F32>;

}  // namespace Blade::Modules
