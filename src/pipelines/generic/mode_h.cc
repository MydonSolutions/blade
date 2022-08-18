#include "blade/pipelines/generic/mode_h.hh"

namespace Blade::Pipelines::Generic {

template<typename IT, typename OT>
ModeH<IT, OT>::ModeH(const Config& config)
     : Accumulator(config.accumulateRate),
       config(config) {
    BL_DEBUG("Initializing Pipeline Mode H.");

    if constexpr (!std::is_same<IT, CF32>::value) {
        BL_DEBUG("Instantiating input cast from CF16 to CF32.");
        this->connect(cast, {
            .inputSize = this->getInputSize(),
            .blockSize = config.castBlockSize,
        }, {
            .buf = this->input,
        });
    }

    BL_DEBUG("Instantiating channelizer with rate {}.", config.channelizerNumberOfTimeSamples *  
                                                        config.accumulateRate);
    this->connect(channelizer, {
        .numberOfBeams = config.channelizerNumberOfBeams,
        .numberOfAntennas = 1,
        .numberOfFrequencyChannels = config.channelizerNumberOfFrequencyChannels,
        .numberOfTimeSamples = config.channelizerNumberOfTimeSamples * config.accumulateRate,
        .numberOfPolarizations = config.channelizerNumberOfPolarizations,
        .rate = config.channelizerNumberOfTimeSamples * config.accumulateRate,
        .blockSize = config.channelizerBlockSize,
    }, {
        .buf = this->getChannelizerInput(),
    });

    BL_DEBUG("Instantiating detector module.");
    this->connect(detector, {
        .numberOfBeams = config.channelizerNumberOfBeams, 
        .numberOfFrequencyChannels = config.channelizerNumberOfFrequencyChannels * 
                                     config.channelizerNumberOfTimeSamples * 
                                     config.accumulateRate,
        .numberOfTimeSamples = 1,
        .numberOfPolarizations = config.channelizerNumberOfPolarizations,

        .integrationSize = 1,
        .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

        .blockSize = config.detectorBlockSize,
    }, {
        .buf = channelizer->getOutput(),
    });
}

template<typename IT, typename OT>
const Result ModeH<IT, OT>::accumulate(const Vector<Device::CUDA, IT>& data,
                                       const cudaStream_t& stream) {
    if (this->accumulationComplete()) {
        BL_FATAL("Can't accumulate block because buffer is full.");
        return Result::BUFFER_FULL;
    }

    // TODO: Check if this copy parameters are correct.
    const auto& width = (data.size() / config.channelizerNumberOfBeams / config.channelizerNumberOfFrequencyChannels) * sizeof(IT);
    const auto& height = config.channelizerNumberOfBeams * config.channelizerNumberOfFrequencyChannels;

    BL_CHECK(
        Memory::Copy2D(
            this->input,
            width * this->getAccumulatorNumberOfSteps(),
            0,
            data,
            width,
            0,
            width,
            height, 
            stream));

    this->incrementAccumulatorStep();

    return Result::SUCCESS;
}

template<typename IT, typename OT>
const Result ModeH<IT, OT>::run(Vector<Device::CPU, OT>& output) {
    if (!this->accumulationComplete()) {
        BL_FATAL("Can't run compute because acumulator buffer is incomplete ({}/{}).", 
            this->getCurrentAccumulatorStep(), this->getAccumulatorNumberOfSteps());
        return Result::BUFFER_INCOMPLETE;
    }

    BL_CHECK(this->compute());
    BL_CHECK(this->copy(output, this->getOutput()));

    this->resetAccumulatorSteps();

    return Result::SUCCESS;
}

template class BLADE_API ModeH<CF16, F32>;
template class BLADE_API ModeH<CF32, F32>;

}  // namespace Blade::Pipelines::Generic