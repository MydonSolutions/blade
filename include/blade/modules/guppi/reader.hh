#ifndef BLADE_MODULES_GUPPI_READER_HH
#define BLADE_MODULES_GUPPI_READER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "guppirawc99.h"
#include "radiointerferometryc99.h"
}

namespace Blade::Modules::Guppi {

template<typename OT>
class BLADE_API Reader : public Module {
 public:
    // Configuration 

    struct Config {
        std::string filepath;
        U64 stepNumberOfTimeSamples;
        U64 stepNumberOfFrequencyChannels;

        U64 requiredMultipleOfTimeSamplesSteps = 1;
        U64 numberOfTimeSampleStepsBeforeFrequencyChannelStep = 1;
        U64 numberOfFilesLimit = 0; // zero for no limit
        U64 blockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input 

    struct Input {
    };

    // Output

    struct Output {
        ArrayTensor<Device::CPU, OT> stepBuffer;
        Tensor<Device::CPU, F64> stepJulianDate;
        Tensor<Device::CPU, F64> stepDut1;
        Tensor<Device::CPU, U64> stepFrequencyChannelOffset;
    };

    constexpr const ArrayTensor<Device::CPU, OT>& getStepOutputBuffer() const {
        return this->output.stepBuffer;
    }

    F64 getUnixDateOfLastReadBlock(const U64 timesamplesOffset = 0);

    constexpr F64 getJulianDateOfLastReadBlock(const U64 timesamplesOffset = 0) {
        return calc_julian_date_from_unix_sec(this->getUnixDateOfLastReadBlock(timesamplesOffset));
    }

    constexpr const Tensor<Device::CPU, F64>& getStepOutputJulianDate() const {
        return this->output.stepJulianDate;
    }

    constexpr const Tensor<Device::CPU, F64>& getStepOutputDut1() const {
        return this->output.stepDut1;
    }

    constexpr const Tensor<Device::CPU, U64>& getStepOutputFrequencyChannelOffset() const {
        return this->output.stepFrequencyChannelOffset;
    }

    ArrayShape getTotalOutputBufferShape() const {
        return ArrayShape({
            this->getDatashape()->n_aspect,
            this->getDatashape()->n_aspectchan,
            this->getDatashape()->n_time * this->gr_iterate.n_block,
            this->getDatashape()->n_pol,
        });
    }

    ArrayShape getStepOutputBufferShape() const {
        return ArrayShape({
            this->getDatashape()->n_aspect,
            this->config.stepNumberOfFrequencyChannels,
            this->config.stepNumberOfTimeSamples,
            this->getDatashape()->n_pol,
        });
    }

    const ArrayShape getNumberOfStepsInDimensions() const {
        auto dimensionSteps = this->getTotalOutputBufferShape() / this->getStepOutputBufferShape();
        auto timesamples = dimensionSteps.numberOfTimeSamples();
        if (this->config.numberOfTimeSampleStepsBeforeFrequencyChannelStep > 0) {
            timesamples -= dimensionSteps.numberOfTimeSamples() % this->config.numberOfTimeSampleStepsBeforeFrequencyChannelStep;
        }
        timesamples -= dimensionSteps.numberOfTimeSamples() % this->config.requiredMultipleOfTimeSamplesSteps;
        return ArrayShape({
            dimensionSteps.numberOfAspects(),
            dimensionSteps.numberOfFrequencyChannels(),
            timesamples,
            dimensionSteps.numberOfPolarizations()
        });
    }

    const U64 getNumberOfSteps() {
        return this->getNumberOfStepsInDimensions().size();
    }

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::PRODUCER; 
    }

    std::string name() const {
        return "Guppi Reader";
    }

    // Constructor & Processing

    explicit Reader(const Config& config, const Input& input, const Stream& stream = {});
    Result process(const U64& currentStepCount, const Stream& stream = {}) final;

    // Miscellaneous 

    F64 getObservationBandwidth() const;
    F64 getChannelBandwidth() const;
    F64 getChannelTimespan() const;
    U64 getChannelStartIndex() const;
    F64 getObservationCenterFrequency() const;
    F64 getCenterFrequency() const;
    F64 getObservationBottomFrequency() const;
    F64 getBottomFrequency() const;
    F64 getObservationTopFrequency() const;
    F64 getTopFrequency() const;
    F64 getBandwidth() const;
    F64 getAzimuthAngle() const;
    F64 getZenithAngle() const;
    F64 getRightAscension() const;
    F64 getDeclination() const;
    F64 getPhaseRightAscension() const;
    F64 getPhaseDeclination() const;
    std::string getSourceName() const;
    std::string getTelescopeName() const;

 private:
    // Variables 

    Config config;
    const Input input;
    Output output;

    U64 lastread_channel_index = 0;
    U64 lastread_block_index = 0;
    U64 lastread_time_index = 0;

    U64 current_time_sample_step = 0;
    U64 checkpoint_block_index = 0;
    U64 checkpoint_time_index = 0;

    guppiraw_iterate_info_t gr_iterate = {0};

    // Helpers

    const bool keepRunning() const {
        return guppiraw_iterate_ntime_remaining(&this->gr_iterate) >= 
                this->config.stepNumberOfTimeSamples;
    }

    const guppiraw_datashape_t* getDatashape() const {
        return guppiraw_iterate_datashape(&this->gr_iterate);
    }
};

}  // namespace Blade::Modules

#endif
