#ifndef BLADE_PIPELINES_VLA_MODE_B_HH
#define BLADE_PIPELINES_VLA_MODE_B_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/beamformer/vla.hh"

namespace Blade::Pipelines::VLA {

template<typename OT = CF16>
class BLADE_API ModeB : public Pipeline {
 public:
    struct Config {
        U64 numberOfAntennas;
        U64 numberOfFrequencyChannels;
        U64 numberOfTimeSamples;
        U64 numberOfPolarizations;

        U64 channelizerRate;

        U64 beamformerBeams;

        U64 outputMemWidth;
        U64 outputMemPad;

        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 phasorsBlockSize = 512;
        U64 beamformerBlockSize = 512;
    };

    explicit ModeB(const Config& config);

    constexpr const U64 getInputSize() const {
        return channelizer->getBufferSize();
    }

    constexpr const U64 getPhasorsSize() const {
        return beamformer->getPhasorsSize();
    }

    constexpr const U64 getOutputSize() const {
        return (((beamformer->getOutputSize() * sizeof(OT)) / 
            config.outputMemWidth) * outputMemPitch) / sizeof(OT);
    }

    Result run(const Vector<Device::CPU, CI8>& input,
               const Vector<Device::CPU, CF32>& phasors,
                     Vector<Device::CPU, OT>& output);

 private:
    const Config config;

    U64 outputMemPitch;

    Vector<Device::CUDA, CI8> input;
    Vector<Device::CUDA, CF32> phasors;

    std::shared_ptr<Modules::Cast<CI8, CF32>> inputCast;
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    std::shared_ptr<Modules::Beamformer::VLA<CF32, CF32>> beamformer;
    std::shared_ptr<Modules::Cast<CF32, OT>> outputCast;

    constexpr const Vector<Device::CUDA, OT>& getOutput() {
        if constexpr (!std::is_same<OT, CF32>::value) {
            // output is casted output
            return outputCast->getOutput();
        } else {
            // output is un-casted beamformer output (CF32)
            return beamformer->getOutput();
        }
    }
};

}  // namespace Blade::Pipelines::VLA

#endif