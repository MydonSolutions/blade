#ifndef BLADE_PIPELINES_ATA_MODE_BS_HH
#define BLADE_PIPELINES_ATA_MODE_BS_HH

#include <memory>
#include <deque>

#include "blade/pipeline.hh"

#include "blade/modules/cast.hh"
#include "blade/modules/channelizer.hh"
#include "blade/modules/beamformer/ata.hh"
#include "blade/modules/phasor/ata.hh"
#include "blade/modules/detector.hh"
#include "blade/modules/seticore/dedoppler.hh"

namespace Blade::Pipelines::ATA {

// TODO: Add input types.

class BLADE_API ModeBS : public Pipeline {
 public:
    // Configuration 

    struct Config {
        ArrayTensorDimensions inputDimensions;

        U64 preBeamformerChannelizerRate;

        F64 phasorObservationFrequencyHz;
        F64 phasorChannelBandwidthHz;
        F64 phasorTotalBandwidthHz;
        U64 phasorFrequencyStartIndex;
        U64 phasorReferenceAntennaIndex;
        LLA phasorArrayReferencePosition; 
        RA_DEC phasorBoresightCoordinate;
        std::vector<XYZ> phasorAntennaPositions;
        std::vector<CF64> phasorAntennaCoefficients; 
        std::vector<RA_DEC> phasorBeamCoordinates;

        BOOL beamformerIncoherentBeam = false;

        U64 detectorIntegrationSize;
        U64 detectorNumberOfOutputPolarizations;

        BOOL searchMitigateDcSpike = true;
        F64 searchMinimumDriftrate = 0.0;
        F64 searchMaximumDriftrate;
        F64 searchSnrThreshold;

        U64 castBlockSize = 512;
        U64 channelizerBlockSize = 512;
        U64 phasorBlockSize = 512;
        U64 beamformerBlockSize = 512;
        U64 detectorBlockSize = 512;
        U64 searchBlockSize = 512;
    };

    // Input

    const Result transferIn(const Vector<Device::CPU, F64>& blockJulianDate,
                            const Vector<Device::CPU, F64>& blockDut1,
                            const ArrayTensor<Device::CPU, CI8>& input,
                            const cudaStream_t& stream);

    constexpr const ArrayTensor<Device::CUDA, CI8>& getInputBuffer() const {
        return input;
    }

    // Output 

    const ArrayTensor<Device::CUDA, F32>& getOutputBuffer() {
        return detector->getOutputBuffer();
    }

    // Constructor

    explicit ModeBS(const Config& config);

 private:
    const Config config;

    ArrayTensor<Device::CUDA, CI8> input;
    Vector<Device::CPU, F64> blockJulianDate;
    Vector<Device::CPU, F64> blockDut1;

    std::shared_ptr<Modules::Cast<CI8, CF32>> inputCast;
    std::shared_ptr<Modules::Channelizer<CF32, CF32>> channelizer;
    std::shared_ptr<Modules::Phasor::ATA<CF32>> phasor;
    std::shared_ptr<Modules::Beamformer::ATA<CF32, CF32>> beamformer;
    std::shared_ptr<Modules::Detector<CF32, F32>> detector;
    std::shared_ptr<Modules::Seticore::Dedoppler> dedoppler;
};

}  // namespace Blade::Pipelines::ATA

#endif
