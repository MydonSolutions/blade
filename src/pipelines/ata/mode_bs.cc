#define BL_LOG_DOMAIN "P::ATA::MODE_BS"

#include "blade/pipelines/ata/mode_bs.hh"

namespace Blade::Pipelines::ATA {

ModeBS::ModeBS(const Config& config) : config(config), blockJulianDate({1}), blockDut1({1}) {
    BL_DEBUG("Initializing ATA Pipeline Mode B.");

    BL_DEBUG("Allocating pipeline buffers.");
    BL_CHECK_THROW(this->input.resize(config.inputDimensions));

    BL_DEBUG("Instantiating input cast from I8 to CF32.");
    this->connect(inputCast, {
        .blockSize = config.castBlockSize,
    }, {
        .buf = this->input,
    });

    BL_DEBUG("Instantiating pre-beamformer channelizer with rate {}.",
            config.preBeamformerChannelizerRate);
    this->connect(channelizer, {
        .rate = config.preBeamformerChannelizerRate,

        .blockSize = config.channelizerBlockSize,
    }, {
        .buf = inputCast->getOutputBuffer(),
    });

    BL_DEBUG("Instantiating phasor module.");
    this->connect(phasor, {
        .numberOfAntennas = channelizer->getOutputBuffer().dims().numberOfAspects(),
        .numberOfFrequencyChannels = channelizer->getOutputBuffer().dims().numberOfFrequencyChannels(),
        .numberOfPolarizations = channelizer->getOutputBuffer().dims().numberOfPolarizations(),

        .observationFrequencyHz = config.phasorObservationFrequencyHz,
        .channelBandwidthHz = config.phasorChannelBandwidthHz,
        .totalBandwidthHz = config.phasorTotalBandwidthHz,
        .frequencyStartIndex = config.phasorFrequencyStartIndex,

        .referenceAntennaIndex = config.phasorReferenceAntennaIndex,
        .arrayReferencePosition = config.phasorArrayReferencePosition,
        .boresightCoordinate = config.phasorBoresightCoordinate,
        .antennaPositions = config.phasorAntennaPositions,
        .antennaCoefficients = config.phasorAntennaCoefficients,
        .beamCoordinates = config.phasorBeamCoordinates,

        .preBeamformerChannelizerRate = config.preBeamformerChannelizerRate,

        .blockSize = config.phasorBlockSize,
    }, {
        .blockJulianDate = this->blockJulianDate,
        .blockDut1 = this->blockDut1,
    });

    BL_DEBUG("Instantiating beamformer module.");
    this->connect(beamformer, {
        .enableIncoherentBeam = config.beamformerIncoherentBeam,
        .enableIncoherentBeamSqrt = true,

        .blockSize = config.beamformerBlockSize,
    }, {
        .buf = channelizer->getOutputBuffer(),
        .phasors = phasor->getOutputPhasors(),
    });

    BL_DEBUG("Instantiating detector module.");
    this->connect(detector, {
        .integrationSize = config.detectorIntegrationSize,
        .numberOfOutputPolarizations = config.detectorNumberOfOutputPolarizations,

        .blockSize = config.detectorBlockSize,
    }, {
        .buf = beamformer->getOutputBuffer(),
    });

    this->connect(dedoppler, {
        .mitigateDcSpike = config.searchMitigateDcSpike,
        .observationBandwidthHz = config.phasorTotalBandwidthHz,
        .minimumDriftRate = config.searchMinimumDriftrate,
        .maximumDriftRate = config.searchMaximumDriftrate,
        .snrThreshold = config.searchSnrThreshold,
        .blockSize = config.searchBlockSize,
    }, {
        .buf = detector->getOutputBuffer(),
    });
}

const Result ModeBS::transferIn(const Vector<Device::CPU, F64>& blockJulianDate,
                                   const Vector<Device::CPU, F64>& blockDut1,
                                   const ArrayTensor<Device::CPU, CI8>& input,
                                   const cudaStream_t& stream) { 
    // Copy input to static buffers.
    BL_CHECK(Memory::Copy(this->blockJulianDate, blockJulianDate));
    BL_CHECK(Memory::Copy(this->blockDut1, blockDut1));
    BL_CHECK(Memory::Copy(this->input, input, stream));

    // Print dynamic arguments on first run.
    if (this->getCurrentComputeStep() == 0) {
        BL_DEBUG("Block Julian Date: {}", this->blockJulianDate[0]);
        BL_DEBUG("Block DUT1: {}", this->blockDut1[0]);
    }

    return Result::SUCCESS;
}

}  // namespace Blade::Pipelines::ATA