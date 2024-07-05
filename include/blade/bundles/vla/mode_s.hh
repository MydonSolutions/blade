#ifndef BLADE_BUNDLES_VLA_MODE_S_HH
#define BLADE_BUNDLES_VLA_MODE_S_HH

#include "blade/bundle.hh"

#include "blade/modules/permutation.hh"
#include "blade/modules/gatherhostside.hh"
#include "blade/modules/seticore/dedoppler.hh"
#include "blade/modules/seticore/hits_stamp_writer.hh"

namespace Blade::Bundles::VLA {

class BLADE_API ModeS : public Bundle {
 public:
    // Configuration

    struct Config {
        U64 inputTelescopeId;
        std::string inputSourceName;
        std::string inputObservationIdentifier;
        RA_DEC inputPhaseCenter;
        U64 inputTotalNumberOfTimeSamples;
        U64 inputTotalNumberOfFrequencyChannels;
        F64 inputFrequencyOfFirstChannelHz;
        U64 inputCoarseStartChannelIndex;
        U64 inputCoarseChannelRatio = 1;
        BOOL inputLastBeamIsIncoherent = false;

        std::vector<std::string> beamNames;
        std::vector<RA_DEC> beamCoordinates;

        BOOL searchMitigateDcSpike;
        BOOL searchDriftRateZeroExcluded = false;
        F64 searchMinimumDriftRate = 0.0;
        F64 searchMaximumDriftRate;
        F64 searchSnrThreshold;
        F64 searchStampFrequencyMarginHz;
        I64 searchHitsGroupingMargin;
        BOOL searchIncoherentBeam = true;

        F64 searchChannelBandwidthHz;
        F64 searchChannelTimespanS;
        std::string searchOutputFilepathStem;
        std::vector<double> searchExclusionSubbandBottomsMHz;
        std::vector<double> searchExclusionSubbandTopsMHz;

        BOOL produceDebugHits = false;

        U64 dedopplerBlockSize = 512;
    };

    constexpr const Config& getConfig() const {
        return this->config;
    }

    // Input

    
    struct Input {
        const ArrayTensor<Device::CUDA, F32>& beamformedData;
        const ArrayTensor<Device::CUDA, CF32>& prebeamformerData;
        const Tensor<Device::CPU, U64>& coarseFrequencyChannelOffset;
        const Tensor<Device::CPU, F64>& frequencyOfFirstChannelHz;
        const Tensor<Device::CPU, F64>& julianDateStart;
    };

    constexpr const ArrayTensor<Device::CUDA, F32>& getInputData() const {
        return this->input.beamformedData;
    }

    constexpr const Tensor<Device::CPU, F64>& getInputFrequencyOfFirstChannelHz() {
        return this->input.frequencyOfFirstChannelHz;
    }

    constexpr const Tensor<Device::CPU, F64>& getInputJulianDate() {
        return this->input.julianDateStart;
    }

    // Output

    // Constructor

    explicit ModeS(const Config& config, const Input& input, const Stream& stream)
         : Bundle(stream), config(config), input(input) {
        BL_DEBUG("Initializing Mode-S Bundle.");

        BL_DEBUG("Instantiating beam-input permutation from AFTP to ATPF .");
        this->connect(inputBeamPermutation, {
            .indexes = ArrayShape({0, 3, 1, 2}), // standard AFTP in -> ATPF for dedoppler search
        }, {
            .buf = input.beamformedData
        });

        BL_DEBUG("Instantiating beam-input gather-hostside along T.");
        this->connect(inputBeamGatherHostside, {
            .axis = 1,
            .multiplier = config.inputTotalNumberOfTimeSamples / input.beamformedData.shape().numberOfTimeSamples(),
        }, {
            .buf = inputBeamPermutation->getOutputBuffer()
        });

        BL_DEBUG("Instantiating antenna-input permutation from AFTP to TPFA.");
        this->connect(inputAntennaPermutation, {
            .indexes = ArrayShape({3, 1, 2, 0}), // standard AFTP in -> TPFA for stamp writing
        }, {
            .buf = input.prebeamformerData
        });

        BL_DEBUG("Instantiating antenna-input gather-hostside along T.");
        this->connect(inputAntennaGatherHostside, {
            .axis = 0,
            .multiplier = config.inputTotalNumberOfTimeSamples / input.beamformedData.shape().numberOfTimeSamples(),
        }, {
            .buf = inputAntennaPermutation->getOutputBuffer()
        });

        BL_DEBUG("Instantiating dedoppler module.");
        F64 minimumDriftRate = config.searchMinimumDriftRate;
        if (config.searchDriftRateZeroExcluded && minimumDriftRate == 0.0) {
            // set the minimum to at least the drift-rate-resolution:
            // Calculation taken from the private Dedoppler.drift_rate_resolution.
            minimumDriftRate = config.searchChannelBandwidthHz / (input.beamformedData.shape().numberOfTimeSamples() * config.searchChannelTimespanS);
            BL_INFO("Set the minimum drift rate of the dedoppler search to the search's resolution of {} Hz/s to exclude zero.", minimumDriftRate);
        }
        this->connect(dedoppler, {
            .mitigateDcSpike = config.searchMitigateDcSpike,
            .minimumDriftRate = minimumDriftRate,
            .maximumDriftRate = config.searchMaximumDriftRate,
            .snrThreshold = config.searchSnrThreshold,
            .frequencyOfFirstChannelHz = config.inputFrequencyOfFirstChannelHz,
            .channelBandwidthHz = config.searchChannelBandwidthHz,
            .channelTimespanS = config.searchChannelTimespanS,
            .coarseChannelRate = config.inputCoarseChannelRatio,
            .lastBeamIsIncoherent = config.inputLastBeamIsIncoherent,
            .searchIncoherentBeam = config.searchIncoherentBeam,

            .searchExclusionSubbandBottomsMHz = config.searchExclusionSubbandBottomsMHz,
            .searchExclusionSubbandTopsMHz = config.searchExclusionSubbandTopsMHz,

            // hits writer requirements -_-
            .filepathPrefix = config.searchOutputFilepathStem,
            .telescopeId = config.inputTelescopeId,
            .sourceName = config.inputSourceName,
            .observationIdentifier = config.inputObservationIdentifier,
            .phaseCenter = config.inputPhaseCenter,
            .aspectNames = config.beamNames,
            .aspectCoordinates = config.beamCoordinates,
            .totalNumberOfTimeSamples = config.inputTotalNumberOfTimeSamples,
            .totalNumberOfFrequencyChannels = config.inputTotalNumberOfFrequencyChannels,

            .produceDebugHits = config.produceDebugHits,
        }, {
            .bufATPF = inputBeamGatherHostside->getOutputBuffer(),
            .coarseFrequencyChannelOffset = input.coarseFrequencyChannelOffset,
            .julianDate = input.julianDateStart,
        });

        BL_DEBUG("Instantiating stamps-writer module.");
        this->connect(hitsStampWriter, {
            .filepathPrefix = config.searchOutputFilepathStem,
            .telescopeId = config.inputTelescopeId,
            .sourceName = config.inputSourceName,
            .observationIdentifier = config.inputObservationIdentifier,
            .phaseCenter = config.inputPhaseCenter,
            .coarseStartChannelIndex = config.inputCoarseStartChannelIndex,
            .coarseChannelRatio = config.inputCoarseChannelRatio,
            .channelBandwidthHz = config.searchChannelBandwidthHz,
            .channelTimespanS = config.searchChannelTimespanS,
            .stampFrequencyMarginHz = config.produceDebugHits ? 0.0 : config.searchStampFrequencyMarginHz,
            .hitsGroupingMargin = config.produceDebugHits ? -((I64) config.inputCoarseChannelRatio) : config.searchHitsGroupingMargin,
        }, {
            .bufferTFPA = inputAntennaGatherHostside->getOutputBuffer(),
            .hits = this->dedoppler->getOutputHits(),
            .frequencyOfFirstChannelHz = input.frequencyOfFirstChannelHz,
            .julianDateStart = input.julianDateStart,
        });
    }

 private:
    const Config config;
    Input input;

    // Input.data: AFTP >-permutation-> ATPF >-gatherhostside(T)-> 
    using InputBeamPermutation = typename Modules::Permutation<F32, F32>;
    std::shared_ptr<InputBeamPermutation> inputBeamPermutation;
    std::shared_ptr<Modules::GatherHostside<F32, F32>> inputBeamGatherHostside;

    // Input.prebeamformerData: AFTP >-permutation-> TFPA >-gatherhostside(T)-> 
    using InputAntennaPermutation = typename Modules::Permutation<CF32, CF32>;
    std::shared_ptr<InputAntennaPermutation> inputAntennaPermutation;
    std::shared_ptr<Modules::GatherHostside<CF32, CF32>> inputAntennaGatherHostside;

    std::shared_ptr<Modules::Seticore::Dedoppler> dedoppler;

    using HitsStampWriter = typename Modules::Seticore::HitsStampWriter<CF32>;
    std::shared_ptr<HitsStampWriter> hitsStampWriter;
};

}  // namespace Blade::Bundles::Generic

#endif
