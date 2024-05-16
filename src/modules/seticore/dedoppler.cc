#define BL_LOG_DOMAIN "M::DEDOPPLER"

#include "blade/modules/seticore/dedoppler.hh"

#include "dedoppler.jit.hh"

namespace Blade::Modules::Seticore {

Dedoppler::Dedoppler(const Config& config, const Input& input)
        : Module(dedoppler_program),
          config(config),
          input(input),
          dedopplerer(
            input.buf.shape().numberOfTimeSamples(),
            input.buf.shape().numberOfFrequencyChannels(),
            1e-6 * this->config.channelBandwidthHz,
            this->config.channelTimespanS,
            config.mitigateDcSpike
          ) {

    this->metadata.source_name = this->config.sourceName;
    this->metadata.fch1 = this->config.frequencyOfFirstChannelHz*1e-6; // MHz
    this->metadata.foff = this->config.channelBandwidthHz*1e-6; // MHz
    this->metadata.tsamp = this->config.channelTimespanS;
    this->metadata.tstart = 0.0; // from MJD, dynamically set
    this->metadata.src_raj = this->config.phaseCenter.RA * 12.0 / BL_PHYSICAL_CONSTANT_PI; // hours
    this->metadata.src_dej = this->config.phaseCenter.DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI; // degrees
    this->metadata.num_timesteps = input.buf.shape().numberOfTimeSamples();
    this->metadata.num_channels = input.buf.shape().numberOfFrequencyChannels();
    this->metadata.telescope_id = this->config.telescopeId;
    this->metadata.coarse_channel_size = this->config.coarseChannelRate;
    this->metadata.num_coarse_channels = metadata.num_channels / metadata.coarse_channel_size;
    this->metadata.source_names = this->config.aspectNames;
    this->metadata.ras = std::vector<F64>();
    this->metadata.decs = std::vector<F64>();
    this->metadata.has_dc_spike = config.mitigateDcSpike;

    for (const RA_DEC& coord : this->config.aspectCoordinates) {
        this->metadata.ras.push_back(coord.RA * 12.0 / BL_PHYSICAL_CONSTANT_PI); // hours
        this->metadata.decs.push_back(coord.DEC * 180.0 / BL_PHYSICAL_CONSTANT_PI); // degrees
    }

    string output_filename = bl::fmt::format("{}.seticore.hits", config.filepathPrefix);
    auto hfw = new HitFileWriter(output_filename, metadata);
    hfw->verbose = false;
    hit_recorder.reset(hfw);

    const auto inputShape = this->input.buf.shape();
    // Search buffer is reduce to a single aspect
    this->searchBuffer = ArrayTensor<Device::CUDA, F32>({
        1,
        inputShape.numberOfFrequencyChannels(),
        inputShape.numberOfTimeSamples(),
        inputShape.numberOfPolarizations()
    });

    if (this->config.lastBeamIsIncoherent) {
        this->incohBuffer = ArrayTensor<Device::CUDA, F32>({
            1,
            inputShape.numberOfFrequencyChannels(),
            inputShape.numberOfTimeSamples(),
            inputShape.numberOfPolarizations()
        });
    }

    BL_INFO("Input Shape: {} -> {}", inputShape, "N/A");
    if (this->config.produceDebugHits) {
        BL_INFO("Producing exhaustive hits for debugging: {}", this->config.produceDebugHits);
    }
    else {
        BL_INFO("Coarse Channel Rate: {}", this->config.coarseChannelRate);
        BL_INFO("Channel Bandwidth: {} Hz", this->config.channelBandwidthHz);
        BL_INFO("Channel Timespan: {} s", this->config.channelTimespanS);
        BL_INFO("Drift Rate Range: [{}, {}] Hz/s", this->config.minimumDriftRate, this->config.maximumDriftRate);
        BL_INFO("SNR Threshold: {}", this->config.snrThreshold);
    }

    const auto t = input.buf.shape().numberOfTimeSamples();
    const auto t_previousPowerOf2 = t == 0 ? 0 : (0x80000000 >> __builtin_clz(t));
    if (t != t_previousPowerOf2) {
        BL_FATAL("Dedoppler must be provided a power of 2 timesamples!");
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }
}

Result Dedoppler::process(const cudaStream_t& stream) {
    this->output.hits.clear();
    const auto inputDims = this->input.buf.shape();
    
    this->metadata.tstart = this->input.julianDate[0] - 2400000.5; // from JD to MJD
    
    const auto skipLastBeam = this->config.lastBeamIsIncoherent & (!this->config.searchIncoherentBeam);
    const auto beamsToSearch = inputDims.numberOfAspects() - (skipLastBeam ? 1 : 0);

    const auto beamByteStride = this->input.buf.size_bytes() / inputDims.numberOfAspects();
    const auto beamElementStride = this->input.buf.size() / inputDims.numberOfAspects();
    size_t hits_after_last_beam = 0;

    // search exclusion variables
    double hitFrequencyStartMHz;
    double hitFrequencyStopMHz;
    double hitFrequencyHitSpanHz;
    // std::vector<double>::iterator lowerBoundTopsIndex;
    int exclusionSubbandIndex;

    if (this->config.produceDebugHits) {
        const double drift_rate_resolution = this->config.channelBandwidthHz / ((inputDims.numberOfTimeSamples()-1) * this->config.channelTimespanS);
        // the stamp.data list-field has some upper limit, try stay under that
        // data is [time, chan, pol, ant, complexity]
        // assume 128 antenna (conservative)
        const int stamp_data_length_limit = 128*1024*1024;
        const int hit_drift_step_limit = stamp_data_length_limit/(inputDims.numberOfTimeSamples()*2*128*2);
        BL_DEBUG("Limited debug drift step to {}.", hit_drift_step_limit);

        for (U64 beam = 0; beam < beamsToSearch; beam++) {
            for (U64 fine_channel_index = 0; fine_channel_index < inputDims.numberOfFrequencyChannels(); fine_channel_index += hit_drift_step_limit) {
                const int hit_drift_steps = fine_channel_index + hit_drift_step_limit >= inputDims.numberOfFrequencyChannels() 
                    ? inputDims.numberOfFrequencyChannels() - fine_channel_index
                    : hit_drift_step_limit;

                this->output.hits.push_back(DedopplerHit(
                    this->metadata,
                    fine_channel_index, // index
                    hit_drift_steps, // drift_steps
                    hit_drift_steps*drift_rate_resolution, // drift_rate
                    0.0, // snr
                    beam, // beam
                    this->input.coarseFrequencyChannelOffset[0], // coarse_channel
                    inputDims.numberOfTimeSamples(), // num_timesteps
                    0.0 // power
                ));
            }

            BL_DEBUG("Beamformed data #{}: {}", beam, this->input.buf.data()[beam*beamElementStride]);
            for (size_t i = hits_after_last_beam; i < this->output.hits.size(); i++) {
                const DedopplerHit& hit = this->output.hits[i];
                BL_DEBUG("Hit: {}", hit.toString());
                hit_recorder->recordHit(hit, this->input.buf.data() + beam*beamElementStride);
            }

            hits_after_last_beam = this->output.hits.size();
        }

        // drop all hits after those for the first hit, to avoid duplicate stamps
        const int single_beam_hit_count = this->output.hits.size()/beamsToSearch;
        this->output.hits.erase(
            std::next(this->output.hits.begin(), single_beam_hit_count),
            std::next(this->output.hits.begin(), this->output.hits.size())
        );

        return Result::SUCCESS;
    }

    FilterbankBuffer beamFilterbankBuffer = FilterbankBuffer(
        inputDims.numberOfTimeSamples(),
        inputDims.numberOfFrequencyChannels(),
        this->searchBuffer.data()
    );

    FilterbankBuffer incohbeamFilterbankBuffer = FilterbankBuffer(
        inputDims.numberOfTimeSamples(),
        inputDims.numberOfFrequencyChannels(),
        this->incohBuffer.data()
    );
    
    if (this->config.lastBeamIsIncoherent) {
        BL_CHECK(Copy2D(
            this->incohBuffer,
            beamByteStride,
            0,
            
            this->input.buf,
            beamByteStride,
            (inputDims.numberOfAspects()-1)*beamByteStride,

            beamByteStride,
            1,

            cudaMemcpyHostToDevice,
            {0} // seticore runs on the default stream
        ));
    }

    BL_DEBUG("processing {} beams", beamsToSearch);
    for (U64 beam = 0; beam < beamsToSearch; beam++) {
        // misuse 2D copy just to effect offset on data copy
        // copy from RAM to VRAM
        BL_CHECK(Copy2D(
            this->searchBuffer,
            beamByteStride,
            0,
            
            this->input.buf,
            beamByteStride,
            beam*beamByteStride,

            beamByteStride,
            1,

            cudaMemcpyHostToDevice,
            {0} // seticore runs on the default stream
        ));

        std::vector<DedopplerHit> beam_hits;
        dedopplerer.search(
            beamFilterbankBuffer,
            this->metadata,
            beam,
            this->input.coarseFrequencyChannelOffset[0],
            this->config.maximumDriftRate,
            this->config.minimumDriftRate,
            this->config.snrThreshold,
            &beam_hits
        );

        if (this->config.lastBeamIsIncoherent) {
            dedopplerer.addIncoherentPower(
                incohbeamFilterbankBuffer,
                beam_hits
            );
        }

        for (const DedopplerHit& hit : beam_hits) {
            hitFrequencyStartMHz = hit.frequency;
            hitFrequencyStopMHz = hit.frequency;
            hitFrequencyHitSpanHz = hit.drift_rate * hit.drift_steps;
            if (hitFrequencyHitSpanHz >= 0) {
                hitFrequencyStopMHz += hitFrequencyHitSpanHz*1e-6;
            }
            else {
                hitFrequencyStartMHz += hitFrequencyHitSpanHz*1e-6;
            }

            // assuming exclusion-subbands are non-overlapping,
            // only ommit the hit if is in a single exclusion-subband
            auto lowerBoundTopsIndex = std::lower_bound(
                config.searchExclusionSubbandTopsMHz.begin(), 
                config.searchExclusionSubbandTopsMHz.end(),
                hitFrequencyStopMHz
            );
            if (lowerBoundTopsIndex != config.searchExclusionSubbandTopsMHz.end()) {
                exclusionSubbandIndex = std::distance(config.searchExclusionSubbandTopsMHz.begin(), lowerBoundTopsIndex);
                if (config.searchExclusionSubbandBottomsMHz.at(exclusionSubbandIndex) <= hitFrequencyStartMHz) {
                    BL_DEBUG(
                        "Hit spanning {}->{} omitted, falling into exclusion-subband #{} [{}, {}]: {}",
                        hitFrequencyStartMHz,
                        hitFrequencyStopMHz,
                        exclusionSubbandIndex,
                        config.searchExclusionSubbandBottomsMHz.at(exclusionSubbandIndex),
                        config.searchExclusionSubbandTopsMHz.at(exclusionSubbandIndex),
                        hit.toString()
                    );
                    continue;
                }
            }

            BL_DEBUG("Hit spanning {}->{}: {}", hit.frequency, hit.frequency+(hit.drift_rate*hit.drift_steps*1e-6), hit.toString());
            hit_recorder->recordHit(hit, this->input.buf.data() + beam*beamElementStride);
            this->output.hits.push_back(hit);
        }
        hits_after_last_beam = this->output.hits.size();
    }

    BL_CUDA_CHECK(cudaStreamSynchronize(0), [&]{
        BL_FATAL("Failed to synchronize default stream: {}", err);
    });

    return Result::SUCCESS;
}

} // namespace Blade::Modules::Seticore
