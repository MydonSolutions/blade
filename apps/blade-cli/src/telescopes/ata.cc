#include "blade-cli/telescopes/ata/base.hh"

namespace Blade::CLI::Telescopes::ATA {

const Result Setup(const Blade::CLI::Telescopes::ATA::Config& config) {
    switch (config.mode) {
#if defined(BLADE_PIPELINE_ATA_MODE_B)
        case ModeId::MODE_B:
            switch (config.outputType) {
                case TypeId::CF16:
                    return ModeB<CI8, CF16>(config);
                case TypeId::CF32:
                    return ModeB<CI8, CF32>(config);
                case TypeId::F16:
                    return ModeB<CI8, F16>(config);
                case TypeId::F32:
                    return ModeB<CI8, F32>(config);
                default:
                    BL_FATAL("This ATA output is not implemented yet.");    
            }
#endif
#if defined(BLADE_PIPELINE_GENERIC_MODE_H) && defined(BLADE_PIPELINE_ATA_MODE_B) && defined(BLADE_PIPELINE_GENERIC_MODE_S)
        case ModeId::MODE_BS:
            switch (config.outputType) {
                case TypeId::CF32:
                    return ModeBS<CI8, CF32>(config);
                case TypeId::F32:
                    return ModeBS<CI8, F32>(config);
                default:
                    BL_FATAL("This ATA output is not implemented yet.");    
            }
#endif
#if defined(BLADE_PIPELINE_ATA_MODE_B) && defined(BLADE_PIPELINE_GENERIC_MODE_H)
#endif
        default:
            BL_FATAL("This ATA mode is not implemented yet.");
    }

    return Result::ERROR;
}


const Result CollectUserInput(int argc, char **argv, Config& config) {
    ::CLI::App app("BLADE (Breakthrough Listen Accelerated DSP Engine) - Command Line Tool - ATA");

    // Read target telescope. 
    app
        .add_option("-t,--telescope", config.telescope, "Telescope ID (ATA)")
            ->required()
            ->transform(::CLI::CheckedTransformer(TelescopeMap, ::CLI::ignore_case));

    // Read target mode. 

    app
        .add_option("-m,--mode", config.mode, "Mode ID (MODE_B, MODE_BS)")
            ->required()
            ->transform(::CLI::CheckedTransformer(ModeMap, ::CLI::ignore_case));

    // Read input GUPPI RAW file.
    app
        .add_option("-i,--input,input", config.inputGuppiFile, "Input GUPPI RAW filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read input BFR5 file.
    app
        .add_option("-r,--recipe,recipe", config.inputBfr5File, "Input BFR5 filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read output GUPPI RAW filepath.
        app
        .add_option("-o,--output,output", config.outputFile, "Output filepath")
            ->required()
            ->capture_default_str()
            ->run_callback_for_default();

    // Read number of workers.
    app
        .add_option("-N,--number-of-workers", config.numberOfWorkers, "Number of workers")
            ->default_val(2);

    // Read target pre-beamformer channelizer rate.
    app
        .add_option("-c,--pre-beamformer-channelizer-rate", config.preBeamformerChannelizerRate, 
                "Pre-beamformer channelizer rate (FFT-size)")
            ->default_val(1024);

    // Read target step number of time samples.
    app
        .add_option("-T,--step-number-of-time-samples", config.stepNumberOfTimeSamples, 
                "Step number of time samples")
            ->default_val(32);

    // Read target step number of frequency channels.
    app
        .add_option("-C,--step-number-of-frequency-channels", config.stepNumberOfFrequencyChannels,
                "Step number of frequency channels")
            ->default_val(32);

    // Read input type format.
    app
        .add_option("--input-type", config.inputType, "Input type format (CI8, CF16, CF32, I8, F16, F32)")
            ->required()
            ->transform(::CLI::CheckedTransformer(TypeMap, ::CLI::ignore_case));

    // Read output type format.
    app
        .add_option("--output-type", config.outputType, "Output type format (CI8, CF16, CF32, I8, F16, F32)")
            ->required()
            ->transform(::CLI::CheckedTransformer(TypeMap, ::CLI::ignore_case));

    // Read target SNR Threshold.
    app
        .add_option("-s,--search-snr-threshold", config.snrThreshold,
                "SETI search SNR threshold")
            ->default_val(6.0);

    // Read target drift rate range.
    app
        .add_option("-d,--search-drift-rate-minimum", config.driftRateMinimum,
                "SETI search drift rate minimum")
            ->default_val(0.0);
    app
        .add_option("-D,--search-drift-rate-maximum", config.driftRateMaximum,
                "SETI search drift rate maximum")
            ->default_val(50.0);
    
    config.driftRateZeroExcluded = false;
    app
        .add_flag("-Z,--search-drift-rate-exclude-zero,!-z,!--search-drift-rate-include-zero", config.driftRateZeroExcluded,
                "SETI search exclude hits with drift rate of zero");

    try {
        app.parse(argc, argv);
    } catch(const ::CLI::ParseError &e) {
        app.exit(e);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}


}  // namespace Blade::CLI::Telescopes::ATA
