#define BL_LOG_DOMAIN "M::BFR5"

#include "blade/modules/bfr5/reader.hh"

#include "bfr5.jit.hh"

namespace Blade::Modules::Bfr5 {

Reader::Reader(const Config& config,
               const Input& input,
               const Stream& stream) 
        : Module(bfr5_program),
          config(config),
          input(input) {
    // Check configuration values.
    if (!std::filesystem::exists(config.filepath)) {
        BL_FATAL("Input file ({}) doesn't not exist.", config.filepath);
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    // Read HDF5 file into memory.
    BFR5open(config.filepath.c_str(), &this->bfr5);
    BFR5read_all(&this->bfr5);
    BFR5close(&this->bfr5);

    // Resize data holders.
    auto shape = getShape();
    beamCoordinates.resize(this->bfr5.beam_info.ra_elements);
    antennaPositions.resize(shape.numberOfAntennas());

    if (Profiler::IsCapturing()) {
        BL_WARN("Capturing: Early setup return.");
        return;
    }
     
    // Extract beam coordinates and names.
    for (U64 i = 0; i < this->bfr5.beam_info.ra_elements; i++) {
        beamCoordinates[i].RA = this->bfr5.beam_info.ras[i];
        beamCoordinates[i].DEC = this->bfr5.beam_info.decs[i];
        beamSourceNames[i] = std::string(
            this->bfr5.beam_info.src_names[i]
        );
    }

    // Calculate antenna positions.
    const U64 antennaPositionsByteSize = shape.numberOfAntennas() * sizeof(XYZ);
    std::memcpy(antennaPositions.data(), this->bfr5.tel_info.antenna_positions, antennaPositionsByteSize);

    std::string antFrame = std::string(this->bfr5.tel_info.antenna_position_frame);

    if (antFrame != "xyz" && antFrame != "XYZ") {
        if (antFrame == "ecef" || antFrame == "ECEF") {
            BL_DEBUG("Translating antenna positions from ECEF to XYZ.")
            calc_position_to_xyz_frame_from_ecef(
                reinterpret_cast<F64*>(antennaPositions.data()),
                antennaPositions.size(),
                this->bfr5.tel_info.latitude,
                this->bfr5.tel_info.longitude,
                this->bfr5.tel_info.altitude);
        }

        if (antFrame == "enu" || antFrame == "ENU") {
            BL_DEBUG("Translating antenna positions from ENU to XYZ.")
            calc_position_to_xyz_frame_from_enu(
                reinterpret_cast<F64*>(antennaPositions.data()),
                antennaPositions.size(),
                this->bfr5.tel_info.latitude,
                this->bfr5.tel_info.longitude,
                this->bfr5.tel_info.altitude);
        }

        else {
            BL_FATAL("Unknown antenna position frame '{}'. Expecting ECEF, XYZ or ENU.", antFrame);
            BL_CHECK_THROW(Result::ASSERTION_ERROR);
        }
    }

    // Print configuration buffers.
    BL_INFO("Input File Path: {}", config.filepath);
    BL_INFO("Data Shape: {} -> {}", "N/A", shape);
}

std::vector<CF64> Reader::getAntennaCoefficients(const U64& numberOfFrequencyChannels, const U64& frequencyChannelStartIndex) {
    // transpose from F,P,A to A,F,1,P
    std::vector<CF64> antennaCoefficients;
    const auto coefficientShape = ArrayShape({
        this->bfr5.cal_info.cal_all_dims[2],
        numberOfFrequencyChannels == 0 ? this->bfr5.cal_info.cal_all_dims[0] : numberOfFrequencyChannels,
        1,
        this->bfr5.cal_info.cal_all_dims[1],
    });
    if (frequencyChannelStartIndex + coefficientShape.numberOfFrequencyChannels() > this->bfr5.cal_info.cal_all_dims[0]) {
        BL_FATAL("Requested frequency-channel range [{}, {}) exceeds dimensions of BFR5 contents ({}).", frequencyChannelStartIndex, frequencyChannelStartIndex + coefficientShape.numberOfFrequencyChannels(), this->bfr5.cal_info.cal_all_dims[0]);
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }
    antennaCoefficients.resize(coefficientShape.size());

    const size_t calAntStride = 1;
    const size_t calPolStride = coefficientShape.numberOfAspects() * calAntStride;
    const size_t calChnStride = coefficientShape.numberOfPolarizations() * calPolStride;

    const size_t weightsPolStride = 1;
    const size_t weightsChnStride = coefficientShape.numberOfPolarizations() * weightsPolStride;
    const size_t weightsAntStride = coefficientShape.numberOfFrequencyChannels() * weightsChnStride;

    for (U64 antIdx = 0; antIdx < coefficientShape.numberOfAspects(); antIdx++) {
        for (U64 chnIdx = 0; chnIdx < coefficientShape.numberOfFrequencyChannels(); chnIdx++) {
            for (U64 polIdx = 0; polIdx < coefficientShape.numberOfPolarizations(); polIdx++) {
                const auto inputIdx = (frequencyChannelStartIndex + chnIdx) * calChnStride +
                                        polIdx * calPolStride + 
                                        antIdx * calAntStride;

                const auto outputIdx = antIdx * weightsAntStride +
                                        polIdx * weightsPolStride +
                                        chnIdx * weightsChnStride;

                const auto& coeff = this->bfr5.cal_info.cal_all[inputIdx];
                antennaCoefficients.data()[outputIdx] = {coeff.re, coeff.im};
            }
        }
    }

    return antennaCoefficients;
}

}  // namespace Blade::Modules::Bfr5
