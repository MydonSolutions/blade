#include "blade/modules/bfr5/reader.hh"

#include "bfr5.jit.hh"

namespace Blade::Modules::Bfr5 {

Reader::Reader(const Config& config, const Input& input) 
        : Module(config.blockSize, bfr5_kernel),
          config(config),
          input(input) {
    BL_INFO("===== BFR5 Reader Module Configuration");

    if (!std::filesystem::exists(config.filepath)) {
        BL_FATAL("Input file ({}) doesn't not exist.", config.filepath);
        BL_CHECK_THROW(Result::ASSERTION_ERROR);
    }

    BFR5open(config.filepath.c_str(), &this->bfr5);
    BFR5read_all(&this->bfr5);

    // Calculate beam coordinates

    for (U64 i = 0; i < this->bfr5.beam_info.ra_elements; i++) {
        beamCoordinates.push_back({
            this->bfr5.beam_info.ras[i],
            this->bfr5.beam_info.decs[i]
        });
    }

    // Calculate antenna positions

    antennaPositions.resize(getTotalNumberOfAntennas());

    const U64 antennaPositionsByteSize = getTotalNumberOfAntennas() * sizeof(XYZ);
    std::memcpy(antennaPositions.data(), this->bfr5.tel_info.antenna_positions, antennaPositionsByteSize);

    std::string antFrame = std::string(this->bfr5.tel_info.antenna_position_frame);

    if (antFrame != "xyz" && antFrame != "XYZ") {
        if (antFrame == "ecef" || antFrame == "ECEF") {
            calc_position_to_xyz_frame_from_ecef(
                reinterpret_cast<F64*>(antennaPositions.data()),
                antennaPositions.size(),
                this->bfr5.tel_info.latitude,
                this->bfr5.tel_info.longitude,
                this->bfr5.tel_info.altitude);
        }

        if (antFrame == "enu" || antFrame == "ENU") {
            calc_position_to_xyz_frame_from_enu(
                reinterpret_cast<F64*>(antennaPositions.data()),
                antennaPositions.size(),
                this->bfr5.tel_info.latitude,
                this->bfr5.tel_info.longitude,
                this->bfr5.tel_info.altitude);
        }
    }

    BL_INFO("Input File Path: {}", config.filepath);
    BL_INFO("Total Number of Beams: {}", getTotalNumberOfBeams());
    BL_INFO("Total Number of Antennas: {}", getTotalNumberOfAntennas());
    BL_INFO("Total Number of Frequency Channels: {}", getTotalNumberOfFrequencyChannels());
    BL_INFO("Total Number of Time Samples: {}", getTotalNumberOfTimeSamples());
    BL_INFO("Total Number of Polarizations: {}", getTotalNumberOfPolarizations());

    BFR5close(&this->bfr5);
}

const std::vector<CF64> Reader::getAntennaCalibrations(const U64& numberOfFrequencyChannels,
                                                       const U64& channelizerRate) {
    std::vector<CF64> antennaCalibrations;

    antennaCalibrations.resize(
            getTotalNumberOfAntennas() *
            numberOfFrequencyChannels * channelizerRate * 
            getTotalNumberOfPolarizations());

    const size_t calAntStride = 1;
    const size_t calPolStride = getTotalNumberOfAntennas() * calAntStride;
    const size_t calChnStride = getTotalNumberOfPolarizations() * calPolStride;

    const size_t weightsPolStride = 1;
    const size_t weightsChnStride = getTotalNumberOfPolarizations() * weightsPolStride;
    const size_t weightsAntStride = numberOfFrequencyChannels * weightsChnStride;

    for (U64 antIdx = 0; antIdx < getTotalNumberOfAntennas(); antIdx++) {
        for (U64 chnIdx = 0; chnIdx < numberOfFrequencyChannels; chnIdx++) {
            for (U64 polIdx = 0; polIdx < getTotalNumberOfPolarizations(); polIdx++) {
                for (U64 fchIdx = 0; fchIdx < channelizerRate; fchIdx++) {
                    const auto inputIdx = chnIdx * calChnStride +
                                          polIdx * calPolStride + 
                                          antIdx * calAntStride;

                    const auto frqIdx = chnIdx * channelizerRate + fchIdx;
                    const auto outputIdx = antIdx * weightsAntStride +
                                           polIdx * weightsPolStride +
                                           frqIdx * weightsChnStride;

                    const auto& coeff = this->bfr5.cal_info.cal_all[inputIdx];
                    antennaCalibrations[outputIdx] = {coeff.re, coeff.im};
                }
            }
        }
    }

    return antennaCalibrations;
}

}  // namespace Blade::Modules::Bfr5