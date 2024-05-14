#ifndef BLADE_MODULES_BFR5_READER_HH
#define BLADE_MODULES_BFR5_READER_HH

#include <filesystem>
#include <string>

#include "blade/base.hh"
#include "blade/module.hh"

extern "C" {
#include "bfr5.h"
#include "radiointerferometryc99.h"
}

namespace Blade::Modules::Bfr5 {

class BLADE_API Reader : public Module {
 public:
    // Configuration

    struct Config {
        std::string filepath;

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
    };

    // Taint Registers

    constexpr Taint getTaint() const {
        return Taint::NONE; 
    }

    std::string name() const {
        return "BFR5 Reader";
    }

    // Constructor & Processing

    explicit Reader(const Config& config, const Input& input,
                    const Stream& stream = {});
    
    // TODO: destructor
    // ~Reader() { BFR5free_all(this->bfr5); };

    // Miscellaneous

    PhasorShape getShape() const {
        return PhasorShape({
            this->bfr5.dim_info.nbeams,
            this->bfr5.dim_info.nants,
            this->bfr5.dim_info.nchan,
            this->bfr5.dim_info.ntimes,
            this->bfr5.dim_info.npol,
        });
    }

    LLA getReferencePosition() const {
        return {
            .LON = calc_rad_from_degree(this->bfr5.tel_info.longitude),
            .LAT = calc_rad_from_degree(this->bfr5.tel_info.latitude),
            .ALT = this->bfr5.tel_info.altitude,
        };
    }

    constexpr RA_DEC getBoresightCoordinates() const {
        return {
            .RA = this->bfr5.obs_info.phase_center_ra,
            .DEC = this->bfr5.obs_info.phase_center_dec
        };
    }

    constexpr const std::vector<XYZ>& getAntennaPositions() const {
        return this->antennaPositions;
    }

    constexpr const std::vector<RA_DEC>& getBeamCoordinates() const {
        return this->beamCoordinates;
    }

    std::vector<CF64> getAntennaCoefficients(const U64& numberOfFrequencyChannels = 0, const U64& frequencyChannelStartIndex = 0);

 private:
    // Variables

    Config config;
    const Input input;
    Output output;

    BFR5_file_t bfr5;

    // TODO: Update from vector to ArrayTensor. 
    std::vector<XYZ> antennaPositions;
    std::vector<RA_DEC> beamCoordinates;
    std::vector<std::string> beamSourceNames;
};

}  // namespace Blade::Modules

#endif

