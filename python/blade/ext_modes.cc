#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"
#include "blade/bundles/base.hh"
#include "blade/memory/custom.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

void NB_SUBMODULE_WRITER(auto& m, const auto& in_name, const auto& out_name) {
    using Class = Bundles::VLA::ModeS;

    auto mm = m.def_submodule(in_name)
               .def_submodule(out_name);

    nb::class_<Class, Bundle> mod(mm, "mod");

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const U64&,
                      const std::string&,
                      const std::string&,
                      const RA_DEC&,
                      const U64&,
                      const U64&,
                      const F64&,
                      const U64&,
                      const U64&,
                      const BOOL&,

                      const std::vector<std::string>&,
                      const std::vector<RA_DEC>&,

                      const BOOL&,
                      const BOOL&,
                      const F64&,
                      const F64&,
                      const F64&,
                      const F64&,
                      const I64&,
                      const BOOL&,

                      const F64&,
                      const F64&,
                      const std::string&,
                      const std::vector<double>&,
                      const std::vector<double>&,

                      const BOOL&,
                      const U64&>(),
                                     "input_telescope_id"_a,
                                     "input_source_name"_a,
                                     "input_observation_identifier"_a,
                                     "input_phase_center"_a,
                                     "input_total_number_of_time_samples"_a,
                                     "input_total_number_of_frequency_channels"_a,
                                     "input_frequency_of_first_channel_hz"_a,
                                     "input_coarse_start_channel_index"_a,
                                     "input_coarse_channel_ratio"_a = 1,
                                     "input_last_beam_is_incoherent"_a = false,

                                     "beam_names"_a,
                                     "beam_coordinates"_a,

                                     "search_mitigate_dc_spike"_a,
                                     "search_drift_rate_zero_excluded"_a = false,
                                     "search_minimum_drift_rate"_a = 0.0,
                                     "search_maximum_drift_rate"_a,
                                     "search_snr_threshold"_a,
                                     "search_stamp_frequency_margin_hz"_a,
                                     "search_hits_grouping_margin"_a,
                                     "search_incoherent_beam"_a = true,

                                     "search_channel_bandwidth_hz"_a,
                                     "search_channel_timespan_s"_a,
                                     "search_output_filepath_stem"_a,
                                     "search_exclusion_subband_bottoms_mhz"_a,
                                     "search_exclusion_subband_tops_mHz"_a,

                                     "produce_debug_hits"_a = false,
                                     "dedoppler_block_size"_a = 512);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<Device::CUDA, F32>&,
                      const ArrayTensor<Device::CUDA, CF32>&,
                      const Tensor<Device::CPU, U64>&,
                      const Tensor<Device::CPU, F64>&,
                      const Tensor<Device::CPU, F64>&>(), "beamformed_data"_a,
                                                          "prebeamformer_data"_a,
                                                          "coarse_frequency_channel_offset"_a,
                                                          "frequency_of_first_channel_hz"_a,
                                                          "julian_date_start"_a);

    mod
        .def(nb::init<const typename Class::Config&,
                      const typename Class::Input&,
                      const Stream&>(), "config"_a,
                                        "input"_a,
                                        "stream"_a)
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_input_data", &Class::getInputData, nb::rv_policy::reference)
        .def("get_input_frequency_of_first_channel_hz", &Class::getInputFrequencyOfFirstChannelHz, nb::rv_policy::reference)
        .def("get_input_julian_date", &Class::getInputJulianDate, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return bl::fmt::format("ModeS(telescope=bl.vla)");
        });
}

NB_MODULE(_modes_impl, m) {
    NB_SUBMODULE_WRITER(m, "taint_writer", "type_f32");
}
