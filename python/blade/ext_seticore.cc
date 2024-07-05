#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "blade/base.hh"
#include "blade/modules/base.hh"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Blade;

void NB_SUBMODULE(auto& m, const auto& name, const auto& typeName) {
    using Class = Modules::Seticore::Dedoppler;

    auto mm = m.def_submodule(name);

    nb::class_<Class, Module> mod(mm, typeName);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const BOOL&,
                      const F64&,
                      const F64&,
                      const F64&,

                      const F64&,
                      const F64&,
                      const F64&,
                      const U64&,
                      const BOOL&,
                      const BOOL&,

                      const std::vector<double>&,
                      const std::vector<double>&,

                      const std::string&,
                      const U64&,
                      const std::string&,
                      const std::string&,
                      const RA_DEC&,
                      const std::vector<std::string>&,
                      const std::vector<RA_DEC>&,
                      const U64&,
                      const U64&,

                      const BOOL& = false

                      const U64& = 512>(), "mitigate_dc_spike"_a,
                                           "minimum_drift_rate"_a = 0.0,
                                           "maximum_drift_rate"_a,
                                           "snr_threshold"_a,

                                           "frequency_of_first_channel_hz"_a,
                                           "channel_bandwidth_hz"_a,
                                           "channel_timespan_s"_a,
                                           "coarse_channel_rate"_a,
                                           "last_beam_is_incoherent"_a = false,
                                           "search_incoherent_beam"_a = true,

                                           "search_exclusion_subband_bottoms_mhz"_a,
                                           "search_exclusion_subband_tops_mhz"_a,

                                           "filepath_prefix"_a,
                                           "telescope_id"_a,
                                           "source_name"_a,
                                           "observation_identifier"_a,
                                           "phase_center"_a,
                                           "aspect_names"_a,
                                           "aspect_coordinates"_a,
                                           "total_number_of_time_samples"_a,
                                           "total_number_of_frequency_channels"_a,

                                           "produce_debug_hits"_a = false,

                                           "block_size"_a = 512>);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<Device::CPU, F32>&>(), "buf"_a)
        .def(nb::init<const Vector<Device::CPU, U64>&>(), "coarse_frequency_channel_offset"_a)
        .def(nb::init<const Vector<Device::CPU, F64>&>(), "julian_date_start"_a);

    mod
        .def(nb::init<const typename Class::Config&,
                      const typename Class::Input&,
                      const Stream&>(), "config"_a,
                                        "input"_a,
                                        "stream"_a)
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_input_coarse_frequency_channel_offset", &Class::getInputCoarseFrequencyChannelOffset, nb::rv_policy::reference)
        .def("get_julian_date_of_input", &Class::getJulianDateOfInput, nb::rv_policy::reference)
        .def("get_output_hits", &Class::getOutputHits, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return bl::fmt::format("SeticoreDedoppler()");
        });
}

template<typename IT>
void NB_SUBMODULE_WRITER(auto& m, const auto& name, const auto& typeName) {
    using Class = Modules::Seticore::HitsStampWriter<IT>;

    auto mm = m.def_submodule(name);

    nb::class_<Class> mod(mm, typeName);

    nb::class_<typename Class::Config>(mod, "config")
        .def(nb::init<const std::string&,

                      const U64&,
                      const std::string&,
                      const std::string&,
                      const RA_DEC&,
                      const U64&,
                      const U64&,
                      const F64&,
                      const F64&,
                      const F64&,
                      const I64&,

                      const U64&>(), "filepath_prefix"_a,
                                     "telescope_id",
                                     "source_name",
                                     "observation_identifier",
                                     "phase_center",
                                     "coarse_start_channel_index",
                                     "coarse_channel_ratio",
                                     "channel_bandwidth_hz",
                                     "channel_timespan_s",
                                     "stamp_frequency_margin_hz" = 500.0,
                                     "hits_grouping_margin" = 30,
                                     "block_size"_a = 512);

    nb::class_<typename Class::Input>(mod, "input")
        .def(nb::init<const ArrayTensor<Device::CPU, IT>&>(), "buffer"_a)
        .def(nb::init<const std::vector<DedopplerHits>&>(), "hits"_a)
        .def(nb::init<const Vector<Device::CPU, U64>&>(), "frequency_of_first_channel_hz"_a)
        .def(nb::init<const Vector<Device::CPU, F64>&>(), "julian_date_start"_a);

    mod
        .def(nb::init<const typename Class::Config&, const typename Class::Input&>())
        .def("process", [](Class& instance, const U64& counter) {
            return instance.process(counter);
        })
        .def("get_config", &Class::getConfig, nb::rv_policy::reference)
        .def("get_input_buffer", &Class::getInputBuffer, nb::rv_policy::reference)
        .def("get_input_frequency_of_first_channel_hz", &Class::getInputFrequencyOfFirstChannelHz, nb::rv_policy::reference)
        .def("get_input_julian_date_start", &Class::getInputJulianDateStart, nb::rv_policy::reference)
        .def("__repr__", [](Class& obj){
            return bl::fmt::format("SeticoreStampsWriter()");
        });
}

NB_MODULE(_seticore_impl, m) {
    NB_SUBMODULE<CF32, F32>(m, "in_cf32", "out_f32");
    NB_SUBMODULE_WRITER<CF16>(m, "taint_writer", "type_cf16");
    NB_SUBMODULE_WRITER<CF32>(m, "taint_writer", "type_cf32");
}
