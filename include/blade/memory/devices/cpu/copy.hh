#ifndef BLADE_MEMORY_CPU_COPY_HH
#define BLADE_MEMORY_CPU_COPY_HH

#include "blade/memory/types.hh"
#include "blade/memory/vector.hh"

namespace Blade::Memory {

template<typename T>
static const Result Copy(Vector<Device::CPU, T>& dst,
                         const Vector<Device::CPU, T>& src) {
    if (dst.size() != src.size()) {
        BL_FATAL("Size mismatch between source and destination ({}, {}).",
                src.size(), dst.size());
        return Result::ASSERTION_ERROR;
    }

    memcpy(dst.data(), src.data(), src.size_bytes());

    return Result::SUCCESS;
}

}  // namespace Blade::Memory

#endif
