# ==============================================================================
# Sapphire Rapids Toolchain (Xeon Max 9462)
# Numerically safe maximum throughput for HHG/TDSE
#
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=cmake.toolchain/saphirerapids.cmake ..
# ==============================================================================

find_program(INTEL_CXX mpiicpx)
find_program(INTEL_C   mpiicx)
find_program(INTEL_FC  mpiifx)

if(INTEL_C AND INTEL_CXX AND INTEL_FC)
    message(STATUS "Toolchain: Intel LLVM (icx/icpx/ifx) — Sapphire Rapids")

    set(CMAKE_C_COMPILER       mpiicx)
    set(CMAKE_CXX_COMPILER     mpiicpx)
    set(CMAKE_Fortran_COMPILER mpiifx)

    # --------------------------------------------------------------------------
    # Architecture
    # -xSAPPHIRERAPIDS  : full SPR ISA — AVX-512 F/BW/DQ/VL/VNNI/BF16/FP16/AMX
    # -qopt-zmm-usage=high : force 512-bit ZMM registers (icpx defaults to 256)
    #
    # NOTE: CMakeLists.txt must have ENABLE_NATIVE=OFF and MARCH_OVERRIDE=""
    # otherwise -march=native/-xHost will override -xSAPPHIRERAPIDS here.
    # Pass -DENABLE_NATIVE=OFF to cmake.
    # --------------------------------------------------------------------------
    set(ARCH_FLAGS "-xSAPPHIRERAPIDS -fp-model=fast -O3 -funroll-loops -fomit-frame-pointer -fiopenmp -qopt-zmm-usage=high")
    # --------------------------------------------------------------------------
    # Floating point — safe for HHG/TDSE
    # -fp-model=fast     : enables FMA fusion and reassociation. Safe for TDSE.
    #                      DO NOT use -fp-model=precise — it disables FMA entirely.
    # -fno-trapping-math : no FP exception traps
    # -fno-math-errno    : skip errno after math calls
    # --------------------------------------------------------------------------
    set(FP_FLAGS "-fp-model=fast=2 -fno-trapping-math -fno-math-errno")

    # --------------------------------------------------------------------------
    # Optimization — only flags verified supported by icpx
    # --------------------------------------------------------------------------
    set(OPT_FLAGS "-O3 -funroll-loops -fomit-frame-pointer")

    # --------------------------------------------------------------------------
    # OpenMP — -fiopenmp is the correct flag for Intel LLVM
    # --------------------------------------------------------------------------
    set(OMP_FLAGS "-fiopenmp")

    # --------------------------------------------------------------------------
    # Assemble — IPO/LTO intentionally excluded, handled by CMakeLists.txt
    # --------------------------------------------------------------------------
    set(COMMON_FLAGS "${ARCH_FLAGS} ${FP_FLAGS} ${OPT_FLAGS} ${OMP_FLAGS}")

    set(CMAKE_C_FLAGS       "${COMMON_FLAGS}" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS     "${COMMON_FLAGS}" CACHE STRING "" FORCE)
    set(CMAKE_Fortran_FLAGS "${COMMON_FLAGS}" CACHE STRING "" FORCE)

    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -fiopenmp -Wl,-O3"
        CACHE STRING "" FORCE
    )

    set(OpenMP_C_FLAGS       "-fiopenmp")
    set(OpenMP_CXX_FLAGS     "-fiopenmp")
    set(OpenMP_Fortran_FLAGS "-fiopenmp")

else()
    # --------------------------------------------------------------------------
    # GCC fallback
    # --------------------------------------------------------------------------
    message(STATUS "Toolchain: Intel not found — falling back to GCC native")

    set(CMAKE_C_COMPILER       mpicc)
    set(CMAKE_CXX_COMPILER     mpic++)
    set(CMAKE_Fortran_COMPILER mpif90)

    set(COMMON_FLAGS
        "-O3 -march=native -mtune=native -mprefer-vector-width=512 \
         -fno-math-errno -fno-trapping-math -fcx-limited-range \
         -ffp-contract=fast -funroll-loops -fomit-frame-pointer \
         -fvect-cost-model=unlimited -fsimd-cost-model=unlimited \
         -fno-plt -fopenmp"
    )

    set(CMAKE_C_FLAGS       "${COMMON_FLAGS}" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS     "${COMMON_FLAGS}" CACHE STRING "" FORCE)
    set(CMAKE_Fortran_FLAGS "${COMMON_FLAGS}" CACHE STRING "" FORCE)

    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -Wl,-O3 -Wl,--as-needed"
        CACHE STRING "" FORCE
    )

    set(OpenMP_C_FLAGS       "-fopenmp")
    set(OpenMP_CXX_FLAGS     "-fopenmp")
    set(OpenMP_Fortran_FLAGS "-fopenmp")

endif()
