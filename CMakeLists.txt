# ============================================================================
#  tdsez/CMakeLists.txt
# ----------------------------------------------------------------------------
#  Self-contained build for the modular TDSEZ code.
#
#  This folder is an independent snapshot of src/ + include/ + tests/ with the
#  legacy PULSE wrapper class removed (its only job -- calling
#  TDSEZParser::PrmReader + initParsers -- is now done inline in
#  TDSEZCore::TDSEZCore). It builds on its own; it does NOT depend on the
#  parent ../CMakeLists.txt.
#
#  DESIGN: the build AUTO-DETECTS how PETSc was compiled and inherits the
#  same optimisation flags, -march/-mtune, OpenMP, and CUDA arch. This keeps
#  TDSEZ ABI-compatible with PETSc and at peak performance without the user
#  having to repeat PETSc's configure flags. The detection reads:
#    $PETSC_DIR/lib/pkgconfig/PETSc.pc          (cxxflags_extra, cudaflags_extra)
#    $PETSC_DIR/include/petscconf.h             (feature macros: CUDA, HDF5, ...)
#    $PETSC_DIR/lib/petsc/conf/petscvariables   (fallback: CXX_FLAGS, CUDAC_FLAGS)
#
#  Required environment:
#    PETSC_DIR   - PETSc install prefix (export it)
#    PETSC_ARCH  - may be empty; not required at configure time
#
#  Build:
#    cd fused-version && mkdir build && cd build
#    cmake .. -DPETSC_DIR=$PETSC_DIR
#    make -j
#    ctest            # runs the fused validation tests
# ============================================================================

cmake_minimum_required(VERSION 3.15)

project(TDSEZFused
    VERSION 1.0.0
    LANGUAGES C CXX
)

# ----------------------------------------------------------------------------
# C++ standard — must match PETSc (C++17) for ABI / symbol compatibility.
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ----------------------------------------------------------------------------
# Options (all overridable with -D)
# ----------------------------------------------------------------------------
option(ENABLE_OPENMP    "Enable OpenMP (matches PETSc)"            ON)
option(ENABLE_LTO       "Enable link-time optimisation"            ON)
option(ENABLE_NATIVE    "Use -march/-mtune detected from PETSc"     ON)
option(ENABLE_AUTO_CUDA "Auto-enable CUDA when PETSc was built with it" ON)
option(PETSC_INHERIT_FLAGS "Inherit PETSc optimisation flags"      ON)

# ----------------------------------------------------------------------------
# Locate PETSc and validate
# ----------------------------------------------------------------------------
if(NOT DEFINED ENV{PETSC_DIR})
    message(FATAL_ERROR
        "PETSC_DIR not set. Export it before configuring:\n"
        "  export PETSC_DIR=/path/to/petsc\n"
        "  (PETSC_ARCH is optional and may be empty)")
endif()
set(PETSC_DIR  $ENV{PETSC_DIR})
# PETSC_ARCH may be unset/empty — that is a valid configuration for many
# installs (PETSc itself sets PETSC_ARCH="" in petscconf.h here). Do NOT fail.
if(DEFINED ENV{PETSC_ARCH})
    set(PETSC_ARCH $ENV{PETSC_ARCH})
else()
    set(PETSC_ARCH "")
endif()

# ----------------------------------------------------------------------------
# Detect PETSc build configuration
# ----------------------------------------------------------------------------
set(_petsc_pc        "${PETSC_DIR}/lib/pkgconfig/PETSc.pc")
set(_petsc_conf      "${PETSC_DIR}/include/petscconf.h")
set(_petsc_variables "${PETSC_DIR}/lib/petsc/conf/petscvariables")

# --- 1) pkg-config file (authoritative for cxxflags_extra, cudaflags_extra) ---
if(EXISTS "${_petsc_pc}")
    message(STATUS "Reading PETSc build flags from ${_petsc_pc}")
    file(STRINGS "${_petsc_pc}" _pc_lines)

    # cxxflags_extra (C++ compile flags used by PETSc, e.g. -O3 -march=...)
    foreach(_line ${_pc_lines})
        if(_line MATCHES "^cxxflags_extra=(.*)")
            set(PETSC_CXXFLAGS_EXTRA "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^cflags_extra=(.*)")
            set(PETSC_CFLAGS_EXTRA "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^cudaflags_extra=(.*)")
            set(PETSC_CUDAFLAGS_EXTRA "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^ldflags=(.*)")
            set(PETSC_LDFLAGS_EXTRA "${CMAKE_MATCH_1}")
        endif()
    endforeach()

    # Strip leading/trailing whitespace
    string(STRIP "${PETSC_CXXFLAGS_EXTRA}"   PETSC_CXXFLAGS_EXTRA)
    string(STRIP "${PETSC_CFLAGS_EXTRA}"     PETSC_CFLAGS_EXTRA)
    string(STRIP "${PETSC_CUDAFLAGS_EXTRA}"  PETSC_CUDAFLAGS_EXTRA)
    string(STRIP "${PETSC_LDFLAGS_EXTRA}"    PETSC_LDFLAGS_EXTRA)
else()
    message(STATUS "PETSc .pc file not found; falling back to petscvariables")
    set(PETSC_CXXFLAGS_EXTRA "")
    set(PETSC_CUDAFLAGS_EXTRA "")
    set(PETSC_LDFLAGS_EXTRA "")
endif()

# --- 2) petscvariables (fallback for CXX_FLAGS if .pc didn't yield flags) ---
if(PETSC_CXXFLAGS_EXTRA STREQUAL "" AND EXISTS "${_petsc_variables}")
    file(STRINGS "${_petsc_variables}" _pv_lines)
    foreach(_line ${_pv_lines})
        if(_line MATCHES "^CXX_FLAGS\\s*=\\s*(.*)")
            set(PETSC_CXXFLAGS_EXTRA "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^CUDAC_FLAGS\\s*=\\s*(.*)")
            set(PETSC_CUDAFLAGS_EXTRA "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    string(STRIP "${PETSC_CXXFLAGS_EXTRA}"   PETSC_CXXFLAGS_EXTRA)
    string(STRIP "${PETSC_CUDAFLAGS_EXTRA}"  PETSC_CUDAFLAGS_EXTRA)
endif()

# --- 3) petscconf.h feature macros (CUDA, HDF5, OpenMP, 64-bit indices) ---
if(EXISTS "${_petsc_conf}")
    file(STRINGS "${_petsc_conf}" _conf_lines)
    foreach(_line ${_conf_lines})
        if(_line MATCHES "^#define PETSC_HAVE_CUDA 1")
            set(PETSC_HAVE_CUDA ON)
        elseif(_line MATCHES "^#define PETSC_HAVE_OPENMP 1")
            set(PETSC_HAVE_OPENMP ON)
        elseif(_line MATCHES "^#define PETSC_HAVE_HDF5 1")
            set(PETSC_HAVE_HDF5 ON)
        elseif(_line MATCHES "^#define PETSC_USE_64BIT_INDICES 1")
            set(PETSC_USE_64BIT_INDICES ON)
        elseif(_line MATCHES "^#define PETSC_USE_REAL_DOUBLE 1")
            set(PETSC_USE_REAL_DOUBLE ON)
        elseif(_line MATCHES "^#define PETSC_USE_REAL_SINGLE 1")
            set(PETSC_USE_REAL_SINGLE ON)
        endif()
    endforeach()
endif()

# ----------------------------------------------------------------------------
# Parse PETSc flags into lists for selective inheritance
# ----------------------------------------------------------------------------
# separate() preserves -march=alderlake as one token, separates -O3 from
# -march=..., etc.
separate_arguments(PETSC_CXXFLAGS_LIST UNIX_COMMAND "${PETSC_CXXFLAGS_EXTRA}")

# Extract -march, -mtune from PETSc (so we match the exact CPU target)
set(PETSC_MARCH "")
set(PETSC_MTUNE "")
set(PETSC_HAS_O3 OFF)
foreach(_flag ${PETSC_CXXFLAGS_LIST})
    if(_flag MATCHES "^-march=(.*)")
        set(PETSC_MARCH "${CMAKE_MATCH_1}")
    elseif(_flag MATCHES "^-mtune=(.*)")
        set(PETSC_MTUNE "${CMAKE_MATCH_1}")
    elseif(_flag STREQUAL "-O3")
        set(PETSC_HAS_O3 ON)
    endif()
endforeach()

# Extract CUDA arch from PETSc cudaflags (e.g. -arch=sm_89)
set(PETSC_CUDA_ARCH "")
if(PETSC_CUDAFLAGS_EXTRA)
    separate_arguments(PETSC_CUDAFLAGS_LIST UNIX_COMMAND "${PETSC_CUDAFLAGS_EXTRA}")
    foreach(_flag ${PETSC_CUDAFLAGS_LIST})
        if(_flag MATCHES "^-arch=sm_(.*)")
            set(PETSC_CUDA_ARCH "${CMAKE_MATCH_1}")
        endif()
    endforeach()
endif()

# --- 4) Detect PETSc version from petscversion.h ---
set(PETSC_VERSION "")
set(_petsc_version_h "${PETSC_DIR}/include/petscversion.h")
if(EXISTS "${_petsc_version_h}")
    file(STRINGS "${_petsc_version_h}" _pv_lines)
    foreach(_line ${_pv_lines})
        if(_line MATCHES "^#define PETSC_VERSION_MAJOR[ \t]+([0-9]+)")
            set(PETSC_VERSION_MAJOR "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^#define PETSC_VERSION_MINOR[ \t]+([0-9]+)")
            set(PETSC_VERSION_MINOR "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^#define PETSC_VERSION_SUBMINOR[ \t]+([0-9]+)")
            set(PETSC_VERSION_SUBMINOR "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    if(PETSC_VERSION_MAJOR AND PETSC_VERSION_MINOR)
        set(PETSC_VERSION "${PETSC_VERSION_MAJOR}.${PETSC_VERSION_MINOR}.${PETSC_VERSION_SUBMINOR}")
    endif()
endif()

# ----------------------------------------------------------------------------
# Detect SLEPc
# ----------------------------------------------------------------------------
# SLEPc installs side-by-side with PETSc under $PETSC_DIR. It has a .pc file
# and a slepcversion.h header with version macros.
set(SLEPC_DIR ${PETSC_DIR})  # SLEPc co-installed with PETSc
set(_slepc_pc        "${SLEPC_DIR}/lib/pkgconfig/SLEPc.pc")
set(_slepc_version_h "${SLEPC_DIR}/include/slepcversion.h")
set(_slepc_lib       "${SLEPC_DIR}/lib/libslepc${CMAKE_SHARED_LIBRARY_SUFFIX}")

set(SLEPC_FOUND OFF)
set(SLEPC_VERSION "")

# Try .pc file first (has Version: field)
if(EXISTS "${_slepc_pc}")
    file(STRINGS "${_slepc_pc}" _slepc_pc_lines)
    foreach(_line ${_slepc_pc_lines})
        if(_line MATCHES "^Version:[ \t]*(.*)")
            string(STRIP "${CMAKE_MATCH_1}" SLEPC_VERSION)
        endif()
    endforeach()
endif()

# Fallback: read slepcversion.h macros
if(SLEPC_VERSION STREQUAL "" AND EXISTS "${_slepc_version_h}")
    file(STRINGS "${_slepc_version_h}" _slepc_v_lines)
    foreach(_line ${_slepc_v_lines})
        if(_line MATCHES "^#define SLEPC_VERSION_MAJOR[ \t]+([0-9]+)")
            set(SLEPC_VERSION_MAJOR "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^#define SLEPC_VERSION_MINOR[ \t]+([0-9]+)")
            set(SLEPC_VERSION_MINOR "${CMAKE_MATCH_1}")
        elseif(_line MATCHES "^#define SLEPC_VERSION_SUBMINOR[ \t]+([0-9]+)")
            set(SLEPC_VERSION_SUBMINOR "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    if(SLEPC_VERSION_MAJOR AND SLEPC_VERSION_MINOR)
        set(SLEPC_VERSION "${SLEPC_VERSION_MAJOR}.${SLEPC_VERSION_MINOR}.${SLEPC_VERSION_SUBMINOR}")
    endif()
endif()

# Confirm library exists
if(EXISTS "${_slepc_lib}")
    set(SLEPC_FOUND ON)
endif()

if(NOT SLEPC_FOUND)
    message(FATAL_ERROR
        "SLEPc not found under ${SLEPC_DIR}/lib.\n"
        "Expected: ${_slepc_lib}\n"
        "SLEPc is required for TDSEZ eigenvalue problems (EPS/PEP).")
endif()

# ----------------------------------------------------------------------------
# Detect PETIGA
# ----------------------------------------------------------------------------
# PetIGA has no .pc file or version header. Detect via the header + library.
# It is co-installed with PETSc under $PETSC_DIR/include/petiga.h and
# $PETSC_DIR/lib/libpetiga.so.
set(PETIGA_DIR ${PETSC_DIR})  # PetIGA co-installed with PETSc
set(_petiga_h   "${PETIGA_DIR}/include/petiga.h")
set(_petiga_lib "${PETIGA_DIR}/lib/libpetiga${CMAKE_SHARED_LIBRARY_SUFFIX}")

set(PETIGA_FOUND OFF)
set(PETIGA_VERSION "")

if(EXISTS "${_petiga_h}" AND EXISTS "${_petiga_lib}")
    set(PETIGA_FOUND ON)
    # PetIGA doesn't expose a version macro. Try to extract the git tag
    # from strings in the shared library (best effort, non-fatal).
    execute_process(
        COMMAND strings "${_petiga_lib}"
        OUTPUT_VARIABLE _petiga_strings
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(REGEX MATCH "PetIGA[^\n]*([0-9]+\\.[0-9]+(\\.[0-9]+)?)"
           _petiga_ver_match "${_petiga_strings}")
    if(_petiga_ver_match)
        set(PETIGA_VERSION "${CMAKE_MATCH_1}")
    else()
        set(PETIGA_VERSION "unknown (no version macro in petiga.h)")
    endif()
endif()

if(NOT PETIGA_FOUND)
    message(FATAL_ERROR
        "PetIGA not found.\n"
        "Expected header: ${_petiga_h}\n"
        "Expected library: ${_petiga_lib}\n"
        "PetIGA is required for TDSEZ B-spline/IGA assembly.")
endif()

# ----------------------------------------------------------------------------
# Report what we detected
# ----------------------------------------------------------------------------
message(STATUS "================ PETSC DETECTION ================")
message(STATUS "PETSc dir   : ${PETSC_DIR}")
message(STATUS "PETSc arch  : ${PETSC_ARCH}")
if(PETSC_VERSION)
    message(STATUS "PETSc version: ${PETSC_VERSION}")
endif()
if(PETSC_MARCH)
    message(STATUS "PETSc -march: ${PETSC_MARCH}")
endif()
if(PETSC_MTUNE)
    message(STATUS "PETSc -mtune: ${PETSC_MTUNE}")
endif()
message(STATUS "PETSc O3    : ${PETSC_HAS_O3}")
message(STATUS "PETSc OpenMP: ${PETSC_HAVE_OPENMP}")
message(STATUS "PETSc CUDA  : ${PETSC_HAVE_CUDA} (arch: sm_${PETSC_CUDA_ARCH})")
message(STATUS "PETSc HDF5  : ${PETSC_HAVE_HDF5}")
message(STATUS "==================================================")
message(STATUS "SLEPc version : ${SLEPC_VERSION}")
message(STATUS "SLEPc lib    : ${_slepc_lib}")
message(STATUS "==================================================")
message(STATUS "PetIGA found : ${PETIGA_FOUND}")
message(STATUS "PetIGA version: ${PETIGA_VERSION}")
message(STATUS "PetIGA lib   : ${_petiga_lib}")
message(STATUS "==================================================")

# ----------------------------------------------------------------------------
# Apply optimisation flags: inherit PETSc's flags + add further optimisations
# ----------------------------------------------------------------------------
# Strategy: PETSc's flags are the BASELINE (ABI + march match). We then add
# TDSEZ-specific flags on top for further performance. This section applies
# to ALL targets via add_compile_options (global), so post-processed targets
# (TDMProcessor, etc.) inherit the same flags.
add_compile_options(-Wall -Wextra)

# Inherit PETSc's optimisation baseline (Release builds)
if(PETSC_INHERIT_FLAGS)
    # Apply PETSc's C++ flags (includes -O3, -march, -mtune, -ftree-vectorize, ...)
    # only in Release builds to avoid polluting Debug/RelWithDebInfo.
    foreach(_flag ${PETSC_CXXFLAGS_LIST})
        # Skip flags that are PETSc-internal or not relevant to us
        # (-fvisibility=hidden is re-applied below; -std=c++17 is already set)
        if(NOT _flag MATCHES "^-std=" AND NOT _flag MATCHES "^-fvisibility")
            add_compile_options($<$<CONFIG:Release>:${_flag}>)
        endif()
    endforeach()

    # fvisibility=hidden: match PETSc (avoids symbol-export surprises)
    add_compile_options($<$<CONFIG:Release>:-fvisibility=hidden>)

    # If PETSc has O3, also add aggressive loop optimisations PETSc doesn't set
    if(PETSC_HAS_O3)
        add_compile_options($<$<CONFIG:Release>:-funroll-loops>)
        # -ffast-math: violates IEEE, but standard for physics solvers (matches
        # PETSc's own openblas which is built with it). Safe because TDSEZ does
        # no signed-zero / NaN-dependent branching.
        add_compile_options($<$<CONFIG:Release>:-ffast-math>)
        # -fno-trapping-math: PETSc/OpenBLAS already assumes no FP traps
        add_compile_options($<$<CONFIG:Release>:-fno-trapping-math>)
        # Limit stdlib allocations in hot loops: inline known-size constructors
        add_compile_options($<$<CONFIG:Release>:-fno-enforce-eh-specs>)
    endif()
endif()

# If ENABLE_NATIVE is ON and PETSc didn't specify -march, fall back to -march=native
if(ENABLE_NATIVE AND NOT PETSC_MARCH)
    add_compile_options($<$<CONFIG:Release>:-march=native>)
endif()

# ----------------------------------------------------------------------------
# CUDA support: auto-detect from PETSc
# ----------------------------------------------------------------------------
set(TDSEZ_USE_CUDA OFF)
if(ENABLE_AUTO_CUDA AND PETSC_HAVE_CUDA AND PETSC_CUDA_ARCH)
    set(TDSEZ_USE_CUDA ON)
    message(STATUS "CUDA auto-detected from PETSc: sm_${PETSC_CUDA_ARCH}")
    # Define the preprocessor macro so source can #ifdef-guard GPU paths
    add_compile_definitions(TDSEZ_USE_CUDA=1)
    # Pass the arch as a define (for any runtime GPU arch checks)
    add_compile_definitions(TDSEZ_CUDA_ARCH=${PETSC_CUDA_ARCH})
endif()

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
find_package(MPI REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# PETSc / SLEPc / PETIGA / HDF5 live under $PETSC_DIR/lib.
include_directories(${PETSC_DIR}/include)
link_directories(${PETSC_DIR}/lib)

# muparser: prefer the CMake config (explicit, self-contained).
find_package(muparser CONFIG QUIET)
if(muparser_FOUND)
    message(STATUS "Found muparser via CMake config")
    set(MUPARSER_LINK muparser::muparser)
else()
    message(STATUS "muparser CMake config not found; relying on link_directories")
    set(MUPARSER_LINK muparser)
endif()

# OpenMP: enable if PETSc was built with it (or if the option is on)
if(ENABLE_OPENMP)
    find_package(OpenMP)
endif()

# ----------------------------------------------------------------------------
# Helper: configure a TDSEZ target with the correct deps + flags
# ----------------------------------------------------------------------------
# Deduplicates the per-target boilerplate. Every target gets:
#  - PETSc/SLEPc/PETIGA includes + link dirs (inherited globally above)
#  - MPI::MPI_CXX
#  - OpenMP if available
#  - LTO if supported
#  - OPENBLAS_NUM_THREADS=1 (thread-safety: OpenBLAS must not spawn its own
#    threads when OpenMP/MPI are already using the cores)
#  - output to ${CMAKE_BINARY_DIR}
#
# Usage: tdsez_add_target(<target> <src1> <src2> ...)
#        tdsez_add_target(<target> <src1> <src2> ... LINK_LIBS slepc petiga hdf5)
function(tdsez_add_target target)
    cmake_parse_arguments(ARG
        ""                          # options
        ""                          # one-value args
        "LINK_LIBS"                 # multi-value args
        ${ARGN}
    )

    add_executable(${target} ${ARG_UNPARSED_ARGUMENTS})

    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PETSC_DIR}/include
    )

    # Always link MPI + petsc; add extra libs from LINK_LIBS
    target_link_libraries(${target} PRIVATE
        MPI::MPI_CXX
        petsc
        ${ARG_LINK_LIBS}
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # OpenBLAS thread-safety: keep it single-threaded when OpenMP/MPI is active
    target_compile_definitions(${target} PRIVATE OPENBLAS_NUM_THREADS=1)

    # LTO: enable per-target (check_ipo_supported is global)
    if(ENABLE_LTO)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endfunction()

# ----------------------------------------------------------------------------
# Global LTO check (once, then applied per-target via the helper)
# ----------------------------------------------------------------------------
if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_OK)
    if(NOT LTO_OK)
        message(STATUS "LTO not supported by compiler -- disabling")
        set(ENABLE_LTO OFF)
    endif()
endif()

# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------

# --- tdsez: main TDSE solver (full feature set) ---------------------------
set(TDSEZ_SOURCES
    src/parser.cpp
    src/assembly/assembly.cpp
    src/assembler.cpp
    src/assembler_class.cpp
    src/core_knots.cpp
    src/core.cpp
    src/manager.cpp
    src/monitor.cpp
    src/propagator.cpp
    src/diagnostics.cpp
    src/tdsez.cpp
    src/debug.cpp
)

tdsez_add_target(tdsez ${TDSEZ_SOURCES}
    LINK_LIBS slepc petiga ${MUPARSER_LINK} hdf5 hdf5_hl
)

# --- tdmprocessor: full transition-dipole matrix post-processor ------
# Loads the saved dipole operator Dx (PETSc binary) and the eigenstates from
# the static HDF5, computes d_ij = <psi_i|Dx|psi_j> block-by-block.
# Standalone; only needs PETSc.
tdsez_add_target(tdmprocessor
    src/tdmprocessor.cpp
)

# --- tdmselect: selective d_ij for a single bra state -------
# CLI: -i <bra> -Ethr <E> [-w0 <omega0>] [-gpu]. Writes CSV.
# Standalone; PETSc + HDF5.
tdsez_add_target(tdmselect
    src/tdmselect.cpp
    LINK_LIBS hdf5 hdf5_hl
)

# --- drecprocessor: exact recombination dipole d_rec(E) --------------------
# Projects onto the outgoing channel using d|psi>/dE (finite-difference over
# neighbouring box eigenstates). Standalone; PETSc + HDF5.
tdsez_add_target(drecprocessor
    src/drecprocessor.cpp
    LINK_LIBS hdf5 hdf5_hl
)

# ----------------------------------------------------------------------------
# Banner -- matches the runtime TDSEZInfo::banner() format exactly
# ----------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/banner.cmake)
message(STATUS "")           # blank separator before banner
tdsez_banner()

# ----------------------------------------------------------------------------
# Build configuration summary
# ----------------------------------------------------------------------------
message(STATUS "  -- FUSED BUILD CONFIG --")
message(STATUS "  Build type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard       : ${CMAKE_CXX_STANDARD}")
message(STATUS "  LTO                : ${ENABLE_LTO}")
message(STATUS "  OpenMP             : ${OpenMP_CXX_FOUND}")
if(TDSEZ_USE_CUDA)
    message(STATUS "  CUDA               : sm_${PETSC_CUDA_ARCH}")
else()
    message(STATUS "  CUDA               : disabled")
endif()
message(STATUS "  Inherit PETSc flags: ${PETSC_INHERIT_FLAGS}")
message(STATUS "  PETSc -march       : ${PETSC_MARCH}")
message(STATUS "  PETSc -mtune       : ${PETSC_MTUNE}")
message(STATUS "  ============================================================")
