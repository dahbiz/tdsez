# Intel Skylake (Server/AVX-512): numerically safe optimization
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_Fortran_COMPILER mpifort)

# -march=skylake-avx512 enables AVX-512F, CD, BW, DQ, and VL.
# We include -mprefer-vector-width=256 as a common HPC practice for Skylake 
# to avoid aggressive frequency downclocking unless the code is perfectly vectorized.
set(OPT_FLAGS "-O3 -march=skylake-avx512 -mtune=skylake-avx512 \
-mfma -mavx512f -mavx512vl -mprefer-vector-width=256 \
-fno-fast-math -fno-unsafe-math-optimizations \
-fexcess-precision=standard -ffp-contract=on \
-funroll-loops -flto -fomit-frame-pointer -fno-trapping-math")

set(CMAKE_C_FLAGS "${OPT_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${OPT_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_Fortran_FLAGS "${OPT_FLAGS}" CACHE STRING "" FORCE)

# OpenMP (optional)
set(OpenMP_C_FLAGS "-fopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_Fortran_FLAGS "-fopenmp")
