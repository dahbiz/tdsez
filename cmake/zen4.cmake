# Zen 4: numerically safe optimization
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)

set(OPT_FLAGS "-O3 -march=znver4 -mtune=znver4 \
-mfma -mavx512f -mavx512vl \
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

