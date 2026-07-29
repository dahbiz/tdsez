#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <slepceps.h>
#include <petiga.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <iostream>
#include <stdexcept>

PetscErrorCode MatCompare(const Mat M1, const Mat M2);
PetscErrorCode MatPrint(const Mat M, PetscInt rows, PetscInt cols);
PetscErrorCode SparsityPatternCheck(const Mat M1, const Mat M2);
PetscErrorCode TDSEXExpectationValue(Mat O, Vec psi, PetscScalar *expectation);
PetscErrorCode TDSEXHermCheck(Mat O, PetscReal tol);




#endif