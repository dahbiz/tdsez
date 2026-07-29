#include "debug.hpp"


PetscErrorCode MatCompare(const Mat M1, const Mat M2)
{
    PetscErrorCode ierr;
    PetscFunctionBeginUser;            

    // compare : petsc has a problem with MatEqual for some types of matrices and the way they are stored
    PetscBool identical; 
    ierr = MatEqual(M1, M2, &identical);CHKERRQ(ierr);
    if (identical) 
    {
        PetscPrintf(PETSC_COMM_WORLD, "The matrices are identical.\n");
    } 
    else 
    {
        PetscPrintf(PETSC_COMM_WORLD, "The matrices are not identical. Something is wrong!\n");
    }

    // Compare mat sizes 
    PetscInt rows_M1, cols_M1, rows_M2, cols_M2;
    ierr = MatGetSize(M1, &rows_M1, &cols_M1);
    ierr = MatGetSize(M2, &rows_M2, &cols_M2);
    if (rows_M1 == rows_M2 && cols_M1 == cols_M2) 
    {
        PetscPrintf(PETSC_COMM_WORLD, "The matrices have the same size: %d x %d\n", (int)rows_M1, (int) cols_M1);
    } 
    else 
    {
        PetscPrintf(PETSC_COMM_WORLD, "The matrices have different sizes!\n");
    }   


    // Check matrix types
    MatType type1, type2;
    ierr = MatGetType(M1, &type1);CHKERRQ(ierr);
    ierr = MatGetType(M2, &type2);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Types - M1: %s, M2: %s\n", type1, type2);

    // Check norms
    PetscReal norm1, norm2;
    ierr = MatNorm(M1, NORM_FROBENIUS, &norm1);CHKERRQ(ierr);
    ierr = MatNorm(M2, NORM_FROBENIUS, &norm2);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Norms - Mat 1: %.15e, Mat 2: %.15e\n", norm1, norm2);
    PetscPrintf(PETSC_COMM_WORLD, "Relative difference: %.15e\n", PetscAbsReal(norm1 - norm2) / PetscMax(norm1, norm2));  

    // MatMulEqual check
    PetscBool flg;
    MatMultEqual(M1, M2, 5, &flg);
    if (!flg) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                "Matrices do not produce identical MatMult results");
    }
    else {
        PetscPrintf(PETSC_COMM_WORLD, "MatMult results of both matrices are identical for 5 random vectors.\n");
    }
    
    PetscFunctionReturn(0);

}

PetscErrorCode MatPrint(const Mat M, PetscInt rows = 10, PetscInt cols = 10)
{
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    (void)cols;

    PetscInt rows_M, cols_M;
    ierr = MatGetSize(M, &rows_M, &cols_M); CHKERRQ(ierr);

    PetscInt r_start, r_end, c_start, c_end;
    ierr = MatGetOwnershipRange(M, &r_start, &r_end); CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(M, &c_start, &c_end); CHKERRQ(ierr);

    PetscInt print_rows = PetscMin(rows, rows_M);

    PetscInt rstart, rend;
    MatGetOwnershipRange(M, &rstart, &rend);
    for (PetscInt row = rstart; row < print_rows; ++row) 
    {
        PetscScalar vals[4];
        PetscInt    cols[4] = {0, 1, 2, 3};
        MatGetValues(M, 1, &row, 4, cols, vals);
        PetscPrintf(PETSC_COMM_WORLD, "Row %d: ", (int)row);
        for (PetscInt j = 0; j < 4; ++j) 
        {
            PetscPrintf(PETSC_COMM_WORLD, "%.6e + %.6ei  ", PetscRealPart(vals[j]), PetscImaginaryPart(vals[j]));
        }
        PetscPrintf(PETSC_COMM_WORLD, "\n");
    
    
    
    }

    PetscFunctionReturn(0);
}



PetscErrorCode SparsityPatternCheck(const Mat M1, const Mat M2)
{
    PetscErrorCode ierr;
    PetscFunctionBeginUser;            

    // Compare the sparsity patterns of two matrices
    Mat C;
    MatDuplicate(M1,MAT_DO_NOT_COPY_VALUES,&C);
    MatAXPY(C,1.0,M2,DIFFERENT_NONZERO_PATTERN);
    
    MatInfo info;
    MatGetInfo(C,MAT_GLOBAL_SUM,&info);
    
    if (info.nz_used == 0) {
      PetscPrintf(PETSC_COMM_WORLD,"Sparsity patterns match exactly\n");
    }
    
    else {
      PetscPrintf(PETSC_COMM_WORLD,"Sparsity patterns do not match.\n");
    }

    ierr = MatDestroy(&C); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}




PetscErrorCode TDSEZMakeStateReal(Vec v_complex, Vec Vre, Mat M)
{
    PetscFunctionBegin;
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject)v_complex,&comm);CHKERRQ(ierr);
    MPI_Comm_rank(comm, &rank);

    // Copy complex vector
    ierr = VecCopy(v_complex, Vre);CHKERRQ(ierr);

    // Get local ownership
    PetscInt start,end;
    ierr = VecGetOwnershipRange(v_complex,&start,&end);CHKERRQ(ierr);

    const PetscScalar *array;
    ierr = VecGetArrayRead(v_complex,&array);CHKERRQ(ierr);

    PetscReal local_phase = 0.0;
    PetscBool found = PETSC_FALSE;
    for (PetscInt i=0;i<end-start;++i) 
    {
        if (PetscAbsScalar(array[i])>1e-14) 
        {
            local_phase = std::atan2(PetscImaginaryPart(array[i]),PetscRealPart(array[i]));
            found = PETSC_TRUE;
            break;
        }
    }
    ierr = VecRestoreArrayRead(v_complex,&array);CHKERRQ(ierr);

    // Determine global phase
    struct {PetscReal phase; int found;} local = {local_phase, found ? 1 : 0}, global;
    MPI_Allreduce(&local,&global,1,MPI_DOUBLE_INT,MPI_MINLOC,comm);
    PetscReal phase = global.phase;

    // Rotate vector by exp(-i*phase)
    PetscScalar *xarr;
    ierr = VecGetArray(Vre,&xarr);CHKERRQ(ierr);
    ierr = VecGetArrayRead(v_complex,&array);CHKERRQ(ierr);
    for (PetscInt i=0;i<end-start;i++) {
        PetscScalar z = array[i] * PetscExpComplex(-PETSC_i * phase);
        xarr[i] = PetscRealPart(z);
    }
    ierr = VecRestoreArrayRead(v_complex,&array);CHKERRQ(ierr);
    ierr = VecRestoreArray(Vre,&xarr);CHKERRQ(ierr);

    // Assemble
    ierr = VecAssemblyBegin(Vre);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Vre);CHKERRQ(ierr);

    // Normalize
    Vec tmp;
    ierr = VecDuplicate(Vre,&tmp);CHKERRQ(ierr);
    ierr = MatMult(M,Vre,tmp);CHKERRQ(ierr);
    PetscScalar norm;
    ierr = VecTDot(tmp,Vre,&norm);CHKERRQ(ierr);
    norm = PetscAbsScalar(norm);
    ierr = VecScale(Vre,1.0/PetscSqrtReal(norm));CHKERRQ(ierr);
    ierr = VecDestroy(&tmp);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}





PetscErrorCode TDSEZCheckStateIsReal(Vec x, Mat H, Mat M, PetscScalar eigval, PetscReal tol)
{
    PetscErrorCode ierr;
    PetscInt start, end;
    const PetscScalar *array;
    PetscReal max_imag = 0.0;

    // Get local ownership range
    ierr = VecGetOwnershipRange(x, &start, &end); CHKERRQ(ierr); // local indices [start, end)

    // Access local portion
    ierr = VecGetArrayRead(x, &array); CHKERRQ(ierr);
    for (PetscInt i = 0; i < end - start; i++) 
    {
        if (PetscAbsScalar(PetscImaginaryPart(array[i])) > max_imag) 
        {
            max_imag = PetscAbsScalar(PetscImaginaryPart(array[i]));
        }
    }
    ierr = VecRestoreArrayRead(x, &array); CHKERRQ(ierr);

    if (max_imag < tol) {
        PetscPrintf(PETSC_COMM_WORLD, "Vector is effectively real (max imag part = %g)\n", (double)max_imag);
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Vector has significant imaginary components (max imag part = %g)\n", (double)max_imag);
    }

    // Compute residual for generalized eigenproblem: r = H*x - eigval*M*x
    Vec tmp, r;
    ierr = VecDuplicate(x, &tmp); CHKERRQ(ierr);
    ierr = VecDuplicate(x, &r); CHKERRQ(ierr);

    ierr = MatMult(M, x, tmp); CHKERRQ(ierr);   // tmp = M*x
    ierr = MatMult(H, x, r); CHKERRQ(ierr);     // r = H*x
    ierr = VecAXPY(r, -PetscRealPart(eigval), tmp); CHKERRQ(ierr); // r = H*x - eigval*M*x

    // Compute norm of residual
    PetscReal res_norm;
    ierr = VecNorm(r, NORM_2, &res_norm); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Residual norm = %g\n", (double)res_norm);

    if (res_norm < tol) {
        PetscPrintf(PETSC_COMM_WORLD, "Vector corresponds to eigenvalue %g (within tolerance %g)\n",
               (double)PetscRealPart(eigval), (double)tol);
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Residual too large! Vector may not correspond to eigenvalue.\n");
    }

    ierr = VecDestroy(&tmp); CHKERRQ(ierr);
    ierr = VecDestroy(&r); CHKERRQ(ierr);

    return 0;
}




PetscErrorCode TDSEZExpectationValue(Mat O, Vec psi, PetscScalar *expectation)
{
    PetscErrorCode ierr;
    Vec temp;
    PetscFunctionBeginUser;

    // duplicate vector for O*psi
    ierr = VecDuplicate(psi, &temp); CHKERRQ(ierr);

    // temp = O * psi
    ierr = MatMult(O, psi, temp); CHKERRQ(ierr);

    // <psi|O|psi>
    ierr = VecTDot(temp, psi, expectation); CHKERRQ(ierr);

    // --- SAFETY CHECK ---
    PetscReal mag = PetscAbsScalar(*expectation);
    const PetscReal maxAllowed = 1e6;  // threshold for "physical" value
    if (mag > maxAllowed || PetscIsInfScalar(*expectation) || PetscIsNanScalar(*expectation)) {
        PetscPrintf(PETSC_COMM_WORLD,
            "Warning: <O> = %.12g is unphysically large. Setting to 0.\n",
            PetscRealPart(*expectation));
        *expectation = 0.0;
    }

    ierr = VecDestroy(&temp); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode TDSEZHermCheck(Mat O, PetscReal tol)
{
    PetscErrorCode ierr;
    Mat O_dag;
    PetscFunctionBeginUser;

    // Create Hermitian conjugate of O
    ierr = MatHermitianTranspose(O, MAT_INITIAL_MATRIX, &O_dag); CHKERRQ(ierr);

    // Compute difference: D = O - O_dag
    Mat D;
    ierr = MatDuplicate(O, MAT_COPY_VALUES, &D); CHKERRQ(ierr);
    ierr = MatAXPY(D, -1.0, O_dag, SAME_NONZERO_PATTERN); CHKERRQ(ierr);

    // Compute norm of D
    PetscReal norm;
    ierr = MatNorm(D, NORM_FROBENIUS, &norm); CHKERRQ(ierr);

    if (norm < tol) {
        PetscPrintf(PETSC_COMM_WORLD, "Operator is Hermitian within tolerance %g (norm of difference = %g)\n",
               (double)tol, (double)norm);
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Operator is NOT Hermitian (norm of difference = %g)\n", (double)norm);
    }

    ierr = MatDestroy(&O_dag); CHKERRQ(ierr);
    ierr = MatDestroy(&D); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
