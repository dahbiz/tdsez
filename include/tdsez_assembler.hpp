#ifndef TDSEZ_ASSEMBLER_HPP
#define TDSEZ_ASSEMBLER_HPP

#include <petiga.h>
#include <petscmat.h>

// ============================================================================
//  TDSEZAssembler  — builds the linear operators of the TDSE Hamiltonian
// ============================================================================
/**
 * @brief Builds the linear operators (matrices) of the TDSE Hamiltonian.
 * @details Owns and assembles the mass, kinetic, potential, dipole, velocity,
 * potential-gradient, CAP, and angular-momentum matrices from the IGA
 * discretisation. The destructor releases all PETSc matrix handles.
 */
class TDSEZAssembler
{
    private:
        /// Reference to the IGA object used for assembly.
        // Private members
        IGA& iga;

    public:
        /**
         * @brief Construct the assembler bound to an IGA object.
         * @param iga Isogeometric analysis context to assemble from.
         */
        // Constructors
        TDSEZAssembler(IGA& iga);

        /**
         * @brief Destructor; destroys all owned PETSc matrices.
         */
        // Destructor
        ~TDSEZAssembler()
        {
            if (Dx)     MatDestroy(&Dx);
            if (Dy)     MatDestroy(&Dy);
            if (Dz)     MatDestroy(&Dz);
            if (VelX)   MatDestroy(&VelX);
            if (VelY)   MatDestroy(&VelY);
            if (VelZ)   MatDestroy(&VelZ);
            if (dVdx)   MatDestroy(&dVdx);
            if (dVdy)   MatDestroy(&dVdy);
            if (dVdz)   MatDestroy(&dVdz);
            if (CAP)    MatDestroy(&CAP);
            if (Md)     MatDestroy(&Md);
            if (K)      MatDestroy(&K);
            if (V)      MatDestroy(&V);
            if (Lz)     MatDestroy(&Lz);
        }


    public:
        // Public references maintain original interface
        /// Mass-distribution operator matrix Md.
        Mat Md = PETSC_NULLPTR; // Md matrix
        /// Dipole (position) operator matrices along x, y, z.
        Mat Dx = PETSC_NULLPTR, Dy = PETSC_NULLPTR, Dz = PETSC_NULLPTR;
        /// Velocity-gauge operator matrices along x, y, z.
        Mat VelX = PETSC_NULLPTR, VelY = PETSC_NULLPTR, VelZ = PETSC_NULLPTR;
        /// Potential-gradient operator matrices dV/dx, dV/dy, dV/dz, and CAP.
        Mat dVdx = PETSC_NULLPTR, dVdy = PETSC_NULLPTR, dVdz = PETSC_NULLPTR, CAP = PETSC_NULLPTR; // CAP matrix
        /// Kinetic energy matrix K.
        Mat K = PETSC_NULLPTR;   // Kinetic energy matrix
        /// Potential energy matrix V.
        Mat V = PETSC_NULLPTR;   // Potential energy matrix
        /// Angular momentum matrix Lz.
        Mat Lz = PETSC_NULLPTR;  // Angular momentum matrix
};


#endif // TDSEZ_ASSEMBLER_HPP
