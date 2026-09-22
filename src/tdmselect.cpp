// ============================================================================
//  compute_dij_select.cpp  --  selective transition-dipole vector with DOS
// ----------------------------------------------------------------------------
//  Computes, for a CHOSEN bra state psi_i and a user-supplied energy
//  threshold E_thr, the dipole matrix elements and density of states (DOS)
//  with full high-precision numerical formatting.
// ============================================================================

/**
 * @file tdmselect.cpp
 * @brief Selective transition-dipole vector computation with density-of-states
 *        weighting for a chosen bra state and energy threshold.
 */

#include <petsc.h>
#include <petscviewerhdf5.h>
#include <hdf5.h>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <tuple>
#include <fstream>
#include <limits>

// --- read spectrum energies (real parts) via direct HDF5 read -------------
/// Read spectrum energies (real parts) via direct HDF5 read.
/**
 * Opens the EigenData HDF5 file and reads the 'spectrum' dataset to extract
 * the number of states and their real-valued energies.
 * @param[in]  h5path  Path to the HDF5 file.
 * @param[out] energies Vector of eigenstate energies (real parts).
 * @param[out] nStates Number of eigenstates found.
 * @return PETSc error code.
 */
static PetscErrorCode loadSpectrum(const std::string& h5path,
                                   std::vector<PetscReal>& energies,
                                   PetscInt& nStates)
{
    PetscFunctionBeginUser;
    hid_t file = H5Fopen(h5path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    PetscCheck(file >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "loadSpectrum: cannot open %s", h5path.c_str());
    hid_t dset = H5Dopen2(file, "spectrum", H5P_DEFAULT);
    PetscCheck(dset >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
               "loadSpectrum: 'spectrum' dataset missing in %s", h5path.c_str());
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, NULL);
    nStates = (PetscInt)dims[0];
    hid_t ftype = H5Dget_type(dset);
    
    // High-precision buffer allocation (double-precision interleaved [re, im])
    std::vector<double> buf(2 * nStates);
    herr_t hr = H5Dread(dset, ftype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    PetscCheck(hr >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
               "loadSpectrum: H5Dread failed");
    
    energies.resize(nStates);
    for (PetscInt k = 0; k < nStates; ++k) {
        energies[k] = static_cast<PetscReal>(buf[2 * k]);
    }
    
    H5Tclose(ftype); H5Sclose(space); H5Dclose(dset); H5Fclose(file);
    PetscFunctionReturn(PETSC_SUCCESS);
}

// --- load a single eigenstate psi_<i> via VecLoad --------------------------
/// Load a single eigenstate psi_i into a fresh Vec via VecLoad.
/**
 * Creates a Vec with the layout of Dx, names it "psi_<i>", and loads the
 * corresponding dataset from the open HDF5 viewer.
 * @param[in]  viewer Already-open HDF5 viewer.
 * @param[in]  Dx     Dipole operator (provides the Vec layout).
 * @param[in]  i      State index to load.
 * @param[out] v      Newly created and loaded Vec (caller owns).
 * @return PETSc error code.
 */
static PetscErrorCode loadState(PetscViewer viewer, Mat Dx,
                                PetscInt i, Vec* v)
{
    PetscFunctionBeginUser;
    Vec tmpl;
    PetscCall(MatCreateVecs(Dx, &tmpl, NULL));
    PetscCall(VecDuplicate(tmpl, v));
    std::string name = "psi_" + std::to_string(i);
    PetscCall(PetscObjectSetName((PetscObject)(*v), name.c_str()));
    PetscCall(VecLoad(*v, viewer));
    PetscCall(VecDestroy(&tmpl));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// Main: compute selective dipole vector with DOS for a chosen bra state.
/**
 * Loads Dx and the eigenstate spectrum, computes d_j = <psi_i|Dx|psi_j> for
 * all states j above an energy threshold, evaluates the density of states
 * (DOS), and writes a high-precision CSV with dipole elements, |d|², and
 * DOS-weighted quantities.
 *
 * @param argc  Number of CLI arguments (expects <stem> -i <state> -Ethr <E>).
 * @param argv  CLI argument strings.
 * @return 0 on success, 1 on usage error.
 */
int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, PETSC_NULLPTR, PETSC_NULLPTR));

    PetscMPIInt commSize = 0;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &commSize));
    PetscCheck(commSize == 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
               "tdmselect: sequential dipole files require a single MPI rank");

    if (argc < 2) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "Usage: %s <stem> -i <state_i> -Ethr <E> [-w0 <omega0>] [-gpu]\n",
            argv[0]));
        PetscCall(PetscFinalize());
        return 1;
    }
    std::string stem = argv[1];

    PetscInt  bra      = 0;
    PetscReal Ethr     = 0.0;
    PetscReal w0       = 0.05655;
    PetscBool gpuB     = PETSC_FALSE;
    PetscCall(PetscOptionsGetInt (NULL, NULL, "-i",    &bra,  NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-Ethr", &Ethr, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-w0",   &w0,   NULL));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-gpu",  &gpuB, NULL));
    bool gpu = (gpuB == PETSC_TRUE);

    std::string dxPath  = "static/Dx_"   + stem + ".bin";
    std::string eigPath = "static/EigenData_" + stem + ".h5";

    Mat Dx;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &Dx));
    PetscCall(MatSetType(Dx, gpu ? MATSEQAIJCUSPARSE : MATSEQAIJ));
    PetscCall(MatSetFromOptions(Dx));
    PetscViewer dv;
    PetscCall(MatCreateVecs(Dx, NULL, NULL)); // Check initialization context
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, dxPath.c_str(),
                                    FILE_MODE_READ, &dv));
    PetscCall(MatLoad(Dx, dv));
    PetscCall(PetscViewerDestroy(&dv));
    PetscCall(MatSetOption(Dx, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(Dx, MAT_HERMITIAN, PETSC_TRUE));

    std::vector<PetscReal> energies;
    PetscInt nStates = 0;
    PetscCall(loadSpectrum(eigPath, energies, nStates));
    if (bra < 0 || bra >= nStates)
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                "bra state -i out of range");

    // --- Compute High-Precision Density of States (DOS) for all states ---
    std::vector<PetscReal> rho(nStates, 0.0);
    for (PetscInt j = 0; j < nStates - 1; ++j) {
        PetscReal dE = energies[j+1] - energies[j];
        if (dE > 1e-15) { // tighter threshold for high-precision steps
            rho[j] = 1.0 / dE;
        }
    }
    if (nStates > 1) {
        rho[nStates - 1] = rho[nStates - 2]; 
    }

    PetscViewer ev;
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, eigPath.c_str(),
                                  FILE_MODE_READ, &ev));

    Vec psi_i, tmp;
    PetscCall(loadState(ev, Dx, bra, &psi_i));
    PetscCall(MatCreateVecs(Dx, &tmp, NULL));
    if (gpu) { PetscCall(VecSetType(tmp, VECCUDA)); PetscCall(VecSetType(psi_i, VECCUDA)); }

    Vec psi_i_conj;
    PetscCall(VecDuplicate(psi_i, &psi_i_conj));
    PetscCall(VecCopy(psi_i, psi_i_conj));
    PetscCall(VecConjugate(psi_i_conj));

    PetscReal Ei = energies[bra];
    std::vector<std::tuple<PetscInt, PetscReal, PetscScalar, PetscReal>> rows;
    rows.reserve(nStates);

    PetscInt kept = 0;
    for (PetscInt j = 0; j < nStates; ++j) {
        if (energies[j] <= Ethr) continue;
        Vec psi_j;
        PetscCall(loadState(ev, Dx, j, &psi_j));
        if (gpu) PetscCall(VecSetType(psi_j, VECCUDA));
        PetscCall(MatMult(Dx, psi_j, tmp));
        PetscScalar d;
        PetscCall(VecDot(psi_i_conj, tmp, &d));
        rows.emplace_back(j, energies[j], d, rho[j]);
        ++kept;
        PetscCall(VecDestroy(&psi_j));
        if ((kept % 500) == 0)
            PetscCall(PetscPrintf(PETSC_COMM_SELF,
                "  computed %" PetscInt_FMT " states (last j=%" PetscInt_FMT ")\n",
                kept, j));
    }
    PetscCall(VecDestroy(&psi_i));
    PetscCall(VecDestroy(&psi_i_conj));
    PetscCall(VecDestroy(&tmp));
    PetscCall(PetscViewerDestroy(&ev));

    std::string csv = "static/dij_select_" + stem + "_i" + std::to_string(bra) + ".csv";
    std::ofstream out(csv);
    
    // Bind output stream to maximum precision representation
    out.precision(std::numeric_limits<double>::max_digits10);

    out << "j,E_j,x,Re_d,Im_d,abs_d2,log10_abs_d2,rho,log10_rho_abs_d2\n";
    
    for (auto& r : rows) {
        PetscInt j; PetscReal Ej; PetscScalar d; PetscReal r_val;
        std::tie(j, Ej, d, r_val) = r;
        PetscReal x_val = (Ej - Ei) / w0;
        PetscReal a2 = PetscRealPart(d) * PetscRealPart(d)
                     + PetscImaginaryPart(d) * PetscImaginaryPart(d);
        PetscReal lg = (a2 > 0.0) ? std::log10(a2) : -300.0;
        
        PetscReal rho_a2 = a2 * r_val;
        PetscReal lg_rho_a2 = (rho_a2 > 0.0) ? std::log10(rho_a2) : -300.0;

        out << j << "," << Ej << "," << x_val << ","
            << PetscRealPart(d) << "," << PetscImaginaryPart(d) << ","
            << a2 << "," << lg << "," << r_val << "," << lg_rho_a2 << "\n";
    }
    out.close();

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "\n=== compute_dij_select with High-Precision DOS ===\n"
        "stem        : %s\n"
        "bra psi_%" PetscInt_FMT "  E_i = %.15f a.u.\n"
        "Ethr        : %.15f  (kept %" PetscInt_FMT " / %" PetscInt_FMT " states)\n"
        "w0          : %.15f\n"
        "csv         : %s\n",
        stem.c_str(), bra, (double)Ei, (double)Ethr, kept, nStates,
        (double)w0, csv.c_str()));

    PetscCall(MatDestroy(&Dx));
    PetscCall(PetscFinalize());
    return 0;
}
