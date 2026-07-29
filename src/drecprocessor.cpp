// ============================================================================
//  resolve_drec.cpp  --  exact recombination dipole d_rec(E)
// ----------------------------------------------------------------------------
//  In a 1-open-channel scattering problem the two independent solutions at
//  energy E are the regular state psi(E) and its energy derivative
//  dpsi/dE (the irregular solution).  A finite-box eigenstate |j> is an
//  arbitrary REAL mixture  a*psi(E) + b*dpsi/dE, which is why the raw
//  matrix element <psi_0|Dx|j> scatters between two branches.
//
//  The physically meaningful (outgoing / recombining) channel is the COMPLEX
//  combination
//        |psi^(+)(E)>  =  |psi(E)>  +  i*g * d|psi>/dE
//  and the recombination dipole is the single smooth function
//        d_rec(E) = <psi_0 | Dx | psi^(+)(E)>
//  which dips to its node at destructive-interference energies.
//
//  We approximate d|psi>/dE by a central finite difference over the two
//  neighbouring box eigenstates:
//        w_j = (psi_{j+1} - psi_{j-1}) / (E_{j+1} - E_{j-1})
//  and set  d_rec(E_j) = d_j + i*g*d_w,
//        d_j = <psi_0|Dx|psi_j>,   d_w = <psi_0|Dx|w_j>.
//  Box states are normalised to 1 in the box; the energy-normalised
//  continuum state is |psi^(+)>/sqrt(rho(E)), so
//        |d_rec|^{phys} = rho(E) * |d_rec|^{box}.
//
//  g is the (energy-independent) Wronskian scale that sets the relative
//  weight of the irregular component; tune with -g (default 0.5*(dE) => the
//  standard unit-flux combination).
//
//  CLI:  resolve_drec <stem> [-i <bra>] [-Ethr <E>] [-w0 <w>] [-g <g>] [-gpu]
// ============================================================================

#include <petsc.h>
#include <petscviewerhdf5.h>
#include <hdf5.h>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <fstream>

static PetscReal g_E0 = 0.0;
static PetscReal g_W0 = 0.05655;

// ---- direct HDF5 read of 'spectrum' (real parts) -------------------------
static PetscErrorCode loadSpectrum(const std::string& h5path,
                                   std::vector<PetscReal>& E)
{
    E.clear();
    hid_t file = H5Fopen(h5path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) SETERRQ(PETSC_COMM_SELF, 1, "cannot open HDF5 %s", h5path.c_str());
    hid_t dset = H5Dopen2(file, "spectrum", H5P_DEFAULT);
    if (dset < 0) { H5Fclose(file); SETERRQ(PETSC_COMM_SELF, 1, "'spectrum' missing"); }
    hid_t ftype = H5Dget_type(dset);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2]; H5Sget_simple_extent_dims(space, dims, NULL);
    std::vector<double> buf(2 * dims[0]);
    H5Dread(dset, ftype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    E.resize(dims[0]);
    for (hsize_t k = 0; k < dims[0]; ++k) E[k] = (PetscReal)buf[2 * k]; // real part
    H5Tclose(ftype); H5Sclose(space); H5Dclose(dset); H5Fclose(file);
    return 0;
}

// ---- load a single psi dataset by index -----------------------------------
static PetscErrorCode loadState(Mat Dx, const std::string& h5path, PetscInt j, Vec* v)
{
    PetscErrorCode ierr;
    PetscViewer view;
    ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, h5path.c_str(), FILE_MODE_READ, &view); CHKERRQ(ierr);
    ierr = MatCreateVecs(Dx, v, NULL); CHKERRQ(ierr);
    std::string name = "psi_" + std::to_string(j);
    ierr = PetscObjectSetName((PetscObject)(*v), name.c_str()); CHKERRQ(ierr);
    ierr = VecLoad(*v, view); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view); CHKERRQ(ierr);
    return 0;
}

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscErrorCode ierr;
    MPI_Comm comm = PETSC_COMM_WORLD;

    std::string stem = "h2p.inp";
    PetscInt iBra = 0;
    PetscReal Ethr = -5.0, gfac = 0.5;
    PetscBool gpu = PETSC_FALSE;
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        if (s == "-i" && a + 1 < argc) iBra = (PetscInt)std::strtol(argv[++a], NULL, 10);
        else if (s == "-Ethr" && a + 1 < argc) Ethr = (PetscReal)std::strtod(argv[++a], NULL);
        else if (s == "-w0" && a + 1 < argc) g_W0 = (PetscReal)std::strtod(argv[++a], NULL);
        else if (s == "-g" && a + 1 < argc) gfac = (PetscReal)std::strtod(argv[++a], NULL);
        else if (s == "-gpu") gpu = PETSC_TRUE;
        else if (s.rfind("-", 0) != 0) stem = s;
    }

    std::string HERE = "."; // run from the dir containing static/
    std::string dxpath = "static/Dx_" + stem + ".bin";
    std::string h5path = "static/EigenData_" + stem + ".h5";

    if (gpu) {
        ierr = PetscOptionsSetValue(NULL, "-vec_type", "cuda"); CHKERRQ(ierr);
        ierr = PetscOptionsSetValue(NULL, "-mat_type", "aijcusparse"); CHKERRQ(ierr);
    }

    // ---- Dx ----
    Mat Dx;
    PetscViewer dv;
    ierr = PetscViewerBinaryOpen(comm, dxpath.c_str(), FILE_MODE_READ, &dv); CHKERRQ(ierr);
    ierr = MatCreate(comm, &Dx); CHKERRQ(ierr);
    ierr = MatLoad(Dx, dv); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&dv); CHKERRQ(ierr);

    // ---- spectrum (energies, sorted by E) ----
    std::vector<PetscReal> Eall;
    ierr = loadSpectrum(h5path, Eall); CHKERRQ(ierr);
    PetscInt N = (PetscInt)Eall.size();

    // sort indices by energy
    std::vector<PetscInt> ord(N);
    for (PetscInt k = 0; k < N; ++k) ord[k] = k;
    std::sort(ord.begin(), ord.end(),
              [&](PetscInt a, PetscInt b) { return Eall[a] < Eall[b]; });
    std::vector<PetscReal> E(N);
    for (PetscInt k = 0; k < N; ++k) E[k] = Eall[ord[k]];
    g_E0 = E[0];

    // density of states rho_j = dN/dE (central diff)
    std::vector<PetscReal> rho(N);
    std::vector<PetscReal> dEhalf(N);
    for (PetscInt k = 0; k < N; ++k) {
        if (k == 0) rho[k] = 1.0 / (E[1] - E[0]);
        else if (k == N - 1) rho[k] = 1.0 / (E[N - 1] - E[N - 2]);
        else rho[k] = 2.0 / (E[k + 1] - E[k - 1]);
        PetscInt klo = (k > 0) ? k - 1 : k;
        PetscInt khi = (k < N - 1) ? k + 1 : k;
        dEhalf[k] = (khi > klo) ? 0.5 * (E[khi] - E[klo]) : 0.0;
    }

    PetscInt nSelected = 0;
    for (PetscInt k = 0; k < N; ++k) if (E[k] > Ethr) ++nSelected;

    PetscPrintf(comm, "bra psi_ %lld  E_i = %.6f a.u.\n", (long long)iBra, (double)g_E0);
    PetscPrintf(comm, "Ethr        : %.6f  (kept %lld / %lld states)\n",
                (double)Ethr, (long long)nSelected, (long long)N);
    PetscPrintf(comm, "w0          :  %.6f\n", (double)g_W0);
    PetscPrintf(comm, "g (Wronskian scale): %.4f\n", (double)gfac);

    // ---- vectors ----
    Vec psi0, pkm1, pk, pkp1, w, Dxpsi, Dxw;
    ierr = MatCreateVecs(Dx, &psi0, NULL); CHKERRQ(ierr);
    ierr = VecDuplicate(psi0, &pkm1); CHKERRQ(ierr);
    ierr = VecDuplicate(psi0, &pk); CHKERRQ(ierr);
    ierr = VecDuplicate(psi0, &pkp1); CHKERRQ(ierr);
    ierr = VecDuplicate(psi0, &w); CHKERRQ(ierr);
    ierr = VecDuplicate(psi0, &Dxpsi); CHKERRQ(ierr);
    ierr = VecDuplicate(psi0, &Dxw); CHKERRQ(ierr);

    // load bra
    ierr = loadState(Dx, h5path, ord[iBra], &psi0); CHKERRQ(ierr);

    // <psi0 | Dx | psi_k> = VecDot( conj(psi0), Dx psi_k ).
    // In this PETSc build VecDot/VecTDot do NOT conjugate the first arg, so
    // the bra must be conjugated explicitly. Pre-conjugate once.
    Vec psi0_conj; ierr = VecDuplicate(psi0, &psi0_conj); CHKERRQ(ierr);
    ierr = VecCopy(psi0, psi0_conj); CHKERRQ(ierr);
    ierr = VecConjugate(psi0_conj); CHKERRQ(ierr);

    std::ofstream out("static/drec_resolved_" + stem + ".csv");
    out << "j,E_j,x,Re_d,Im_d,abs_d2,log10_abs_d2,Re_dj,Im_dj,Re_dw,Im_dw,dEhalf,rho\n";

    PetscReal gscale = gfac; // weight of irregular component (tuned)

    for (PetscInt k = 0; k < N; ++k) {
        if (E[k] <= Ethr) continue;

        PetscInt klo = (k > 0) ? k - 1 : k;
        PetscInt khi = (k < N - 1) ? k + 1 : k;
        ierr = loadState(Dx, h5path, ord[klo], &pkm1); CHKERRQ(ierr);
        ierr = loadState(Dx, h5path, ord[k],   &pk);    CHKERRQ(ierr);
        ierr = loadState(Dx, h5path, ord[khi], &pkp1);  CHKERRQ(ierr);

        PetscReal dE = (khi > klo) ? (E[khi] - E[klo]) : 1.0;
        // w = (psi_{k+1} - psi_{k-1}) / dE   (central; one-sided at edges)
        VecWAXPY(w, -1.0, pkm1, pkp1);
        VecScale(w, 1.0 / dE);

        // d_j = <psi0 | Dx | psi_k>   (bra conjugated explicitly)
        MatMult(Dx, pk, Dxpsi);
        PetscScalar dj; VecDot(psi0_conj, Dxpsi, &dj);
        // d_w = <psi0 | Dx | w>
        MatMult(Dx, w, Dxw);
        PetscScalar dw; VecDot(psi0_conj, Dxw, &dw);

        // d_rec = dj + i * gscale * (dE/2) * dw   (unit-flux combination)
        PetscScalar d = dj + PetscScalar(0.0, gscale * 0.5 * dE) * dw;
        PetscReal absd2 = PetscRealPart(d * PetscConj(d));
        PetscReal absd2_phys = absd2 * rho[k];          // energy-normalised
        PetscReal logd = (absd2_phys > 0) ? std::log10(absd2_phys) : -999.0;
        PetscReal x = (E[k] - g_E0) / g_W0;

        out << k << "," << E[k] << "," << x << ","
            << PetscRealPart(d) << "," << PetscImaginaryPart(d) << ","
            << absd2_phys << "," << logd << ","
            << PetscRealPart(dj) << "," << PetscImaginaryPart(dj) << ","
            << PetscRealPart(dw) << "," << PetscImaginaryPart(dw) << ","
            << dEhalf[k] << "," << rho[k] << "\n";
    }
    out.close();

    PetscPrintf(comm, "csv         : static/drec_resolved_%s.csv\n", stem.c_str());

    ierr = VecDestroy(&psi0); CHKERRQ(ierr);
    ierr = VecDestroy(&psi0_conj); CHKERRQ(ierr);
    ierr = VecDestroy(&pkm1); CHKERRQ(ierr);
    ierr = VecDestroy(&pk); CHKERRQ(ierr);
    ierr = VecDestroy(&pkp1); CHKERRQ(ierr);
    ierr = VecDestroy(&w); CHKERRQ(ierr);
    ierr = VecDestroy(&Dxpsi); CHKERRQ(ierr);
    ierr = VecDestroy(&Dxw); CHKERRQ(ierr);
    ierr = MatDestroy(&Dx); CHKERRQ(ierr);
    PetscFinalize();
    return 0;
}
