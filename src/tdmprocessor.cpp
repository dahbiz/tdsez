// ============================================================================
//  tdmprocessor.cpp  —  standalone post-processor
// ----------------------------------------------------------------------------
//  Reads the assembled dipole operator Dx (PETSc binary, written by the
//  SaveDipoleMatrix / -save_dipole path) and the eigenstates from the static
//  HDF5 (EigenData_<stem>.h5, datasets psi_0, psi_1, ...), then computes the
//  transition-dipole matrix
//
//        d_ij = < psi_i | Dx | psi_j >
//
//  Dx is Hermitian, so d_ji = conj(d_ij); only the upper triangle is computed
//  and the lower triangle is filled by conjugation. Writes d_ij to
//  static/TDM_Dx_<stem>.txt (human-readable) and NumPy .npy files:
//  static/TDM_Dx_<stem>.npy (complex NxN, D[i,j]==d_ij),
//  static/EigenEnergies_<stem>.npy (float64 N), static/States_<stem>.npy (int32 N).
//
//  Build (separately, against the same PETSc stack):
//    mpicxx tdmprocessor.cpp -I$PETSC_DIR/include -L$PETSC_DIR/lib
//          -lpetsc -o tdmprocessor
//
//  Run:
//    ./tdmprocessor <input-stem>
//  e.g. for input file foo.prm:
//    ./tdmprocessor foo.prm
//  (it looks for static/Dx_foo.prm.bin and static/EigenData_foo.prm.h5)
// ============================================================================

#include <petsc.h>
#include <petscviewerhdf5.h>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>

// ----------------------------------------------------------------------------
// Minimal NumPy .npy writer (version 1.0, little-endian), no external deps.
// Writes a contiguous array with the given dtype string and shape. This makes
// the TDM trivially loadable from Python:
//     import numpy as np
//     D = np.load("static/TDM_Dx_foo.npy")          # D[i,j] == d_ij
//     E = np.load("static/EigenEnergies_foo.npy")   # E[i]  == energy of psi_i
//     idx = np.load("static/States_foo.npy")        # state labels 0..N-1
// Supported descr: "<c16" (complex128), "<f8" (float64), "<i4" (int32).
// ----------------------------------------------------------------------------
static bool writeNpy(const std::string& path, const char* descr,
                     const std::vector<PetscInt64>& shape,
                     const void* data, size_t nbytes)
{
    // Build the header dict string.
    std::string shapeStr;
    for (size_t k = 0; k < shape.size(); ++k) {
        shapeStr += std::to_string(shape[k]);
        if (k + 1 < shape.size()) shapeStr += ", ";
    }
    if (shape.size() == 1) shapeStr += ","; // NumPy 1-tuple notation
    std::string dict = "{'descr': '" + std::string(descr) +
                       "', 'fortran_order': False, 'shape': (" + shapeStr + "), }";
    // Header must be padded so that magic(6)+2+2+len(dict) is a multiple of 64,
    // and the total header length (after the 10-byte prefix) fits in a uint16.
    size_t unpadded = 10 + dict.size() + 1; // +1 for trailing '\n'
    size_t pad = (64 - (unpadded % 64)) % 64;
    while (dict.size() + 1 + pad > 0xFFFF) --pad; // safety (never triggers)

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write("\x93""NUMPY", 6);
    uint8_t vmajor = 1, vminor = 0;
    f.write(reinterpret_cast<const char*>(&vmajor), 1);
    f.write(reinterpret_cast<const char*>(&vminor), 1);
    uint16_t hlen = (uint16_t)(dict.size() + 1 + pad);
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(dict.c_str(), dict.size());
    for (size_t p = 0; p < pad; ++p) f.put(' ');
    f.put('\n');
    f.write(reinterpret_cast<const char*>(data), nbytes);
    f.close();
    return (bool)f;
}

// --- Streaming .npy support (write a huge array row-block by row-block so the
//     full matrix never lives in host RAM). ---
// createNpy(): writes header + zero-filled body for an array of the given
//     shape, returns the header length (bytes) the caller must pass to
//     writeNpyRows().
static size_t createNpy(const std::string& path, const char* descr,
                        const std::vector<PetscInt64>& shape)
{
    std::string shapeStr;
    for (size_t k = 0; k < shape.size(); ++k) {
        shapeStr += std::to_string(shape[k]);
        if (k + 1 < shape.size()) shapeStr += ", ";
    }
    if (shape.size() == 1) shapeStr += ",";
    std::string dict = "{'descr': '" + std::string(descr) +
                       "', 'fortran_order': False, 'shape': (" + shapeStr + "), }";
    size_t unpadded = 10 + dict.size() + 1;
    size_t pad = (64 - (unpadded % 64)) % 64;
    while (dict.size() + 1 + pad > 0xFFFF) --pad;

    std::ofstream f(path, std::ios::binary);
    if (!f) return 0;
    f.write("\x93""NUMPY", 6);
    uint8_t vmajor = 1, vminor = 0;
    f.write(reinterpret_cast<const char*>(&vmajor), 1);
    f.write(reinterpret_cast<const char*>(&vminor), 1);
    uint16_t hlen = (uint16_t)(dict.size() + 1 + pad);
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(dict.c_str(), dict.size());
    for (size_t p = 0; p < pad; ++p) f.put(' ');
    f.put('\n');
    // zero-filled body
    size_t total = 1;
    for (auto s : shape) total *= (size_t)s;
    std::string zeros(64*1024, '\0');
    for (size_t written = 0; written < total; written += zeros.size()) {
        size_t chunk = std::min(zeros.size(), total - written);
        f.write(zeros.data(), chunk);
    }
    f.close();
    return 10 + hlen; // header length (prefix 10 + hlen)
}

// writeNpyRect(): overwrite a sub-rectangle rows [rowStart, rowStart+nRows) x
//     cols [colStart, colStart+nCols). buf holds nRows rows of nCols each
//     (row-major), i.e. buf[r*nCols + c] -> matrix (rowStart+r, colStart+c).
//     Used so the symmetric (conjugate) fill writes ONLY its own rectangle and
//     never clobbers the upper triangle computed elsewhere.
static bool writeNpyRect(const std::string& path, size_t hdrLen,
                         PetscInt64 nCols, PetscInt64 rowStart, PetscInt64 nRows,
                         PetscInt64 colStart, PetscInt64 nColsRect,
                         const void* buf, size_t esize)
{
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) return false;
    const char* p = reinterpret_cast<const char*>(buf);
    for (PetscInt64 r = 0; r < nRows; ++r) {
        std::streampos off = (std::streampos)hdrLen +
            (std::streampos)((rowStart + r) * nCols + colStart) * (PetscInt64)esize;
        f.seekp(off, std::ios::beg);
        // buf is laid out with full stride nCols: buf[r*nCols + (colStart+c)]
        f.write(p + (size_t)r * (size_t)nCols * esize + (size_t)colStart * esize,
                (size_t)nColsRect * esize);
    }
    f.close();
    return (bool)f;
}

// --- Lightweight eigenstate access for streaming ---
// loadSpectrum(): read only Nstates + energies (no eigenstate vectors held).
static PetscErrorCode loadSpectrum(const std::string& h5path,
                                   PetscInt& nStates,
                                   std::vector<PetscReal>& energies)
{
    PetscFunctionBeginUser;
    PetscViewer viewer;
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, h5path.c_str(),
                                  FILE_MODE_READ, &viewer));
    Vec spec;
    PetscCall(VecCreate(PETSC_COMM_WORLD, &spec));
    PetscCall(VecSetType(spec, VECSEQ));
    PetscCall(VecSetFromOptions(spec));
    PetscCall(PetscObjectSetName((PetscObject)spec, "spectrum"));
    PetscErrorCode perr = VecLoad(spec, viewer);
    PetscCheck(perr == PETSC_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "tdmprocessor: 'spectrum' dataset missing in %s", h5path.c_str());
    PetscCall(VecGetSize(spec, &nStates));
    const PetscScalar* arr;
    PetscCall(VecGetArrayRead(spec, &arr));
    energies.resize(nStates);
    for (PetscInt k = 0; k < nStates; ++k) energies[k] = PetscRealPart(arr[k]);
    PetscCall(VecRestoreArrayRead(spec, &arr));
    PetscCall(VecDestroy(&spec));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
}

// loadState(): load a single eigenstate psi_i into a fresh Vec using an
// already-open HDF5 viewer (reused across all states to avoid reopening the
// file thousands of times). Caller owns the returned Vec.
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

int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, PETSC_NULLPTR, PETSC_NULLPTR));

    if (argc < 2) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "Usage: %s <input-stem>\n  e.g. %s foo.prm\n", argv[0], argv[0]));
        PetscCall(PetscFinalize());
        return 1;
    }
    std::string stem = argv[1];
    std::string dxPath   = "static/Dx_"   + stem + ".bin";
    std::string eigPath  = "static/EigenData_" + stem + ".h5";

    // Optional -gpu flag: route Dx and the eigenstate vecs to CUDA so the
    // (potentially huge) d_ij inner products run on the GPU. Single-GPU only
    // (uses SEQAIJCUSPARSE / VecCUDA), so it requires a serial run. We read it
    // through PetscOptionsGetBool so PETSc consumes the flag (no "Option left"
    // warning) and we get a clean bool.
    PetscBool gpuB = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-gpu", &gpuB, NULL));
    bool gpu = (gpuB == PETSC_TRUE);
    if (gpu) {
        PetscMPIInt sz = 0;
        PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &sz));
        PetscCheck(sz == 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
                   "tdmprocessor: -gpu uses single-GPU (SEQAIJCUSPARSE); run with np=1");
    }

    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    // --- 1. Load Dx operator matrix (serial, seqaij) ---
    // The matrix is written as a sequential PETSc binary by the solver. For now
    // we load it serially on COMM_WORLD (seqaij). Parallel distribution of Dx
    // is left for later; the eigenstates are likewise loaded serially here.
    Mat Dx;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &Dx));
    PetscCall(MatSetType(Dx, MATSEQAIJ));
    {
        PetscViewer v;
        PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, dxPath.c_str(),
                                        FILE_MODE_READ, &v));
        PetscCall(MatLoad(Dx, v));
        PetscCall(PetscViewerDestroy(&v));
    }
    PetscCall(MatAssemblyBegin(Dx, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Dx, MAT_FINAL_ASSEMBLY));

    if (gpu) {
        // Move the (large) dipole operator onto the GPU as a cuSPARSE matrix.
        // Subsequent MatCreateVecs(Dx,...) then yield CUDA vectors, so the
        // eigenstate loads and all MatMult/VecDot run on device.
        Mat DxGPU;
        PetscCall(MatConvert(Dx, MATSEQAIJCUSPARSE, MAT_INITIAL_MATRIX, &DxGPU));
        PetscCall(MatDestroy(&Dx));
        Dx = DxGPU;
    }

    PetscInt nrows, ncols;
    PetscCall(MatGetSize(Dx, &nrows, &ncols));
    PetscCheck(nrows == ncols, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "tdmprocessor: Dx is not square (%" PetscInt_FMT " x %" PetscInt_FMT ")",
               nrows, ncols);

    // --- 2. Get Nstates + energies (no eigenstate vectors held in RAM) ---
    std::vector<PetscReal> energies;
    PetscInt N = 0;
    PetscCall(loadSpectrum(eigPath, N, energies));
    PetscCheck(N > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
               "tdmprocessor: no eigenstates found in %s", eigPath.c_str());

    // --- 3. Compute d_ij = <psi_i | Dx | psi_j>  (streaming, block by block) ---
    // At most B row-vectors + one column vector + Dx reside in memory at once,
    // and the full N x N matrix is written incrementally to the .npy, so host
    // RAM stays O(B) regardless of N (e.g. 7000 states / 30008^2 Dx).
    const PetscInt B = 256; // row-block size; tune to fit device/host memory
    std::vector<PetscInt64> shape2 = {(PetscInt64)N, (PetscInt64)N};
    std::vector<PetscInt64> shape1 = {(PetscInt64)N};
    std::string npyD = "static/TDM_Dx_" + stem + ".npy";
    std::string npyE = "static/EigenEnergies_" + stem + ".npy";
    std::string npyS = "static/States_" + stem + ".npy";
    std::error_code ec;
    std::filesystem::create_directories("static", ec);
    size_t hdrLen = createNpy(npyD, "<c16", shape2);
    PetscCheck(hdrLen > 0, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "tdmprocessor: cannot create %s", npyD.c_str());

    // Open the eigenstate HDF5 viewer ONCE and reuse it for every state load
    // (reopening per state would be catastrophic at N=7000).
    PetscViewer ev;
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, eigPath.c_str(),
                                  FILE_MODE_READ, &ev));

    Vec Dpsi;
    PetscCall(loadState(ev, Dx, 0, &Dpsi)); // any state gives the layout
    PetscCall(VecDestroy(&Dpsi));

    // --- 3. Compute d_ij = <psi_i | Dx | psi_j>, upper triangle only ---
    // Dx is Hermitian => d_ji = conj(d_ij). We compute each off-diagonal pair
    // exactly once (upper triangle, j >= i) and write the conjugate into the
    // symmetric slot in the same super-block, so no entry is ever recomputed
    // and no full matrix is held in RAM. Diagonal (i=j) is computed once.
    std::vector<PetscScalar> blockRows((size_t)B * N, 0.0); // rows [i0,i1)
    std::vector<PetscScalar> symRows((size_t)B * N, 0.0);   // rows [j0,j1) (conj)
    PetscInt totalMult = 0, totalDot = 0;
    for (PetscInt i0 = 0; i0 < N; i0 += B) {
        PetscInt i1 = std::min(N, i0 + B);
        PetscInt nblk = i1 - i0;
        // Load the B row-vectors psi_i for i in [i0, i1).
        std::vector<Vec> rowVecs((size_t)nblk);
        for (PetscInt b = 0; b < nblk; ++b)
            PetscCall(loadState(ev, Dx, i0 + b, &rowVecs[(size_t)b]));

        // Pre-conjugate the bra block ONCE: <psi_i|Dx|psi_j> = VecDot(conj(psi_i), Dx psi_j).
        // NOTE: in this PETSc build VecDot == VecTDot (neither conjugates the
        // first argument), so the bra MUST be conjugated explicitly. Skipping
        // this makes the TDM wrong by a phase for any genuinely complex state.
        std::vector<Vec> rowVecsConj((size_t)nblk);
        for (PetscInt b = 0; b < nblk; ++b) {
            PetscCall(VecDuplicate(rowVecs[(size_t)b], &rowVecsConj[(size_t)b]));
            PetscCall(VecCopy(rowVecs[(size_t)b], rowVecsConj[(size_t)b]));
            PetscCall(VecConjugate(rowVecsConj[(size_t)b]));
        }

        // Upper-triangle column blocks only: j0 >= i0.
        for (PetscInt j0 = i0; j0 < N; j0 += B) {
            PetscInt j1 = std::min(N, j0 + B);
            PetscInt mblk = j1 - j0;
            // Load the B column-vectors psi_j for j in [j0, j1).
            std::vector<Vec> colVecs((size_t)mblk);
            for (PetscInt c = 0; c < mblk; ++c)
                PetscCall(loadState(ev, Dx, j0 + c, &colVecs[(size_t)c]));

            if (rank == 0)
                PetscCall(PetscPrintf(PETSC_COMM_SELF,
                    "  compute d[i,j]  i in [% " PetscInt_FMT " , % " PetscInt_FMT " )  j in [% " PetscInt_FMT " , % " PetscInt_FMT " )  (upper triangle)\n",
                    i0, i1, j0, j1));

            // For each column j: Dpsi_j = Dx * psi_j (once), dot with every row vec.
            Vec Dpsi_j;
            PetscCall(VecDuplicate(colVecs[0], &Dpsi_j));
            for (PetscInt c = 0; c < mblk; ++c) {
                PetscCall(MatMult(Dx, colVecs[(size_t)c], Dpsi_j));
                ++totalMult;
                PetscInt j = j0 + c;
                for (PetscInt b = 0; b < nblk; ++b) {
                    PetscInt i = i0 + b;
                    if (j < i) continue;            // upper triangle only
                    PetscScalar val;
                    // <psi_i | Dx | psi_j> = VecDot( conj(psi_i), Dx psi_j )
                    // (bra explicitly conjugated; VecDot/VecTDot do NOT
                    //  conjugate the first arg in this PETSc build).
                    PetscCall(VecDot(rowVecsConj[(size_t)b], Dpsi_j, &val));
                    ++totalDot;
                    blockRows[(size_t)b * N + j] = val;       // d[i,j] (upper)
                    if (i != j) {
                        // d[j,i] = conj(d[i,j]) (Hermitian). In the diagonal
                        // super-block (j0==i0) the lower row j lies inside this
                        // same row range, so merge into blockRows and write once;
                        // otherwise stash in symRows and write to disjoint rows.
                        if (j0 == i0) blockRows[(size_t)c * N + i] = PetscConj(val);
                        else          symRows[(size_t)c * N + i] = PetscConj(val);
                    }
                }
            }
            PetscCall(VecDestroy(&Dpsi_j));
            for (auto& v : colVecs) PetscCall(VecDestroy(&v));

            // Flush to the .npy via disjoint rectangle writes (no full-matrix
            // buffer, no clobbering). Upper triangle: rows [i0,i1) x cols [j0,j1).
            // Off-diagonal block: the conjugate lives in rows [j0,j1) x cols
            // [i0,j0) (disjoint from the upper rect), written separately.
            // Diagonal block (j0==i0): the lower triangle was merged into
            // blockRows, so write the whole block [i0,i1) x [i0,i1).
            if (j0 == i0) {
                if (!writeNpyRect(npyD, hdrLen, N, i0, nblk, i0, nblk,
                                  blockRows.data(), sizeof(PetscScalar)))
                    PetscCheck(false, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE,
                               "tdmprocessor: write failed for %s", npyD.c_str());
            } else {
                if (!writeNpyRect(npyD, hdrLen, N, i0, nblk, j0, mblk,
                                  blockRows.data(), sizeof(PetscScalar)))
                    PetscCheck(false, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE,
                               "tdmprocessor: write failed for %s", npyD.c_str());
                if (!writeNpyRect(npyD, hdrLen, N, j0, mblk, i0, j0 - i0,
                                  symRows.data(), sizeof(PetscScalar)))
                    PetscCheck(false, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE,
                               "tdmprocessor: write failed for %s", npyD.c_str());
            }
            if (rank == 0)
                PetscCall(PetscPrintf(PETSC_COMM_SELF,
                    "  wrote rows [% " PetscInt_FMT " , % " PetscInt_FMT " )%s\n",
                    i0, i1, (j0 != i0) ? " (and symmetric block)" : ""));
            std::fill(symRows.begin(), symRows.end(), 0.0);
        }
        std::fill(blockRows.begin(), blockRows.end(), 0.0);
        for (auto& v : rowVecs) PetscCall(VecDestroy(&v));
        for (auto& v : rowVecsConj) PetscCall(VecDestroy(&v));
    }
    if (rank == 0)
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
            "  computed upper triangle only: % " PetscInt_FMT " MatMult, % " PetscInt_FMT " VecTDot (vs % " PetscInt_FMT " each for full matrix)\n",
            totalMult, totalDot, (PetscInt)((PetscInt64)N * N)));

    // Energies + state-index .npy (small, written whole).
    if (writeNpy(npyE, "<f8", shape1, energies.data(), (size_t)N * sizeof(PetscReal)))
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Wrote %s\n", npyE.c_str()));
    {
        std::vector<int32_t> idx((size_t)N);
        for (PetscInt i = 0; i < N; ++i) idx[(size_t)i] = (int32_t)i;
        if (writeNpy(npyS, "<i4", shape1, idx.data(), (size_t)N * sizeof(int32_t)))
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Wrote %s\n", npyS.c_str()));
    }

    // --- 4. Report ---
    if (rank == 0) {
        const char* dxtype;
        PetscCall(MatGetType(Dx, &dxtype));
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
            "\n══════════════════════════════════════════════════════════════════════\n"
            "  TRANSITION DIPOLE MATRIX  d_ij = <psi_i | Dx | psi_j>\n"
            "  Dx : %s  (%" PetscInt_FMT " x %" PetscInt_FMT ", type=%s%s)\n"
            "  States from %s : %" PetscInt_FMT "   (streaming, block=%" PetscInt_FMT ")\n"
            "══════════════════════════════════════════════════════════════════════\n",
            dxPath.c_str(), nrows, ncols, dxtype, gpu ? " [GPU]" : "",
            eigPath.c_str(), N, B));
        // Compact table only for small N (avoids 49M prints at N=7000).
        if (N <= 200) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Re(d_ij) matrix:\n   "));
            for (PetscInt j = 0; j < N; ++j) PetscCall(PetscPrintf(PETSC_COMM_SELF, "   j=%-3" PetscInt_FMT, j));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
            // read back a few rows from the .npy for display
            std::vector<PetscScalar> rowbuf((size_t)N);
            std::fstream fin(npyD, std::ios::in | std::ios::binary);
            for (PetscInt i = 0; i < N; ++i) {
                PetscCall(PetscPrintf(PETSC_COMM_SELF, " i=%-3" PetscInt_FMT " ", i));
                fin.seekg((std::streampos)hdrLen + (std::streampos)(i * N * (PetscInt64)sizeof(PetscScalar)), std::ios::beg);
                fin.read(reinterpret_cast<char*>(rowbuf.data()), (size_t)N * sizeof(PetscScalar));
                for (PetscInt j = 0; j < N; ++j)
                    PetscCall(PetscPrintf(PETSC_COMM_SELF, " %+8.4f", (double)PetscRealPart(rowbuf[(size_t)j])));
                PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
            }
            fin.close();
        }
    }

    // --- 5. Optional human-readable .txt (small N only) ---
    if (rank == 0 && N <= 200) {
        std::string outPath = "static/TDM_Dx_" + stem + ".txt";
        std::FILE* fp = std::fopen(outPath.c_str(), "w");
        if (fp) {
            std::fprintf(fp, "# Transition dipole matrix d_ij = <psi_i | Dx | psi_j>\n");
            std::fprintf(fp, "# Nstates = %" PetscInt_FMT "   Dx = %s\n", N, dxPath.c_str());
            std::fprintf(fp, "# row = i, col = j ; columns: Re(d) Im(d) |d|\n");
            std::vector<PetscScalar> rb((size_t)N);
            std::fstream fin(npyD, std::ios::in | std::ios::binary);
            for (PetscInt i = 0; i < N; ++i) {
                fin.seekg((std::streampos)hdrLen + (std::streampos)(i * N * (PetscInt64)sizeof(PetscScalar)), std::ios::beg);
                fin.read(reinterpret_cast<char*>(rb.data()), (size_t)N * sizeof(PetscScalar));
                for (PetscInt j = 0; j < N; ++j) {
                    PetscScalar z = rb[(size_t)j];
                    PetscReal mag = PetscSqrtReal(PetscRealPart(z * PetscConj(z)));
                    std::fprintf(fp, "%" PetscInt_FMT " %" PetscInt_FMT " %+.8e %+.8e %.8e\n",
                                 i, j, (double)PetscRealPart(z), (double)PetscImaginaryPart(z), (double)mag);
                }
            }
            fin.close();
            std::fclose(fp);
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Wrote %s\n", outPath.c_str()));
        }
    }
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Wrote %s\n", npyD.c_str()));


    // --- cleanup ---
    PetscCall(PetscViewerDestroy(&ev));
    PetscCall(MatDestroy(&Dx));

    PetscCall(PetscFinalize());
    return 0;
}
