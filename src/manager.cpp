#include "tdsez_paths.hpp"
#include "tdsez_internal.hpp"
#include "tdsez_forcecoeffs.hpp"
#include <algorithm>
#include <cctype>

/**
 * @file manager.cpp
 * @brief TDSEZManager: propagation orchestration, HDF5 I/O, and polarization
 *        callback selection.
 * @author TDSEZ Project
 */

/// @brief Select the IFunction and IJacobian callbacks matching the
///        configured polarization (x, y, z, xy, xz, yz, or xyz/all).
///        Assigns TDSEZIFunctionPtr and TDSEZIJacobianPtr accordingly.
/// @return PetscErrorCode — PETSC_SUCCESS on success, or SETERRQ for an
///         unrecognized polarization string.
PetscErrorCode TDSEZManager::PolarizationSelector()
{
    PetscFunctionBegin;
    std::string polType = TDSEZParser::Polarization;
    if (TDSEZParser::NCouplingTerms() > 0) {
        // coupling terms (FieldTerm, multipoles): one residual/Jacobian over the term list
        TDSEZIFunctionPtr = TDSEZIFunctionTerms; TDSEZIJacobianPtr = TDSEZIJacobianTerms;
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    if      (polType == "x")    {TDSEZIFunctionPtr = TDSEZIFunctionPolX,  TDSEZIJacobianPtr = TDSEZIJacobianPolX;}
    else if (polType == "y")    {TDSEZIFunctionPtr = TDSEZIFunctionPolY,  TDSEZIJacobianPtr = TDSEZIJacobianPolY;}
    else if (polType == "z")    {TDSEZIFunctionPtr = TDSEZIFunctionPolZ,  TDSEZIJacobianPtr = TDSEZIJacobianPolZ;}
    else if (polType == "xy")   {TDSEZIFunctionPtr = TDSEZIFunctionPolXY, TDSEZIJacobianPtr = TDSEZIJacobianPolXY;}
    else if (polType == "xz")   {TDSEZIFunctionPtr = TDSEZIFunctionPolXZ, TDSEZIJacobianPtr = TDSEZIJacobianPolXZ;}
    else if (polType == "yz")   {TDSEZIFunctionPtr = TDSEZIFunctionPolYZ, TDSEZIJacobianPtr = TDSEZIJacobianPolYZ;}
    else if (polType == "xyz" || polType == "all") {TDSEZIFunctionPtr = TDSEZIFunctionPolXYZ, TDSEZIJacobianPtr = TDSEZIJacobianPolXYZ;}
    else 
    {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "The polarization choice is not compatible with the  dimension of the problem. Please check your input file.");
    }

        PetscFunctionReturn(PETSC_SUCCESS);
}



/// @brief Write accumulated time-series data (dipole, population, energy,
///        current, autocorrelation) from ring buffers to an HDF5 file.
///        Runs on rank 0 only. Opens the HDF5 file on demand and keeps the
///        handle open across flushes for incremental writes.
/// @param filename  Path to the HDF5 output file.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZManager::WriteHDF5(const std::string& filename)
{
    PetscFunctionBeginUser;
    if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);

    PetscInt nRows = recordedSteps - lastWrittenSteps;
    if (nRows <= 0) PetscFunctionReturn(PETSC_SUCCESS);
    if (maxAllocatedSteps > 0 && nRows > maxAllocatedSteps) nRows = maxAllocatedSteps;
    if (nRows <= 0) PetscFunctionReturn(PETSC_SUCCESS);

    std::string absPath = std::filesystem::path(filename).is_absolute()
                          ? filename : std::filesystem::absolute(filename).string();

    // Open-on-demand and keep the handle open across flushes (fix D).
    // The first flush creates the file (TRUNC); later flushes reopen (RDWR).
    if (h5File < 0) {
        bool exists = std::filesystem::exists(absPath);
        h5File = exists
                     ? H5Fopen(absPath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT)
                     : H5Fcreate(absPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (h5File < 0) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                    "HDF5: could not open/create %s", absPath.c_str());
        }
        h5Compress      = (bool)TDSEZParser::HDF5Compress;
        h5CompressLevel = (int)TDSEZParser::HDF5CompressLevel;
        if (h5CompressLevel < 1) h5CompressLevel = 1;
        if (h5CompressLevel > 9) h5CompressLevel = 9;
    }
    hid_t file = h5File;

    // Fixed, robust chunk shape (fix E): independent of the first-flush window
    // size so we never end up with many tiny chunks when OutputStrideTS is small.
    const PetscInt CHUNK_ROWS = 256;

    auto AppendDataset = [&](const char* name, const PetscScalar* buf, PetscInt cols) -> herr_t {
        hsize_t curDims[2] = {0, (hsize_t)cols};
        hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
        if (dset < 0) {
            // Dataset does not exist yet -> create it (chunked, extendable, optional gzip).
            hsize_t maxdims[2] = {H5S_UNLIMITED, (hsize_t)cols};
            hsize_t chunk_dims[2] = {(hsize_t)CHUNK_ROWS, (hsize_t)cols};
            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            if (dcpl < 0) return -1;
            if (H5Pset_chunk(dcpl, 2, chunk_dims) < 0) { H5Pclose(dcpl); return -1; }
            if (h5Compress) {
                // shuffle improves deflate ratio for the interleaved real/imag layout
                if (H5Pset_shuffle(dcpl) < 0 ||
                    H5Pset_deflate(dcpl, (unsigned)h5CompressLevel) < 0) {
                    H5Pclose(dcpl);
                    return -1;
                }
            }
            hid_t fspace = H5Screate_simple(2, curDims, maxdims);
            if (fspace < 0) { H5Pclose(dcpl); return -1; }
            dset = H5Dcreate(file, name, H5T_IEEE_F64LE, fspace,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Sclose(fspace);
            H5Pclose(dcpl);
        } else {
            // Existing dataset -> read current extent (do NOT reopen, fix A).
            hid_t fspace = H5Dget_space(dset);
            if (fspace < 0) { H5Dclose(dset); return -1; }
            if (H5Sget_simple_extent_dims(fspace, curDims, NULL) < 0) {
                H5Sclose(fspace);
                H5Dclose(dset);
                return -1;
            }
            H5Sclose(fspace);
        }
        if (dset < 0) return -1;

        // Locate the window inside the ring buffer (fix C).
        // The buffer may be split (wrap) or contiguous; we pass a pointer +
        // contiguous length and write in at most two hyperslab calls. This
        // avoids a full-window heap copy in the common non-wrap case while
        // staying byte-identical to the previous in-place double view.
        PetscInt startOff = (maxAllocatedSteps > 0) ? (lastWrittenSteps % maxAllocatedSteps) : 0;
        PetscInt firstLen = (maxAllocatedSteps > 0) ? (maxAllocatedSteps - startOff) : nRows;
        if (firstLen > nRows) firstLen = nRows;
        PetscInt secondLen = nRows - firstLen;

        const PetscScalar* seg0 = buf + (size_t)startOff * cols;
        const PetscScalar* seg1 = buf; /* used only when secondLen > 0 */

        auto writeSeg = [&](const PetscScalar* p, PetscInt r0, PetscInt len) -> herr_t {
            (void)r0;
            if (len <= 0) return 0;
            hsize_t newDims[2] = {curDims[0] + (hsize_t)len, curDims[1]};
            if (H5Dset_extent(dset, newDims) < 0) return -1;
            hsize_t start[2] = {curDims[0], 0};
            hsize_t count[2] = {(hsize_t)len, (hsize_t)cols};
            hid_t memspace  = H5Screate_simple(2, count, NULL);
            hid_t filespace = H5Dget_space(dset);
            if (memspace < 0 || filespace < 0) {
                if (memspace >= 0) H5Sclose(memspace);
                if (filespace >= 0) H5Sclose(filespace);
                return -1;
            }
            // PetscScalar may be complex; store only the real part so the HDF5
            // dataset has exactly `cols` real columns (matching the schema and
            // avoiding re/im interleaving that corrupts the output).
            std::vector<double> realbuf((size_t)len * (size_t)cols);
            for (PetscInt i = 0; i < len * cols; ++i) realbuf[i] = PetscRealPart(p[i]);
            herr_t werr = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                               start, NULL, count, NULL);
            if (werr >= 0)
                werr = H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace,
                                H5P_DEFAULT, realbuf.data());
            H5Sclose(memspace);
            H5Sclose(filespace);
            curDims[0] = newDims[0];
            return werr;
        };

        herr_t err = writeSeg(seg0, 0, firstLen);
        if (err >= 0 && secondLen > 0) err = writeSeg(seg1, firstLen, secondLen);

        H5Dclose(dset);
        return err;
    };

    herr_t overall = 0;
    if (nRows > 0) {
        // Contract: every dataset is always present with the same shape.
        // A quantity disabled via PhysicsOutput keeps its time column and is
        // zero elsewhere (stable schema for downstream readers) while its
        // per-step compute is skipped.
        // The buffers are zeroed at allocation and the monitors fill column 0
        // (time) of every dataset, the other columns only for the requested
        // quantities, so a disabled quantity is time + zeros.
        const PetscScalar* dipSrc = dipBuffer;
        const PetscScalar* popSrc = popBuffer;
        const PetscScalar* eneSrc = energyBuffer;
        const PetscScalar* curSrc = currBuffer;
        const PetscScalar* acSrc  = acBuffer;

        overall |= AppendDataset("dipoles",         dipSrc, dipWidth);
        overall |= AppendDataset("populations",     popSrc, popWidth);
        overall |= AppendDataset("energies",        eneSrc, energyWidth);
        overall |= AppendDataset("currents",        curSrc, (TDSEZParser::Dimension == 3) ? 15 : 11);
        overall |= AppendDataset("autocorrelation", acSrc,  3);
    }

    if (overall < 0) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                "HDF5: failed to append datasets to %s", absPath.c_str());
    }
    lastWrittenSteps = recordedSteps;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Close the HDF5 output file handle, flushing any remaining
///        unflushed rows. Called after propagation completes to finalize
///        the time-evolution data file.
/// @return PetscErrorCode — PETSC_SUCCESS on success.
PetscErrorCode TDSEZManager::CloseHDF5()
{
    PetscFunctionBeginUser;
    if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);
    // Flush any remaining rows, then close the persistent handle (fix D).
    if (recordedSteps > lastWrittenSteps) {
        PetscCall(WriteHDF5(outputFilename));
    }
    if (h5File >= 0) {
        const herr_t flushStatus = H5Fflush(h5File, H5F_SCOPE_GLOBAL);
        const herr_t closeStatus = H5Fclose(h5File);
        h5File = -1;
        if (flushStatus < 0 || closeStatus < 0)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                    "HDF5: could not flush/close %s", outputFilename.c_str());
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Write the open knot vectors (full, p+1-clamped ends) for each active axis plus
// the SplineDegree attribute into an already-open HDF5 file `file`. This makes a
// wavefunction file self-describing: together with the dof coefficients
// (psi_*/wavefunction) a reader can rebuild the B-spline basis (Cox-de Boor) and
// evaluate the wavefunction on any spatial grid. Returns 0 on success, <0 on error.
/// @brief Write the open knot vectors and spline degree attribute into an
///        already-open HDF5 file, making wavefunction files self-describing.
/// @param file  Open HDF5 file handle.
/// @param iga   IGA context containing the axis knot vectors.
/// @param dim   Spatial dimension.
/// @param pdeg  Spline degree.
/// @return 0 on success, <0 on error.
static herr_t TDSEZWriteKnotsToH5(hid_t file, IGA iga, PetscInt dim, PetscInt pdeg)
{
    const char *axis_names[] = {"knots_x", "knots_y", "knots_z"};
    for (PetscInt d = 0; d < dim; ++d) {
        IGAAxis axis;
        IGAGetAxis(iga, d, &axis);
        PetscInt   nknots;
        PetscReal *knotvals;
        IGAAxisGetKnots(axis, &nknots, &knotvals);

        PetscInt start_mult = 1;
        for (PetscInt i = 1; i < nknots; ++i) { if (knotvals[i] == knotvals[i-1]) ++start_mult; else break; }
        PetscInt end_mult = 1;
        for (PetscInt i = nknots-2; i >= 0; --i) { if (knotvals[i] == knotvals[i+1]) ++end_mult; else break; }

        // Save the EXACT PetIGA knot vector (compact form): PetIGA keeps the
        // right end at multiplicity p (not p+1) and drops the final trailing
        // Lmax. Its length is nknots = nfuncs + p, so a reader recovers the
        // standard open vector (length nfuncs + p + 1) by appending one final
        // knot = last value (= Lmax). Trimming further to length nfuncs would
        // drop the domain endpoint and break basis/dof alignment, so keep as-is.
        std::vector<PetscReal> kv(knotvals, knotvals + nknots);

        hsize_t dims[1] = {(hsize_t)kv.size()};
        hid_t space = H5Screate_simple(1, dims, NULL);
        if (space < 0) return -1;
        if (H5Lexists(file, axis_names[d], H5P_DEFAULT) > 0)   // restarted run: same grid, replace
            H5Ldelete(file, axis_names[d], H5P_DEFAULT);
        hid_t dset = H5Dcreate2(file, axis_names[d], H5T_NATIVE_DOUBLE, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dset < 0) { H5Sclose(space); return -1; }
        if (H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, kv.data()) < 0) {
            H5Dclose(dset); H5Sclose(space); return -1;
        }

        // mult_start_end = actual endpoint multiplicities (PetIGA may keep the
        // right end at multiplicity p).
        hsize_t mdims[1] = {2};
        hid_t mspace = H5Screate_simple(1, mdims, NULL);
        hid_t attr = H5Acreate2(dset, "mult_start_end", H5T_NATIVE_INT, mspace,
                               H5P_DEFAULT, H5P_DEFAULT);
        if (attr < 0) { H5Sclose(mspace); H5Dclose(dset); H5Sclose(space); return -1; }
        PetscInt mult[2] = {start_mult, end_mult};
        if (H5Awrite(attr, H5T_NATIVE_INT, mult) < 0) {
            H5Aclose(attr); H5Sclose(mspace); H5Dclose(dset); H5Sclose(space); return -1;
        }
        H5Aclose(attr);

        // SplineDegree p — required to rebuild the basis from the knot vector.
        hid_t pspace = H5Screate(H5S_SCALAR);
        hid_t pattr  = H5Acreate2(dset, "SplineDegree", H5T_NATIVE_INT, pspace,
                                 H5P_DEFAULT, H5P_DEFAULT);
        if (pattr < 0) { H5Sclose(pspace); H5Dclose(dset); H5Sclose(space); return -1; }
        PetscInt pval = pdeg;
        if (H5Awrite(pattr, H5T_NATIVE_INT, &pval) < 0) {
            H5Aclose(pattr); H5Sclose(pspace); H5Dclose(dset); H5Sclose(space); return -1;
        }
        H5Aclose(pattr); H5Sclose(pspace);

        // nfuncs = nknots - p: exact size of each wavefunction coefficient vector
        // and the basis-function count a reader must use (PetIGA's knot vector
        // keeps the right end at multiplicity p, so the generic nknots-p-1 is wrong).
        {
            PetscInt nfuncs = (PetscInt)kv.size() - pdeg;
            hid_t nspace = H5Screate(H5S_SCALAR);
            hid_t nattr  = H5Acreate2(dset, "nfuncs", H5T_NATIVE_INT, nspace,
                                     H5P_DEFAULT, H5P_DEFAULT);
            if (nattr < 0) { H5Sclose(nspace); H5Dclose(dset); H5Sclose(space); return -1; }
            if (H5Awrite(nattr, H5T_NATIVE_INT, &nfuncs) < 0) {
                H5Aclose(nattr); H5Sclose(nspace); H5Dclose(dset); H5Sclose(space); return -1;
            }
            H5Aclose(nattr); H5Sclose(nspace);
        }

        H5Sclose(mspace); H5Dclose(dset); H5Sclose(space);
    }
    return 0;
}




// ---- TDSEZManager constructor / destructor (out-of-line) ----

/// @brief Construct the TDSEZManager. Aliases matrices from the core and
///        assembler, allocates work vectors and ring buffers for time-series
///        output, creates HDF5 viewers (WFS, TIME, AC), ensures the td/
///        output directory exists, and calls PolarizationSelector().
/// @param core      Pointer to the TDSEZCore holding IGA, H, M, eigenstates.
/// @param assembler Pointer to the TDSEZAssembler holding dipole/velocity/
///                  gradient/CAP/Lz operators.
        TDSEZManager::TDSEZManager(TDSEZCore* core, TDSEZAssembler* assembler) :
        core_(core), assembler_(assembler), m_M(core_->M),
        m_H(core_->H),  m_K(assembler_->K), m_V(assembler_->V), m_Dx(assembler_->Dx), m_Dy(assembler_->Dy),
        m_Dz(assembler_->Dz), m_VelX(assembler_->VelX), m_VelY(assembler_->VelY),
        m_VelZ(assembler_->VelZ), m_dVdx(assembler_->dVdx), m_dVdy(assembler_->dVdy),
        m_dVdz(assembler_->dVdz), m_Md(assembler_->Md), m_CAP(assembler_->CAP), m_Lz(assembler_->Lz),
        m_G(assembler_->G), charge(TDSEZParser::Q)
        {
            // Time stepping folder data
            int mpi_rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);
            rank = mpi_rank;

            // Ensure the output directory `td/` exists. Every HDF5 file written
            // during propagation (<name>_observables, _wfs, _surff, _ts)
            // is created inside `td/`. The previous code relied on `td/` being
            // pre-existing (left over from a prior run), so a FRESH run in a
            // clean directory crashed at H5Fcreate() -> Status -1 (the WFSViewer
            // open is the first H5Fcreate into td/). Create it once, on rank 0,
            // then barrier so no rank proceeds to H5Fcreate before it exists.
            if (rank == 0) {
                std::error_code td_ec;
                std::filesystem::create_directories("td", td_ec);
            }
            MPI_Barrier(PETSC_COMM_WORLD);

            // Resource Allocation (No MPI calls here)
            // numVecs must cover the LARGEST monitor VecMDot batch. The 3D
            // monitor indexes tmpVec up to nBatch-1; for xyz polarization that
            // is 14 (M,H,Md,K,V + Dx,dVdx,VelX + Dy,dVdy,VelY + Dz,dVdz,VelZ).
            // 12 was too small and caused a SEGV (out-of-range tmpVec write) on
            // the first monitor call for multi-axis polarization. 16 leaves
            // headroom for the optional Lz term and future operators.
            numVecs = 16, numIFuncVec = 4;
            VecDuplicateVecs(core_->initialPsi,numVecs, &tmpVec);
            VecDuplicateVecs(core_->initialPsi,numIFuncVec, &IFuncVec);
            for (int i = 0; i < numVecs; ++i) {
                VecSet(tmpVec[i], 0.0);
                if (i < numIFuncVec)
                {
                    VecSet(IFuncVec[i], 0.0);
                    VecSetFromOptions(IFuncVec[i]);
                }
                VecSetFromOptions(tmpVec[i]);
            }

            // Init population and popdots
            NPOP = core_->boundstates.size();
            population.resize(NPOP, 0.0);
            popDots.resize(NPOP, 0.0);

            // observed axes: those with a dipole operator (with -tdsez_mat_free
            // the uncoupled dipoles are factorised, not assembled)
            const bool mf = TDSEZMatrixFree();
            hasX = (m_Dx != PETSC_NULLPTR || (mf && TDSEZParser::Observed(0))) ? PETSC_TRUE : PETSC_FALSE;
            hasY = (m_Dy != PETSC_NULLPTR || (mf && TDSEZParser::Observed(1))) ? PETSC_TRUE : PETSC_FALSE;
            hasZ = (m_Dz != PETSC_NULLPTR || (mf && TDSEZParser::Observed(2))) ? PETSC_TRUE : PETSC_FALSE;

            // Determine size and safety margin
            PetscInt expectedSteps = static_cast<PetscInt>(std::ceil(TDSEZParser::FinalTime / TDSEZParser::TimeStep)) + 500;           
            PetscInt bufSize = (expectedSteps > FLUSH_INTERVAL) ? FLUSH_INTERVAL : expectedSteps;
            this->maxAllocatedSteps = bufSize;
            this->recordedSteps     = 0;
        
            // Dynamic Width Logic
            PetscInt d = TDSEZParser::Dimension;
            this->dipWidth = (d == 1) ? 4 : (d == 2) ? 7 : 10;
            this->popWidth = NPOP + 1;
            this->energyWidth = 7; 

            // Vec sizes
            PetscInt currSize = (d == 3) ? 15 : 11;

            // Ring buffers (allocated on all ranks; only rank 0 actually writes
            // into them inside the monitors, but the original allocation on every
            // rank is preserved to avoid any rank/ownership mismatch).
            // THE SQUEEZE: Safe, zeroed allocation
            PetscCalloc1(expectedSteps * dipWidth,    &dipBuffer);
            PetscCalloc1(expectedSteps * popWidth,    &popBuffer);
            PetscCalloc1(expectedSteps * energyWidth, &energyBuffer);
            PetscCalloc1(expectedSteps * currSize, &currBuffer);
            PetscCalloc1(expectedSteps * 3, &acBuffer);

            // Setup the Single Unified TIMEViewer (stride=0 = disabled)
            if (TDSEZParser::OutputStrideTS > 0) {
                std::string TIMEFILE = TDSEZPaths::TD(core_->inputFile, "ts.h5");
                if (rank == 0) {
                    std::string absPath = std::filesystem::absolute(TIMEFILE).string();
                    hid_t tf = H5Fcreate(absPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                    if (tf >= 0) H5Fclose(tf);
                }
                MPI_Barrier(PETSC_COMM_WORLD);
                PetscViewerHDF5Open(PETSC_COMM_WORLD, TIMEFILE.c_str(), FILE_MODE_UPDATE, &TIMEViewer);
                PetscViewerHDF5SetCollective(TIMEViewer, PETSC_TRUE);
                if (TDSEZParser::HDF5Compress) PetscViewerHDF5SetCompress(TIMEViewer, PETSC_TRUE);
                PetscViewerHDF5PushTimestepping(TIMEViewer);
            }

            // Setup the WFSViewer for wavefunction snapshots (inside td/)
            std::string wfsBase = std::filesystem::path(core_->inputFile).filename().string();
            std::string WFSFILE = TDSEZPaths::TD(wfsBase, "wfs.h5");
            // Create the (empty) file on rank 0, then a collective MPI_Barrier,
            // then open it COLLECTIVELY on all ranks. The knot vectors are NOT
            // embedded here: doing a serial H5Fopen/H5Dwrite/H5Fclose on this
            // same file before the collective viewer opens it desyncs HDF5's
            // metadata cache and makes the collective VecView(psi) silently
            // no-op (wavefunction dataset never written). Knots are embedded
            // instead in the destructor, AFTER WFSViewer is destroyed, when no
            // collective handle is live (see ~TDSEZManager).
            // Restart: keep the snapshots of the first run (the restarted run
            // overwrites the ones from the checkpoint step on).
            if (rank == 0 && !(TDSEZParser::Restart && std::filesystem::exists(WFSFILE))) {
                std::string absPath = std::filesystem::absolute(WFSFILE).string();
                hid_t tf = H5Fcreate(absPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                if (tf >= 0) H5Fclose(tf);
            }
            MPI_Barrier(PETSC_COMM_WORLD);
            PetscViewerHDF5Open(PETSC_COMM_WORLD, WFSFILE.c_str(), FILE_MODE_UPDATE, &WFSViewer);
            PetscViewerHDF5SetCollective(WFSViewer, PETSC_TRUE);
            if (TDSEZParser::HDF5Compress) PetscViewerHDF5SetCompress(WFSViewer, PETSC_TRUE);
            PetscViewerHDF5PushTimestepping(WFSViewer);

            // The force matrices dV/dx_d feed only the operator-form
            // acceleration; with the discrete form they are never read.
            if (TDSEZDiscreteAccelerationActive()) {
                MatDestroy(&assembler_->dVdx); MatDestroy(&assembler_->dVdy); MatDestroy(&assembler_->dVdz);
                m_dVdx = m_dVdy = m_dVdz = PETSC_NULLPTR;
            }

            // Final Housekeeping
            inputFile = core_->inputFile;
            boundstates = core_->boundstates;
            energies    = core_->energies;
                        
            // Reset ranks 
            // MPI_Barrier(PETSC_COMM_WORLD);
            PolarizationSelector();
        }

/// @brief Destroy the TDSEZManager. Frees work vectors, ring buffers,
///        and HDF5 viewers. The operator matrices are owned by the core and the assembler;
///        aliased matrices (CAP, Md, Lz, etc.) are owned by the assembler.
///        Embeds knot vectors into the WFS file after the collective viewer
///        is closed (rank 0 only).
        TDSEZManager::~TDSEZManager() 
        {
            // Destroy vectors first
            if (tmpVec)      VecDestroyVecs(numVecs, &tmpVec);
            if (IFuncVec)    VecDestroyVecs(numIFuncVec, &IFuncVec);

            // Destroy matrices
            // NOTE: Lz (and Md, CAP) are ALIASED from assembler_->* (see ctor
            // initializer list) — the TDSEZAssembler destructor already owns
            // and destroys them. Destroying them here too is a double-free
            // (SEGV at teardown). Only destroy matrices the manager owns.
            if (kspMass)       KSPDestroy(&kspMass);
            delete tensorOps_;
            delete quadObs_;
            TDSEZForceCoeffStore().Release();
            for (Vec &v : accelWork) if (v) VecDestroy(&v);
            for (Vec &v : termPsi_) if (v) VecDestroy(&v);

            // Free the buffers
            if (dipBuffer)    PetscFree(dipBuffer);
            if (popBuffer)    PetscFree(popBuffer);
            if (energyBuffer) PetscFree(energyBuffer);
            if (currBuffer)   PetscFree(currBuffer);
            if (acBuffer)     PetscFree(acBuffer);
            if (m_psi0)        VecDestroy(&m_psi0);


            //  Destroy viewers 
            if (WFSViewer){
                    PetscViewerHDF5PopTimestepping(WFSViewer);
                    PetscViewerDestroy(&WFSViewer);
                }
            // Embed knot vectors into the wfs file NOW (after the collective
            // viewer is fully closed on all ranks) so the file is self-
            // describing. Must be rank-0-only serial H5 (no collective handle is
            // live anymore, so no metadata desync / VecView no-op risk).
            if (rank == 0 && core_) {
                std::string wfsBase = std::filesystem::path(core_->inputFile).filename().string();
                std::string WFSFILE = TDSEZPaths::TD(wfsBase, "wfs.h5");
                std::string absPath = std::filesystem::absolute(WFSFILE).string();
                hid_t wf = H5Fopen(absPath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                if (wf >= 0) {
                    if (TDSEZWriteKnotsToH5(wf, core_->iga, TDSEZParser::Dimension,
                                             TDSEZParser::SplineDegree) < 0) {
                        TDSEZInfo(PETSC_COMM_SELF).warn("HDF5 failed to embed knots into %s", absPath.c_str());
                    }
                    H5Fclose(wf);
                }
            }
            if (TIMEViewer){
                    PetscViewerHDF5PopTimestepping(TIMEViewer);
                    PetscViewerDestroy(&TIMEViewer);
            }

            // Safety: if the evolution file handle is still open (e.g. propagation
            // skipped or an early return), flush and close it so the file is valid.
            if (h5File >= 0) {
                H5Fflush(h5File, H5F_SCOPE_GLOBAL);
                H5Fclose(h5File);
                h5File = -1;
            }
                    
        }


/// @brief Whether a(t) is evaluated by DiscreteAcceleration().
/// @details "auto" selects the discrete form only when the mass expression
///          actually depends on x, y or z (and MassIsConstant is not set), so
///          constant-mass inputs keep the operator form and reproduce
///          earlier results.
PetscBool TDSEZManager::UseDiscreteAcceleration() const
{
    return TDSEZDiscreteAccelerationActive();
}


/// @brief Free-function form of TDSEZManager::UseDiscreteAcceleration(), usable
///        from the assembly kernels.
PetscBool TDSEZDiscreteAccelerationActive()
{
    const std::string &m = TDSEZParser::AccelerationMethod;
    if (m == "discrete") return PETSC_TRUE;
    if (m == "operator") return PETSC_FALSE;
    if (TDSEZParser::NCouplingTerms() > 0) return PETSC_TRUE;          // "auto": the forces of the terms are not assembled
    if (TDSEZParser::MassIsConstant) return PETSC_FALSE;               // "auto"
    // GetUsedVar() re-parses the expression and this is queried from the
    // assembly kernels at every quadrature point, so cache the answer per
    // mass expression.
    static std::string cachedExpr;
    static PetscBool   cachedVaries = PETSC_TRUE, cacheValid = PETSC_FALSE;
    const std::string &expr = TDSEZParser::MassExpr.GetExpr();
    if (!cacheValid || expr != cachedExpr) {
        try {
            cachedVaries = TDSEZParser::MassExpr.GetUsedVar().empty() ? PETSC_FALSE : PETSC_TRUE;
        } catch (const mu::Parser::exception_type &) {
            cachedVaries = PETSC_TRUE;
        }
        cachedExpr = expr;
        cacheValid = PETSC_TRUE;
    }
    return cachedVaries;
}


/// @brief Monitor products tmpVec[0..4] = (M, H, Md, K, V) psi.
/// @details On the first call the full set of products is formed and two
///          identities are checked on psi: H psi = K psi + V psi (holds when no
///          CAP was merged into H) and Md psi = c M psi (constant mass). Each
///          shortcut is used afterwards only if it holds to 1e-12 relative, so
///          the written observables are unchanged up to rounding; K and Md are
///          then freed.
PetscErrorCode TDSEZManager::MonitorProducts(Vec psi, PetscBool needKVMd, PetscBool needM, PetscBool needH)
{
    PetscFunctionBeginUser;
    if (!needKVMd) {                        // only what the requested outputs read
        if (needM) PetscCall(ApplyOp(TDSEZTensorOps::Mass, m_M, psi, tmpVec[0]));
        if (needH) PetscCall(MatMult(m_H, psi, tmpVec[1]));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(ApplyOp(TDSEZTensorOps::Mass, m_M, psi, tmpVec[0]));
    if (!monitorChecked_ && matFree_) {
        // nothing assembled to check against: set up the quadrature values
        // and continue with the steady-state products below
        monitorChecked_ = PETSC_TRUE;
        PetscCall(SetupQuadObs(psi));
    } else if (!monitorChecked_) {
        PetscCall(MatMult(m_V, psi, tmpVec[4]));
        PetscCall(MatMult(m_K, psi, tmpVec[3]));
        PetscCall(MatMult(m_H,  psi, tmpVec[1]));
        PetscCall(MatMult(m_Md, psi, tmpVec[2]));
        Vec r;
        PetscReal nr, nref;
        PetscCall(VecDuplicate(psi, &r));
        // H psi - K psi - V psi
        PetscCall(VecWAXPY(r, 1.0, tmpVec[3], tmpVec[4]));
        PetscCall(VecAYPX(r, -1.0, tmpVec[1]));
        PetscCall(VecNorm(r, NORM_2, &nr));
        PetscCall(VecNorm(tmpVec[1], NORM_2, &nref));
        hFromKV_ = (!TDSEZParser::EnableCAP && nr <= 1e-12 * PetscMax(nref, (PetscReal)1e-300)) ? PETSC_TRUE : PETSC_FALSE;
        // Md psi - c M psi with c = <psi|Md|psi>/<psi|M|psi>
        PetscBool massConst = TDSEZParser::MassIsConstant;
        try { if (TDSEZParser::MassExpr.GetUsedVar().empty()) massConst = PETSC_TRUE; }
        catch (const mu::Parser::exception_type &) {}
        invMassConst_ = 0.0;
        if (massConst) {
            PetscScalar a, b;
            PetscCall(VecDot(tmpVec[2], psi, &a));
            PetscCall(VecDot(tmpVec[0], psi, &b));
            const PetscScalar c = (PetscAbsScalar(b) > 0.0) ? a / b : 0.0;
            PetscCall(VecWAXPY(r, -c, tmpVec[0], tmpVec[2]));
            PetscCall(VecNorm(r, NORM_2, &nr));
            PetscCall(VecNorm(tmpVec[2], NORM_2, &nref));
            if (c != 0.0 && nr <= 1e-12 * PetscMax(nref, (PetscReal)1e-300)) invMassConst_ = c;
        }
        PetscCall(VecDestroy(&r));
        monitorChecked_ = PETSC_TRUE;
        PetscCall(SetupTensorOps(psi));
        PetscCall(SetupQuadObs(psi));
        // Operators the shortcuts make redundant are freed: K (K psi = H psi -
        // V psi) and Md (Md = M/m). Each is a full matrix (memory-bound in 3D).
        if (m_K && ((hFromKV_ && !qPot_) || qKin_ || (tensorOps_ && tensorOps_->Usable(TDSEZTensorOps::Kin)))) {
            PetscCall(MatDestroy(&assembler_->K)); m_K = PETSC_NULLPTR;
        }
        if (m_Md && (invMassConst_ != 0.0 || qInvM_)) { PetscCall(MatDestroy(&assembler_->Md)); m_Md = PETSC_NULLPTR; }
        PetscCall(PetscInfo(NULL, "Monitor shortcuts: H=K+V %s, constant 1/m %s\n",
                            hFromKV_ ? "on" : "off", invMassConst_ != 0.0 ? "on" : "off"));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    // Steady state. Products replaced by quadrature values are zeroed; the
    // monitors overwrite their expectation values (OverrideExpectations).
    if (quadObs_) PetscCall(quadObs_->Expect(psi, qv_));
    if (qPot_) PetscCall(VecZeroEntries(tmpVec[4]));
    else       PetscCall(MatMult(m_V, psi, tmpVec[4]));
    const bool tensorK = tensorOps_ && tensorOps_->Usable(TDSEZTensorOps::Kin);
    if (qH_) {
        // operator acceleration: H psi is not needed, <H> = <K> + <V> (+ <CAP>)
        if (tensorK) PetscCall(ApplyOp(TDSEZTensorOps::Kin, m_K, psi, tmpVec[3]));
        else         PetscCall(VecZeroEntries(tmpVec[3]));
        PetscCall(VecZeroEntries(tmpVec[1]));
    } else if (hFromKV_ && tensorK && !qPot_) {
        // no CAP: H psi = K psi + V psi with the factorised K (one sparse product less)
        PetscCall(ApplyOp(TDSEZTensorOps::Kin, m_K, psi, tmpVec[3]));
        PetscCall(VecWAXPY(tmpVec[1], 1.0, tmpVec[3], tmpVec[4]));
    } else {
        PetscCall(MatMult(m_H, psi, tmpVec[1]));
        if (m_K)                   PetscCall(MatMult(m_K, psi, tmpVec[3]));
        else if (tensorK)          PetscCall(ApplyOp(TDSEZTensorOps::Kin, m_K, psi, tmpVec[3]));
        else if (hFromKV_ && !qPot_) PetscCall(VecWAXPY(tmpVec[3], -1.0, tmpVec[4], tmpVec[1]));   // K psi = H psi - V psi
        else                       PetscCall(VecZeroEntries(tmpVec[3]));                         // <K> from quadrature
    }
    if (invMassConst_ != 0.0) {
        PetscCall(VecCopy(tmpVec[0], tmpVec[2]));
        PetscCall(VecScale(tmpVec[2], invMassConst_));
    } else if (qInvM_) {
        PetscCall(VecZeroEntries(tmpVec[2]));
    } else {
        PetscCall(MatMult(m_Md, psi, tmpVec[2]));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode TDSEZManager::SetupMatrixFree(Vec templ)
{
    PetscFunctionBeginUser;
    if (!TDSEZMatrixFree() || tensorOps_) PetscFunctionReturn(PETSC_SUCCESS);
    matFree_ = true;
    const PetscReal invMass = TDSEZParser::MassIsConstant ? 1.0 / TDSEZParser::MassDist(0.0, 0.0, 0.0) : 0.0;
    tensorOps_ = new TDSEZTensorOps;
    PetscCall(tensorOps_->Setup(core_->iga, templ, invMass));
    PetscCheck(tensorOps_->Enable(TDSEZTensorOps::Mass), PETSC_COMM_WORLD, PETSC_ERR_SUP,
               "-tdsez_mat_free: the factorised operators are not available for this vector type or geometry");
    const char *names[] = {"M", "Dx", "Dy", "Dz", "VelX", "VelY", "VelZ", "K"};
    std::string used = "M";
    for (int o = TDSEZTensorOps::DipX; o < TDSEZTensorOps::NOps; ++o)
        if (tensorOps_->Enable((TDSEZTensorOps::Op)o)) used += std::string(", ") + names[o];
    hFromKV_ = TDSEZParser::EnableCAP ? PETSC_FALSE : PETSC_TRUE;           // H = K + V by construction
    invMassConst_ = invMass;                                                 // Md = M/m for a constant mass
    TDSEZInfo().kv("Factorised operators", "%s (matrix-free)", used.c_str());
    PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode TDSEZManager::ApplyOp(TDSEZTensorOps::Op op, Mat A, Vec x, Vec y)
{
    PetscFunctionBeginUser;
    if (tensorOps_ && tensorOps_->Usable(op)) PetscCall(tensorOps_->Apply(op, x, y));
    else                                      PetscCall(MatMult(A, x, y));
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief Factorised monitor operators (see TDSEZTensorOps): each separable
///        operator is checked against its assembled matrix on psi and used only
///        if they agree to 1e-10 relative. Matrices then read by nothing else
///        are freed: the velocities, the dipoles of uncoupled axes (the
///        propagator keeps the coupled ones) and K. -tdsez_tensor_ops 0 keeps
///        the assembled products.
PetscErrorCode TDSEZManager::SetupTensorOps(Vec psi)
{
    PetscFunctionBeginUser;
    PetscBool on = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-tdsez_tensor_ops", &on, NULL));
    if (!on || TDSEZParser::Dimension < 2 || tensorOps_) PetscFunctionReturn(PETSC_SUCCESS);
    // Dirichlet walls replace the wall rows of the assembled matrices by the
    // identity; the Kronecker form has the free rows. A state that is small at
    // the walls cannot tell them apart, so the walls are excluded structurally.
    std::string bt = TDSEZParser::BoundaryType;
    std::transform(bt.begin(), bt.end(), bt.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (bt == "dirichlet" || bt == "wall") PetscFunctionReturn(PETSC_SUCCESS);
    tensorOps_ = new TDSEZTensorOps;
    const PetscReal invMass = TDSEZParser::MassIsConstant ? 1.0 / TDSEZParser::MassDist(0.0, 0.0, 0.0) : 0.0;
    PetscCall(tensorOps_->Setup(core_->iga, psi, invMass));
    struct { TDSEZTensorOps::Op op; Mat A; const char *name; } ops[] = {
        {TDSEZTensorOps::Mass, m_M, "M"}, {TDSEZTensorOps::DipX, m_Dx, "Dx"}, {TDSEZTensorOps::DipY, m_Dy, "Dy"},
        {TDSEZTensorOps::DipZ, m_Dz, "Dz"}, {TDSEZTensorOps::VelX, m_VelX, "VelX"}, {TDSEZTensorOps::VelY, m_VelY, "VelY"},
        {TDSEZTensorOps::VelZ, m_VelZ, "VelZ"}, {TDSEZTensorOps::Kin, m_K, "K"}};
    std::string used;
    PetscReal worst = 0;
    for (auto &o : ops) {
        PetscReal err;
        PetscCall(tensorOps_->Validate(o.op, o.A, psi, &err));
        if (tensorOps_->Usable(o.op)) { used += std::string(used.empty() ? "" : ", ") + o.name; worst = PetscMax(worst, err); }
    }
    if (!used.empty()) {
        TDSEZInfo().kv("Factorised operators", "%s", used.c_str());
        TDSEZInfo().kv("  max relative difference", "%.1e", (double)worst);
    }
    Mat *vel[3] = {&assembler_->VelX, &assembler_->VelY, &assembler_->VelZ};
    Mat *dip[3] = {&assembler_->Dx, &assembler_->Dy, &assembler_->Dz};
    Mat *mv[3] = {&m_VelX, &m_VelY, &m_VelZ}, *md[3] = {&m_Dx, &m_Dy, &m_Dz};
    for (int a = 0; a < 3; ++a) {
        if (tensorOps_->Usable((TDSEZTensorOps::Op)(TDSEZTensorOps::VelX + a))) { PetscCall(MatDestroy(vel[a])); *mv[a] = PETSC_NULLPTR; }
        if (tensorOps_->Usable((TDSEZTensorOps::Op)(TDSEZTensorOps::DipX + a)) && !TDSEZParser::Coupled(a)) {
            PetscCall(MatDestroy(dip[a])); *md[a] = PETSC_NULLPTR;
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief Quadrature expectation values (see TDSEZQuadObs): built at the first
///        monitor call, each value checked against its assembled matrix on psi
///        (|quad - matrix| <= 1e-10 |A psi| |psi|), and only then used; the
///        matrices then read by nothing else are freed. Dirichlet walls keep the
///        matrices; -tdsez_quad_obs 0 switches this off. Called from
///        MonitorProducts while tmpVec[1..4] hold H, Md, K, V psi.
PetscErrorCode TDSEZManager::SetupQuadObs(Vec psi)
{
    PetscFunctionBeginUser;
    if (quadObs_) PetscFunctionReturn(PETSC_SUCCESS);
    if (!TDSEZQuadObsEnabled()) { TDSEZForceCoeffStore().Release(); PetscFunctionReturn(PETSC_SUCCESS); }

    const bool op = !TDSEZDiscreteAccelerationActive();
    const bool tensorK = tensorOps_ && tensorOps_->Usable(TDSEZTensorOps::Kin);
    if (matFree_) {
        // -tdsez_mat_free: every quantity the monitors read comes from here
        unsigned what = TDSEZQuadObs::Pot, forceAxes = 0;
        if (invMassConst_ == 0.0) what |= TDSEZQuadObs::InvMass;
        if (!tensorK) what |= TDSEZQuadObs::Kin;
        if (op && TDSEZParser::EnableCAP) what |= TDSEZQuadObs::Cap;
        for (int a = 0; a < (int)TDSEZParser::Dimension; ++a)
            if (op && TDSEZParser::Observed(a)) { what |= TDSEZQuadObs::Force; forceAxes |= 1u << a; }
        quadObs_ = new TDSEZQuadObs;
        PetscCall(quadObs_->Setup(core_->iga, what, op && MassVaries(), forceAxes));
        TDSEZForceCoeffStore().Release();
        PetscCheck(quadObs_->Ready(), PETSC_COMM_WORLD, PETSC_ERR_SUP,
                   "-tdsez_mat_free: the quadrature expectation values are not available for this run");
        PetscCall(quadObs_->Expect(psi, qv_));
        qPot_  = true;
        qInvM_ = (what & TDSEZQuadObs::InvMass) != 0;
        qKin_  = (what & TDSEZQuadObs::Kin) != 0;
        for (int a = 0; a < 3; ++a) qForce_[a] = (forceAxes >> a) & 1u;
        qH_   = op && (!TDSEZParser::EnableCAP || (what & TDSEZQuadObs::Cap));
        qCap_ = qH_ && (what & TDSEZQuadObs::Cap);
        std::string used = "V";
        if (qInvM_) used += ", 1/m";
        if (qKin_) used += ", K";
        for (int a = 0; a < 3; ++a) if (qForce_[a]) used += std::string(", dV/d") + "xyz"[a];
        if (qH_) used += ", H";
        TDSEZInfo().kv("Quadrature-point expectations", "%s (matrix-free)", used.c_str());
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    const Mat dV[3] = {m_dVdx, m_dVdy, m_dVdz}, Vl[3] = {m_VelX, m_VelY, m_VelZ};
    unsigned what = 0;
    if (m_V) what |= TDSEZQuadObs::Pot;
    if (m_Md && invMassConst_ == 0.0) what |= TDSEZQuadObs::InvMass;
    if (m_K && !tensorK) what |= TDSEZQuadObs::Kin;
    if (op && TDSEZParser::EnableCAP) what |= TDSEZQuadObs::Cap;
    for (int a = 0; a < 3; ++a) {
        if (op && dV[a]) what |= TDSEZQuadObs::Force;
        const bool tensorV = tensorOps_ && tensorOps_->Usable((TDSEZTensorOps::Op)(TDSEZTensorOps::VelX + a));
        if (Vl[a] && !tensorV) what |= TDSEZQuadObs::Vel;
    }
    if (!what) { TDSEZForceCoeffStore().Release(); PetscFunctionReturn(PETSC_SUCCESS); }
    quadObs_ = new TDSEZQuadObs;
    unsigned forceAxes = 0;
    for (int a = 0; a < 3; ++a) if (op && dV[a]) forceAxes |= 1u << a;
    PetscCall(quadObs_->Setup(core_->iga, what, op && MassVaries(), forceAxes));
    TDSEZForceCoeffStore().Release();
    if (!quadObs_->Ready()) { delete quadObs_; quadObs_ = PETSC_NULLPTR; PetscFunctionReturn(PETSC_SUCCESS); }
    TDSEZQuadObs::Values v;
    PetscCall(quadObs_->Expect(psi, v));

    PetscReal npsi, worst = 0;
    PetscCall(VecNorm(psi, NORM_2, &npsi));
    Vec tmp;
    PetscCall(VecDuplicate(psi, &tmp));
    auto agree = [&](Vec Apsi, PetscScalar q, bool *flag) -> PetscErrorCode {
        PetscFunctionBeginUser;
        PetscScalar ref; PetscReal na;
        PetscCall(VecDot(Apsi, psi, &ref));                 // psi^H A psi
        PetscCall(VecNorm(Apsi, NORM_2, &na));
        const PetscReal err = PetscAbsScalar(q - ref) / PetscMax(na * npsi, (PetscReal)1e-300);
        *flag = err <= 1e-10;
        if (*flag) worst = PetscMax(worst, err);
        PetscFunctionReturn(PETSC_SUCCESS);
    };
    std::string used;
    auto note = [&](bool f, const char *n) { if (f) used += std::string(used.empty() ? "" : ", ") + n; };
    if (what & TDSEZQuadObs::Pot) { PetscCall(agree(tmpVec[4], v.pot, &qPot_)); note(qPot_, "V"); }
    if (what & TDSEZQuadObs::InvMass) { PetscCall(agree(tmpVec[2], v.invMass, &qInvM_)); note(qInvM_, "1/m"); }
    if (what & TDSEZQuadObs::Kin) { PetscCall(agree(tmpVec[3], v.kin, &qKin_)); note(qKin_, "K"); }
    for (int a = 0; a < 3; ++a) {
        if ((what & TDSEZQuadObs::Force) && dV[a]) {
            PetscCall(MatMult(dV[a], psi, tmp));
            PetscCall(agree(tmp, v.force[a], &qForce_[a]));
            note(qForce_[a], (std::string("dV/d") + "xyz"[a]).c_str());
        }
        const bool tensorV = tensorOps_ && tensorOps_->Usable((TDSEZTensorOps::Op)(TDSEZTensorOps::VelX + a));
        if ((what & TDSEZQuadObs::Vel) && Vl[a] && !tensorV) {
            PetscCall(MatMult(Vl[a], psi, tmp));
            PetscCall(agree(tmp, v.vel[a], &qVel_[a]));
            note(qVel_[a], (std::string("Vel") + "xyz"[a]).c_str());
        }
    }
    // operator acceleration: <H> = <K> + <V> (+ <CAP>) without the product with H
    if (op && qPot_ && (qKin_ || tensorK || m_K) && (!TDSEZParser::EnableCAP || (what & TDSEZQuadObs::Cap))) {
        PetscScalar kin;
        if (qKin_) kin = v.kin;
        else PetscCall(VecDot(tmpVec[3], psi, &kin));     // K or factorised K (equal, checked)
        const PetscScalar h = kin + v.pot + ((what & TDSEZQuadObs::Cap) ? v.cap : 0.0);
        bool ok;
        PetscCall(agree(tmpVec[1], h, &ok));
        qH_ = ok && (qKin_ || tensorK);
        qCap_ = qH_ && (what & TDSEZQuadObs::Cap);
        note(qH_, "H");
    }
    PetscCall(VecDestroy(&tmp));
    qv_ = v;                                   // this call's values (the monitor body reads them)
    if (!used.empty()) {
        TDSEZInfo().kv("Quadrature-point expectations", "%s", used.c_str());
        TDSEZInfo().kv("  max relative difference", "%.1e", (double)worst);
    }
    // free what nothing else reads
    if (qPot_) { PetscCall(MatDestroy(&assembler_->V)); m_V = PETSC_NULLPTR; }
    Mat *dVa[3] = {&assembler_->dVdx, &assembler_->dVdy, &assembler_->dVdz}, *mdV[3] = {&m_dVdx, &m_dVdy, &m_dVdz};
    Mat *Vla[3] = {&assembler_->VelX, &assembler_->VelY, &assembler_->VelZ}, *mVl[3] = {&m_VelX, &m_VelY, &m_VelZ};
    for (int a = 0; a < 3; ++a) {
        if (qForce_[a]) { PetscCall(MatDestroy(dVa[a])); *mdV[a] = PETSC_NULLPTR; }
        if (qVel_[a])   { PetscCall(MatDestroy(Vla[a])); *mVl[a] = PETSC_NULLPTR; }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}


void TDSEZManager::OverrideExpectations(PetscScalar res[]) const
{
    if (!quadObs_) return;
    if (qInvM_) res[2] = qv_.invMass;
    if (qKin_)  res[3] = qv_.kin;
    if (qPot_)  res[4] = qv_.pot;
    if (qH_)    res[1] = res[3] + res[4] + (qCap_ ? qv_.cap : 0.0);
}


/// @brief Whether the time-series buffers should be written now (see header).
bool TDSEZManager::FlushDue()
{
    if (recordedSteps <= 0) return false;
    // Buffer full (more steps than planned, e.g. with -ts_adapt): write now,
    // otherwise the ring buffer would overwrite rows not yet written.
    if (maxAllocatedSteps > 0 && recordedSteps - lastWrittenSteps >= maxAllocatedSteps) {
        PetscTime(&lastFlushTime_); return true;
    }
    if (recordedSteps % FLUSH_INTERVAL == 0) { PetscTime(&lastFlushTime_); return true; }
    const PetscInt stride = TDSEZParser::OutputStrideTS;
    if (stride > 0 && recordedSteps % stride == 0) {
        PetscLogDouble now;
        PetscTime(&now);
        if (now - lastFlushTime_ >= kMinFlushSeconds) { lastFlushTime_ = now; return true; }
    }
    return false;
}


/// @brief Dipole acceleration from the discrete equation of motion.
/// @details With i M c' = H_t c and H_t = H + E.D, the exact time derivative
///          of <v> = c^H Vel c is d<v>/dt = -2 Im( w^H Vel c ),
///          w = M^{-1} H_t c, for Hermitian Vel. Without a CAP this is used
///          as is. With a CAP, H_t is replaced by its Hermitian part, which
///          gives the discrete Ehrenfest acceleration
///          c^H i[H_h + E.D, Vel]_M c of the physical Hamiltonian.
/// @param[in]  t    Current time.
/// @param[in]  psi  Current coefficient vector.
/// @param[in]  Hpsi H psi if already available (may be NULL).
/// @param[out] acc  Acceleration components (x, y, z), scaled by the charge.
/// @return PETSc error code.
PetscErrorCode TDSEZManager::DiscreteAcceleration(PetscReal t, Vec psi, Vec Hpsi, PetscScalar acc[3])
{
    PetscFunctionBeginUser;
    if (!accelWork[0])
        for (Vec &v : accelWork) PetscCall(VecDuplicate(psi, &v));
    Vec r = accelWork[0], w = accelWork[1], s = accelWork[2];

    if (!kspMass) {
        // M is Hermitian positive definite: CG with block-Jacobi ILU(0) (a
        // symmetric preconditioner for a symmetric M) converges in about 16
        // iterations to 1e-14 relative and keeps no LU factor. A direct LU
        // (-accel_ksp_type preonly -accel_pc_type lu) has a large fill in 3D
        // and a full triangular solve at every monitor call.
        PC pc;
        PetscCall(KSPCreate(PetscObjectComm((PetscObject)m_M), &kspMass));
        PetscCall(KSPSetOptionsPrefix(kspMass, "accel_"));
        PetscCall(KSPSetOperators(kspMass, m_M, m_M));
        PetscCall(KSPSetType(kspMass, KSPCG));
        PetscCall(KSPSetTolerances(kspMass, 1e-14, PETSC_DEFAULT, PETSC_DEFAULT, 1000));
        // w = M^{-1}(H + E.D) psi changes little between monitor calls: start
        // from the previous one (accelWork[1]); the tolerance is unchanged.
        PetscCall(KSPSetInitialGuessNonzero(kspMass, PETSC_TRUE));
        PetscCall(KSPGetPC(kspMass, &pc));
        PetscCall(PCSetType(pc, PCBJACOBI));
        PetscCall(KSPSetFromOptions(kspMass));
        PetscCall(KSPSetUp(kspMass));
    }

    // r = (H_h + E.D) psi with H_h = (H + H^H)/2 the Hermitian part of H.
    // When a CAP is active it has been merged into H as an anti-Hermitian
    // term; dropping it gives the Ehrenfest acceleration of the physical
    // Hamiltonian instead of adding the absorber's -<{W, v}> drag.
    if (Hpsi) PetscCall(VecCopy(Hpsi, r));
    else      PetscCall(MatMult(m_H, psi, r));
    if (TDSEZParser::EnableCAP) {
        PetscCall(MatMultHermitianTranspose(m_H, psi, s));
        PetscCall(VecAXPBY(r, 0.5, 0.5, s));
    }
    const Mat       D[3]   = {m_Dx, m_Dy, m_Dz};
    const Mat       Vel[3] = {m_VelX, m_VelY, m_VelZ};
    const TDSEZTensorOps::Op dop[3] = {TDSEZTensorOps::DipX, TDSEZTensorOps::DipY, TDSEZTensorOps::DipZ};
    const TDSEZTensorOps::Op vop[3] = {TDSEZTensorOps::VelX, TDSEZTensorOps::VelY, TDSEZTensorOps::VelZ};
    const PetscBool on[3]  = {hasX, hasY, hasZ};             // observed axes
    // field term of H, s E(t), on the coupled axes only
    const PetscScalar E[3] = {TDSEZParser::Coupling(0, t), TDSEZParser::Coupling(1, t),
                              TDSEZParser::Coupling(2, t)};
    for (int a = 0; a < 3; ++a) {
        if (!TDSEZParser::Coupled(a) || !HasOp(dop[a], D[a])) continue;
        PetscCall(ApplyOp(dop[a], D[a], psi, s));
        PetscCall(VecAXPY(r, E[a], s));
    }
    if (!m_G.empty()) {                                // coupling terms, c_k(t) G_k psi
        PetscCall(FieldTermProducts(psi));
        for (size_t k = 0; k < m_G.size(); ++k)
            PetscCall(VecAXPY(r, TDSEZParser::CouplingTerms_[k].coeff(t), termPsi_[k]));
    }
    PetscCall(KSPSolve(kspMass, r, w));
    {
        KSPConvergedReason why;
        PetscCall(KSPGetConvergedReason(kspMass, &why));
        PetscCheck(why > 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
                   "discrete acceleration: the mass solve did not converge (%s)", KSPConvergedReasons[why]);
    }
    PetscScalar zq[3] = {0, 0, 0};
    if (qVel_[0] || qVel_[1] || qVel_[2]) PetscCall(quadObs_->VelocityBilinear(w, psi, zq));

    for (int a = 0; a < 3; ++a) {
        acc[a] = 0.0;
        PetscScalar z;
        if (HasOp(vop[a], Vel[a])) {
            if (!on[a]) continue;
            PetscCall(ApplyOp(vop[a], Vel[a], psi, s));
            PetscCall(VecDot(s, w, &z));             // z = w^H (Vel psi)
        } else if (on[a] && qVel_[a]) {
            z = zq[a];                               // w^H Vel psi from the quadrature points
        } else continue;
        acc[a] = -2.0 * charge * PetscImaginaryPart(z);
    }
    // Gauge = velocity: Vel = p/m is the canonical velocity; the kinetic one
    // is (p - qA)/m, so d<v>/dt gains -(q/m) dA/dt = (q/m) E.
    if (TDSEZParser::VelocityGauge) {
        const PetscReal m0 = TDSEZParser::MassDist(0.0, 0.0, 0.0);
        for (int a = 0; a < 3; ++a)
            if (on[a] && TDSEZParser::Coupled(a)) acc[a] += charge * charge / m0 * TDSEZParser::Field(a, t);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}


/// Dipoles of the coupled axes, then the FieldTerm matrices.
std::vector<Mat> TDSEZManager::TermMatrices() const
{
    std::vector<Mat> T;
    const Mat D[3] = {m_Dx, m_Dy, m_Dz};
    for (int a = 0; a < 3; ++a) if (TDSEZParser::Coupled(a) && D[a]) T.push_back(D[a]);
    for (Mat g : m_G) T.push_back(g);
    return T;
}

/// Coefficients of TermMatrices() at time t: s E_a(t), then s f_k(t).
void TDSEZManager::TermCoefficients(PetscReal t, std::vector<PetscScalar> &c) const
{
    c.clear();
    const Mat D[3] = {m_Dx, m_Dy, m_Dz};
    for (int a = 0; a < 3; ++a) if (TDSEZParser::Coupled(a) && D[a]) c.push_back(TDSEZParser::Coupling(a, t));
    for (size_t k = 0; k < m_G.size(); ++k) c.push_back(TDSEZParser::CouplingTerms_[k].coeff(t));
}

PetscErrorCode TDSEZManager::FieldTermProducts(Vec psi)
{
    PetscFunctionBeginUser;
    PetscObjectId id;
    PetscObjectState st;
    PetscCall(PetscObjectGetId((PetscObject)psi, &id));
    PetscCall(PetscObjectStateGet((PetscObject)psi, &st));
    if (id == termPsiId_ && st == termPsiState_) PetscFunctionReturn(PETSC_SUCCESS);
    if (termPsi_.size() != m_G.size()) {
        termPsi_.assign(m_G.size(), PETSC_NULLPTR);
        termExp_.assign(m_G.size(), 0.0);
        for (Vec &v : termPsi_) PetscCall(VecDuplicate(psi, &v));
    }
    for (size_t k = 0; k < m_G.size(); ++k) PetscCall(MatMult(m_G[k], psi, termPsi_[k]));
    if (!m_G.empty()) PetscCall(VecMDot(psi, (PetscInt)m_G.size(), termPsi_.data(), termExp_.data()));
    termPsiId_ = id;
    termPsiState_ = st;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZManager::FieldTermEnergy(PetscReal t, Vec psi, PetscScalar *e)
{
    PetscFunctionBeginUser;
    *e = 0.0;
    if (m_G.empty()) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(FieldTermProducts(psi));
    for (size_t k = 0; k < m_G.size(); ++k) *e += TDSEZParser::CouplingTerms_[k].coeff(t) * termExp_[k];
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZManager::TruncateObservables(PetscReal t)
{
    PetscFunctionBeginUser;
    if (rank == 0 && std::filesystem::exists(outputFilename)) {
        const std::string absPath = std::filesystem::absolute(outputFilename).string();
        hid_t file = H5Fopen(absPath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        PetscCheck(file >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Restart: could not open %s", absPath.c_str());
        const PetscReal tol = 1e-9 * PetscMax((PetscReal)1, TDSEZParser::TimeStep);
        PetscInt kept = -1;
        for (const char *name : {"dipoles", "populations", "energies", "currents", "autocorrelation"}) {
            hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
            if (dset < 0) continue;
            hid_t space = H5Dget_space(dset);
            hsize_t dims[2] = {0, 0};
            H5Sget_simple_extent_dims(space, dims, NULL);
            H5Sclose(space);
            std::vector<double> tcol(dims[0]);
            if (dims[0] > 0) {                         // column 0 is the time
                hid_t fs = H5Dget_space(dset);
                hsize_t start[2] = {0, 0}, count[2] = {dims[0], 1};
                H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, NULL, count, NULL);
                hid_t ms = H5Screate_simple(2, count, NULL);
                H5Dread(dset, H5T_NATIVE_DOUBLE, ms, fs, H5P_DEFAULT, tcol.data());
                H5Sclose(ms);
                H5Sclose(fs);
            }
            hsize_t n = 0;
            while (n < dims[0] && tcol[n] < t - tol) ++n;          // rows strictly before t
            hsize_t newDims[2] = {n, dims[1]};
            PetscCheck(H5Dset_extent(dset, newDims) >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Restart: could not truncate %s", name);
            H5Dclose(dset);
            kept = (PetscInt)n;
        }
        H5Fclose(file);
        if (kept >= 0) TDSEZInfo(PETSC_COMM_SELF).kv("Observables", "continued after %" PetscInt_FMT " rows (t < %.6g a.u.)", kept, (double)t);
    }
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    PetscFunctionReturn(PETSC_SUCCESS);
}
