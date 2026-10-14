#include "tdsez_internal.hpp"

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
    {
        // Observables are computed only on the coupled axes (their operators are
        // assembled only there); say so when some spatial axes are left out.
        std::string missing;
        for (PetscInt d = 0; d < TDSEZParser::Dimension; ++d)
            if (polType.find("xyz"[d]) == std::string::npos && polType != "all") missing += "xyz"[d];
        if (!missing.empty())
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "      Note: Polarization = %s; dipole, current and acceleration along %s are not "
                "computed (written as 0). Add the axis with a zero field (e.g. LaserY = 0) to record them.\n",
                polType.c_str(), missing.c_str()));
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
        // A quantity disabled via PhysicsOutput is written as zeros (stable
        // schema for downstream readers) while still skipping the per-step
        // compute that would have filled it. We reuse one zeroed scratch
        // buffer. CRITICAL: it must be sized to the FULL RING CAPACITY
        // (maxAllocatedSteps rows x ZW), NOT just nRows rows. The write path
        // indexes it as buf + startOff*cols where startOff = lastWrittenSteps %
        // maxAllocatedSteps (the ring position, which grows across flushes and
        // exceeds nRows after the first flush). Sizing it to only nRows rows
        // caused a heap-buffer-overflow (ASan) on every disabled dataset from
        // the 2nd flush onward. ZW is the widest dataset width in use.
        const PetscInt ZW = std::max<PetscInt>({dipWidth, popWidth, energyWidth,
                                      (TDSEZParser::Dimension == 3 ? 15 : 11), (PetscInt)3});
        const PetscInt zRows = (maxAllocatedSteps > 0) ? maxAllocatedSteps : nRows;
        std::vector<PetscScalar> zeros((size_t)zRows * (size_t)ZW, 0.0);
        const PetscScalar* dipSrc = TDSEZParser::out_dipole     ? dipBuffer    : zeros.data();
        const PetscScalar* popSrc = TDSEZParser::out_population ? popBuffer    : zeros.data();
        const PetscScalar* eneSrc = TDSEZParser::out_energy     ? energyBuffer : zeros.data();
        const PetscScalar* curSrc = TDSEZParser::out_current    ? currBuffer   : zeros.data();
        const PetscScalar* acSrc  = TDSEZParser::out_autocorr   ? acBuffer     : zeros.data();

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
        charge(TDSEZParser::Q)
        {
            // Time stepping folder data
            int mpi_rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);
            rank = mpi_rank;

            // Ensure the output directory `td/` exists. Every HDF5 file written
            // during propagation (TimeEvolutionData, wfs_*, ac_*, ts_*)
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

            // select polarization
            hasX = (m_Dx != PETSC_NULLPTR); 
            hasY = (m_Dy != PETSC_NULLPTR);
            hasZ = (m_Dz != PETSC_NULLPTR);

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
                std::string TIMEFILE = "ts_" + std::filesystem::path(core_->inputFile).filename().string() + ".h5";
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
            std::string WFSFILE = "td/wfs_" + wfsBase + ".h5";
            // Create the (empty) file on rank 0, then a collective MPI_Barrier,
            // then open it COLLECTIVELY on all ranks. The knot vectors are NOT
            // embedded here: doing a serial H5Fopen/H5Dwrite/H5Fclose on this
            // same file before the collective viewer opens it desyncs HDF5's
            // metadata cache and makes the collective VecView(psi) silently
            // no-op (wavefunction dataset never written). Knots are embedded
            // instead in the destructor, AFTER WFSViewer is destroyed, when no
            // collective handle is live (see ~TDSEZManager).
            if (rank == 0) {
                std::string absPath = std::filesystem::absolute(WFSFILE).string();
                hid_t tf = H5Fcreate(absPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                if (tf >= 0) H5Fclose(tf);
            }
            MPI_Barrier(PETSC_COMM_WORLD);
            PetscViewerHDF5Open(PETSC_COMM_WORLD, WFSFILE.c_str(), FILE_MODE_UPDATE, &WFSViewer);
            PetscViewerHDF5SetCollective(WFSViewer, PETSC_TRUE);
            if (TDSEZParser::HDF5Compress) PetscViewerHDF5SetCompress(WFSViewer, PETSC_TRUE);
            PetscViewerHDF5PushTimestepping(WFSViewer);

            // autocorrelation (stride=0 = disabled)
            if (TDSEZParser::OutputStrideAC > 0) {
                std::string acFile = "ac_" + wfsBase + ".h5";
                if (rank == 0) {
                    std::string absPath = std::filesystem::absolute(acFile).string();
                    hid_t tf = H5Fcreate(absPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                    if (tf >= 0) H5Fclose(tf);
                }
                MPI_Barrier(PETSC_COMM_WORLD);
                PetscViewerHDF5Open(PETSC_COMM_WORLD, acFile.c_str(), FILE_MODE_WRITE, &ACViewer);
                PetscViewerHDF5SetCollective(ACViewer, PETSC_TRUE);
                if (TDSEZParser::HDF5Compress) PetscViewerHDF5SetCompress(ACViewer, PETSC_TRUE);
                PetscViewerHDF5PushTimestepping(ACViewer);
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
///        and HDF5 viewers. Only destroys matrices the manager owns (Ht);
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
            if (m_Ht)          MatDestroy(&m_Ht);
            if (kspMass)       KSPDestroy(&kspMass);
            for (Vec &v : accelWork) if (v) VecDestroy(&v);

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
                std::string WFSFILE = "td/wfs_" + wfsBase + ".h5";
                std::string absPath = std::filesystem::absolute(WFSFILE).string();
                hid_t wf = H5Fopen(absPath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                if (wf >= 0) {
                    if (TDSEZWriteKnotsToH5(wf, core_->iga, TDSEZParser::Dimension,
                                             TDSEZParser::SplineDegree) < 0) {
                        PetscPrintf(PETSC_COMM_SELF,
                            "WARNING: HDF5 failed to embed knots into %s\n", absPath.c_str());
                    }
                    H5Fclose(wf);
                }
            }
            if (TIMEViewer){
                    PetscViewerHDF5PopTimestepping(TIMEViewer);
                    PetscViewerDestroy(&TIMEViewer);
            }

            if (ACViewer){
                    PetscViewerHDF5PopTimestepping(ACViewer);
                    PetscViewerDestroy(&ACViewer);
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
///          the written observables are unchanged up to rounding.
PetscErrorCode TDSEZManager::MonitorProducts(Vec psi, PetscBool needKVMd)
{
    PetscFunctionBeginUser;
    PetscCall(MatMult(m_M, psi, tmpVec[0]));
    if (!needKVMd) {
        PetscCall(MatMult(m_H, psi, tmpVec[1]));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(MatMult(m_K, psi, tmpVec[3]));
    PetscCall(MatMult(m_V, psi, tmpVec[4]));
    if (!monitorChecked_) {
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
        PetscCall(PetscInfo(NULL, "Monitor shortcuts: H=K+V %s, constant 1/m %s\n",
                            hFromKV_ ? "on" : "off", invMassConst_ != 0.0 ? "on" : "off"));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (hFromKV_) PetscCall(VecWAXPY(tmpVec[1], 1.0, tmpVec[3], tmpVec[4]));
    else          PetscCall(MatMult(m_H, psi, tmpVec[1]));
    if (invMassConst_ != 0.0) {
        PetscCall(VecCopy(tmpVec[0], tmpVec[2]));
        PetscCall(VecScale(tmpVec[2], invMassConst_));
    } else {
        PetscCall(MatMult(m_Md, psi, tmpVec[2]));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
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
        PC pc;
        PetscCall(KSPCreate(PetscObjectComm((PetscObject)m_M), &kspMass));
        PetscCall(KSPSetOptionsPrefix(kspMass, "accel_"));
        PetscCall(KSPSetOperators(kspMass, m_M, m_M));
        PetscCall(KSPSetType(kspMass, KSPPREONLY));
        PetscCall(KSPGetPC(kspMass, &pc));
        PetscCall(PCSetType(pc, PCLU));
#if defined(PETSC_HAVE_MUMPS)
        PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
#endif
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
    const PetscBool on[3]  = {hasX, hasY, hasZ};
    const PetscScalar E[3] = {on[0] ? TDSEZParser::Ex(t) : 0.0,
                              on[1] ? TDSEZParser::Ey(t) : 0.0,
                              on[2] ? TDSEZParser::Ez(t) : 0.0};
    for (int a = 0; a < 3; ++a) {
        if (!on[a] || D[a] == PETSC_NULLPTR) continue;
        PetscCall(MatMult(D[a], psi, s));
        PetscCall(VecAXPY(r, E[a], s));
    }
    PetscCall(KSPSolve(kspMass, r, w));

    for (int a = 0; a < 3; ++a) {
        acc[a] = 0.0;
        if (!on[a] || Vel[a] == PETSC_NULLPTR) continue;
        PetscScalar z;
        PetscCall(MatMult(Vel[a], psi, s));
        PetscCall(VecDot(s, w, &z));                 // z = w^H (Vel psi)
        acc[a] = -2.0 * charge * PetscImaginaryPart(z);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}
