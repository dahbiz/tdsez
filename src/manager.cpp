#include "tdsez_internal.hpp"

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

        PetscFunctionReturn(PETSC_SUCCESS);
}



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
                H5Pset_shuffle(dcpl);
                H5Pset_deflate(dcpl, (unsigned)h5CompressLevel);
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
            H5Sget_simple_extent_dims(fspace, curDims, NULL);
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
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL);
            herr_t werr = H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace,
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
        const PetscInt ZW = std::max({dipWidth, popWidth, energyWidth,
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

    // Update the cursor even on partial success so we never re-write or stall.
    lastWrittenSteps = recordedSteps;

    if (overall < 0) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB,
                "HDF5: failed to append datasets to %s", absPath.c_str());
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TDSEZManager::CloseHDF5()
{
    PetscFunctionBeginUser;
    if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);
    // Flush any remaining rows, then close the persistent handle (fix D).
    if (recordedSteps > lastWrittenSteps) {
        PetscCall(WriteHDF5(outputFilename));
    }
    if (h5File >= 0) {
        if (H5Fflush(h5File, H5F_SCOPE_GLOBAL) < 0) { /* non-fatal */ }
        if (H5Fclose(h5File) < 0) {
            PetscFunctionReturn(PETSC_ERR_FILE_OPEN);
        }
        h5File = -1;
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

// Write the open knot vectors (full, p+1-clamped ends) for each active axis plus
// the SplineDegree attribute into an already-open HDF5 file `file`. This makes a
// wavefunction file self-describing: together with the dof coefficients
// (psi_*/wavefunction) a reader can rebuild the B-spline basis (Cox-de Boor) and
// evaluate the wavefunction on any spatial grid. Returns 0 on success, <0 on error.
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


// ============================================================================
// t-SURFF (time-dependent surface flux, box faces, length-gauge input)
//
//   A(t)     = -int_0^t E(t') dt'                              (running, causal)
//   k_eff(t) = k + s*q*A(t)                                     s = SurffCouplingSign
//   Phi(k,t) = (1/2m) int_0^t |k_eff(t')|^2 dt'                 (running, causal)
//
//   b(k,T) = (2pi)^{-3/2}/(2m) * int_0^T dt e^{i Phi(k,t)} *
//            sum_faces sum_q w_q e^{-i k_eff(t).r_q} [ (d psi/dn)_q
//                                                 + i(k_eff(t).n_q) psi_q ]
//

//
// COST:    the naive face reduction is O(Nk^dim * Nq_face) per face per fold.
//          Because each box face has a FIXED normal axis, the k-component
//          along that axis enters ONLY as a scalar phase exp(-i k_d X_d) and
//          as a scalar factor i*sign*k_d in the (d psi/dn) + i(k.n)psi
//          integrand.  We therefore fold the expensive (tangent x quad) sum
//          into A_f(k_u) and B_f(k_u) (size Nk^(dim-1) each) and apply the
//          normal-k part analytically per fold.  This reduces the per-fold
//          cost to O(Nk^(dim-1)*Nq_face + Nk^dim) -- an exact factor-Nk
//          speedup for every active face, with NO approximation.
// ============================================================================

PetscErrorCode TDSEZManager::SetupSurff()
{
    PetscFunctionBeginUser;
    if (!TDSEZParser::out_tsurff) PetscFunctionReturn(PETSC_SUCCESS);

    surffNk   = TDSEZParser::SurffNk;
    surffKmax = TDSEZParser::SurffKmax;

    if (surffNk < 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                             "SurffNk must be >= 1");
    if (surffKmax <= 0.0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                                  "SurffKmax must be > 0");

    // Active spatial dimension and (constant) particle mass. t-SURFF uses the
    // standard constant-mass flux formula; we sample the mass at the origin
    // (exact for MassIsConstant runs, the natural point value otherwise).
    PetscInt dim = TDSEZParser::Dimension;
    surffMass = TDSEZParser::MassDist(0.0, 0.0, 0.0);

    kAxis.resize(surffNk);
    for (PetscInt i = 0; i < surffNk; ++i) {
        PetscReal frac = (surffNk == 1) ? 0.0 : (PetscReal)i / (PetscReal)(surffNk - 1);
        kAxis[i] = -surffKmax + 2.0*surffKmax*frac;
    }

    // diagnostic: per-face PES split (env TSURFF_FACE_SPLIT=1) — read BEFORE
    // allocating surffAmpFace below.
    {
        const char* ev = std::getenv("TSURFF_FACE_SPLIT");
        surffFaceSplit = (ev && std::atoi(ev) != 0) ? PETSC_TRUE : PETSC_FALSE;
    }

    if (rank == 0) {
        PetscInt Nprod = 1;
        for (PetscInt d = 0; d < dim; ++d) Nprod *= surffNk;
        surffPhi.assign(Nprod, 0.0);
        surffAmp.assign(Nprod, 0.0);
        if (surffFaceSplit)
            for (PetscInt f = 0; f < 6; ++f) surffAmpFace[f].assign(Nprod, 0.0);
    }

    Ax = Ay = Az = 0.0;
    surffTprev = 0.0;
    surffExPrev = TDSEZParser::Ex(0.0);
    surffEyPrev = TDSEZParser::Ey(0.0);
    surffEzPrev = TDSEZParser::Ez(0.0);
    surffInitialized = PETSC_TRUE;

    PetscCall(BuildFaceOperators());

    if (rank == 0) {
        PetscPrintf(PETSC_COMM_SELF,
            "  [t-SURFF debug] faceNquad = [%d,%d,%d,%d,%d,%d]  Bval_null = [%d,%d,%d,%d,%d,%d]  faceSign=[%+.0f,%+.0f,%+.0f,%+.0f,%+.0f,%+.0f]\n",
            (int)faceNquad[0],(int)faceNquad[1],(int)faceNquad[2],(int)faceNquad[3],(int)faceNquad[4],(int)faceNquad[5],
            (Bval_face[0]==PETSC_NULLPTR),(Bval_face[1]==PETSC_NULLPTR),(Bval_face[2]==PETSC_NULLPTR),(Bval_face[3]==PETSC_NULLPTR),(Bval_face[4]==PETSC_NULLPTR),(Bval_face[5]==PETSC_NULLPTR),
            (double)faceSign[0],(double)faceSign[1],(double)faceSign[2],(double)faceSign[3],(double)faceSign[4],(double)faceSign[5]);
    }

    // Create the psi->rank0 gather scatter (collective). All ranks participate;
    // rank 0 receives a sequential copy (psiZero) that the SELF face matrices
    // MatMult against in AccumulateSurff. We derive the parallel layout from M.
    {
        Vec tmpl = PETSC_NULLPTR;
        PetscCall(MatCreateVecs(m_M, &tmpl, NULL));   // parallel, matching psi
        PetscCall(VecScatterCreateToZero(tmpl, &surffScatter, &psiZero));
        PetscCall(VecDestroy(&tmpl));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


// Build Bval_face[f] / Bder_face[f] by iterating PetIGA boundary points.
//
// This mirrors the assembler's exact loop idiom (IGABeginElement /
// IGANextElement / IGAElementNextPoint) -- there is NO separate boundary
// loop in this PetIGA build.  Boundary points are flagged point->atboundary
// with point->boundary_id = 2*axis + side (0=x-,1=x+,2=y-,3=y+,4=z-,5=z+),
// and carry point->normal[] (outward normal) plus point->detS (surface
// measure dS = w_q, already multiplied by w_q in IGAElementBuildTabulation).
//
// All Nq_f rows of each face matrix are owned by rank 0 so that the MatMult
// result is naturally rank-0-local, matching the rest of the diagnostics.
PetscErrorCode TDSEZManager::BuildFaceOperators()
{
    PetscFunctionBeginUser;

    IGA iga = core_->iga;

    // Tangent axes (the directions lying IN a face) for each face are only needed
    // in AccumulateSurff; BuildFaceOperators just iterates all boundary forms.
    // Step 1: each rank gathers its own boundary quad data + records the
    // global dof indices it touches.
    struct FaceData {
        std::vector<PetscReal> X, Y, Z, W, N0, N1, N2; // geometry + normal
        std::vector<PetscInt>  rows;       // quad-point row index on this rank
        std::vector<PetscInt>  cols;       // global dof indices (nen per row)
        std::vector<PetscReal> Bval, Bder; // basis value / normal-deriv per (row,nen)
        PetscInt nen = 0;
        PetscInt count = 0;                // number of quad points on this rank
    };
    FaceData local[6];

    // t-SURFF collects flux at EVERY face of the spatial simulation box,
    // independent of which laser axis drives the dynamics. A 3D box has 6
    // faces (x± y± z±), a 2D box 4 (x± y±), a 1D box 2 (x±). The active
    // faces are therefore the geometric dimension, NOT hasX/hasY/hasZ (which
    // only reflect which dipole matrices were assembled for the Hamiltonian).
    // An in-plane (xy) drive still lets electrons exit through the z faces;
    // silently dropping them yields an incomplete 3D PES. The boundary flux
    // depends only on geometry + ψ, never on Dz/VelZ/dVdz, so visiting all
    // faces is always safe — even when those matrices are NULL.
    PetscInt dim = TDSEZParser::Dimension;
    PetscBool visit[3][2];
    for (PetscInt d = 0; d < 3; ++d)
        for (PetscInt s = 0; s < 2; ++s)
            visit[d][s] = (d < dim);

    IGAElement element = PETSC_NULLPTR;
    IGAPoint  point    = PETSC_NULLPTR;

    PetscCall(IGABeginElement(iga, &element));
    while (IGANextElement(iga, element)) {
        // advances element->atboundary + element->boundary_id and primes the
        // boundary quadrature points; we pass our own visit matrix so only the
        // faces consistent with the active dimensions are iterated. This is the
        // SAME idiom the assembler uses (petigaksp.c IGAElementNextFormMatrix).
        while (IGAElementNextForm(element, visit)) {
            if (!element->atboundary) continue;   // skip the trailing volume form
            PetscCall(IGAElementBeginPoint(element, &point));
            while (IGAElementNextPoint(element, point)) {
                PetscInt f = point->boundary_id;       // 2*axis + side
                PetscInt axis = f / 2, side = f % 2;
                if (!visit[axis][side]) continue;

                FaceData &fd = local[f];
                if (fd.nen == 0) {
                    fd.nen = point->nen;
                    fd.Bval.reserve(256 * point->nen);
                    fd.Bder.reserve(256 * point->nen);
                }
                PetscInt qrow = fd.count++;
                fd.rows.push_back(qrow);

                // geometry
                PetscReal xyz[3] = {0.0,0.0,0.0};
                PetscCall(IGAPointFormGeomMap(point, xyz));
                fd.X.push_back(xyz[0]); fd.Y.push_back(xyz[1]); fd.Z.push_back(xyz[2]);

                // surface measure dS = w_q (already includes the boundary Jacobian)
                PetscReal w = point->detS[0];
                if (TDSEZParser::SurffRadius > 0.0) {
                    // active-shell mask: only count points with |r| > SurffRadius
                    PetscReal r = std::sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]);
                    w *= (r > TDSEZParser::SurffRadius) ? 1.0 : 0.0;
                }
                fd.W.push_back(w);
                fd.N0.push_back(point->normal[0]);
                fd.N1.push_back(point->normal[1]);
                fd.N2.push_back(point->normal[2]);

                // global dof indices for this element's nen basis functions.
                // IGAElementGetClosure returns LOCAL (per-rank) PetIGA indices in
                // the local tensor layout (strides = node_gwidth[dim]). The
                // assembler's matrices carry iga->ao (app->petsc) so MatSetValues
                // converts automatically; our SELF face matrices do NOT, and
                // psiZero is a globally-gathered vector. Convert local->global via
                // the constant Cartesian-block rank offset:
                //   OFF = gstart[0] + gstart[1]*sizes[0] + gstart[2]*sizes[0]*sizes[1]
                // (AOApplicationToPetsc is collective and deadlocks inside the
                // element loop, so we add the offset inline instead.)
                const PetscInt *map = NULL;
                PetscInt nen = 0;
                PetscCall(IGAElementGetClosure(element, &nen, &map));
                for (PetscInt a = 0; a < nen; ++a) fd.cols.push_back(map[a]);

                // basis values N_a(q) and normal derivative dN_a/dn(q)
                const PetscReal *B = NULL;
                PetscCall(IGAPointGetShapeFuns(point, 0, (const PetscReal**)&B));
                const PetscReal *dB = NULL;
                PetscCall(IGAPointGetShapeFuns(point, 1, (const PetscReal**)&dB));
                const PetscInt dstride = dim;
                for (PetscInt a = 0; a < nen; ++a) {
                    PetscReal der = 0.0;
                    for (PetscInt ax = 0; ax < dim; ++ax)
                        der += point->normal[ax] * dB[a*dstride + ax];
                    fd.Bval.push_back(B[a]);
                    fd.Bder.push_back(der);
                }
            }
            PetscCall(IGAElementEndPoint(element, &point));
        }
    }
    PetscCall(IGAEndElement(iga, &element));

    // Convert LOCAL closure indices to GLOBAL petsc dof indices so the gathered
    // (global) psiZero is indexed correctly. IGAElementGetClosure returns per-rank
    // local indices; the assembler relies on iga->ao / the local-to-global mapping
    // (iga->map->mapping) installed on its matrices. We apply that same mapping
    // here. NOTE: AOApplicationToPetsc(iga->ao,...) deadlocks in this context, so
    // we use ISLocalToGlobalMappingApply on iga->map->mapping directly (local, no
    // MPI collective -> cannot deadlock).
    if (iga->map && iga->map->mapping) {
        for (PetscInt f = 0; f < 6; ++f) {
            if (!visit[f/2][f%2]) continue;
            std::vector<PetscInt>& c = local[f].cols;
            if (!c.empty())
                ISLocalToGlobalMappingApply(iga->map->mapping, (PetscInt)c.size(),
                                            c.data(), c.data());
        }
    }

    // Global row offset per face: the face matrix is built ONLY on rank 0 with
    // rank 0 owning every row, so each rank's local quad-point rows must be made
    // globally unique (prefix sum of lower ranks' counts) before insertion.
    for (PetscInt f = 0; f < 6; ++f) {
        if (!visit[f/2][f%2]) continue;
        PetscInt ln = (PetscInt)local[f].X.size();
        PetscInt off = 0;
        MPI_Exscan(&ln, &off, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
        if (rank == 0) off = 0;            // Exscan result undefined on rank 0
        for (PetscInt q = 0; q < (PetscInt)local[f].rows.size(); ++q)
            local[f].rows[q] += off;        // make row ids globally unique
    }

    // Step 2: gather per-face quad-point counts onto rank 0, then build the
    // matrices there (all rows owned by rank 0).
    for (PetscInt f = 0; f < 6; ++f) {
        if (!visit[f/2][f%2]) continue;
        PetscInt localNq = (PetscInt)local[f].X.size();
        PetscInt totalNq = 0;
        MPI_Reduce(&localNq, &totalNq, 1, MPIU_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
        faceNquad[f] = (rank == 0) ? totalNq : 0;

        if (rank != 0) continue; // only rank 0 builds + receives

        // Allocate PES work buffers for this face's tangent grid.
        PetscInt Nk = surffNk;
        // tangent k-grid dim = number of axes IN the face = dim-1
        PetscInt tdim = (dim <= 1) ? 0 : ((dim == 2) ? 1 : 2);
        PetscInt tanSize = 1;
        for (PetscInt t = 0; t < tdim; ++t) tanSize *= Nk;
        surffAf[f].assign(tanSize, 0.0);
        surffBf[f].assign(tanSize, 0.0);

        // Create Mat (MATAIJ works for a rank-0-owned matrix; on >1 MPI rank we
        // need MATMPIAIJ with 0 local rows on non-0 ranks, but since all rows
        // live on rank 0 we build an MPIAIJ with nrows_owner = totalNq on rank 0
        // and 0 elsewhere).
        PetscInt Ndof = 0;
        PetscCall(MatGetSize(m_M, &Ndof, NULL));

        // Face matrices + work vectors are SEQUENTIAL (PETSC_COMM_SELF), built and
        // owned entirely on rank 0. This avoids any collective MatMult that would
        // deadlock the other ranks. psi is gathered to rank 0 (psiZero) via a
        // VecScatter that ALL ranks participate in (see SetupSurff/AccumulateSurff).
        PetscCall(MatCreate(PETSC_COMM_SELF, &Bval_face[f]));
        PetscCall(MatSetSizes(Bval_face[f], totalNq, Ndof, totalNq, Ndof));
        PetscCall(MatSetType(Bval_face[f], MATSEQAIJ));
        PetscCall(MatSetUp(Bval_face[f]));
        PetscCall(MatCreate(PETSC_COMM_SELF, &Bder_face[f]));
        PetscCall(MatSetSizes(Bder_face[f], totalNq, Ndof, totalNq, Ndof));
        PetscCall(MatSetType(Bder_face[f], MATSEQAIJ));
        PetscCall(MatSetUp(Bder_face[f]));

        // Work vectors (seq, matching the SELF matrices).
        PetscCall(VecCreateSeq(PETSC_COMM_SELF, totalNq, &faceValVec[f]));
        PetscCall(VecCreateSeq(PETSC_COMM_SELF, totalNq, &faceDerVec[f]));

        // Receive this rank's local data and MatSetValues into the matrices.
        // (We are rank 0 here.) Loop over all ranks via MPI, receiving each
        // rank's FaceData and inserting rows.
        // First insert rank 0's own data.
        {
            FaceData &fd = local[f];
            PetscInt nen = fd.nen;
            for (PetscInt q = 0; q < (PetscInt)fd.rows.size(); ++q) {
                std::vector<PetscInt>  cols(nen);
                std::vector<PetscScalar> vrow(nen), drow(nen);
                for (PetscInt a = 0; a < nen; ++a) {
                    cols[a]  = fd.cols[(size_t)q*nen + a];
                    vrow[a]  = fd.Bval[(size_t)q*nen + a];
                    drow[a]  = fd.Bder[(size_t)q*nen + a];
                }
                PetscCall(MatSetValues(Bval_face[f], 1, &fd.rows[q], nen, cols.data(), vrow.data(), INSERT_VALUES));
                PetscCall(MatSetValues(Bder_face[f], 1, &fd.rows[q], nen, cols.data(), drow.data(), INSERT_VALUES));
            }
            // store geometry + weight on rank 0
            faceX[f] = std::move(fd.X);
            faceY[f] = std::move(fd.Y);
            faceZ[f] = std::move(fd.Z);
            faceW[f] = std::move(fd.W);
        }

        // Receive from ranks 1..size-1.
        PetscMPIInt size = 0;
        MPI_Comm_size(PETSC_COMM_WORLD, &size);
        for (PetscMPIInt src = 1; src < size; ++src) {
            // 1) localNq for src
            PetscInt srcNq = 0;
            MPI_Recv(&srcNq, 1, MPIU_INT, src, 100+f, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
            if (srcNq <= 0) continue;
            // 2) rows, cols, Bval, Bder, geometry, weights
            std::vector<PetscInt>  srcRows(srcNq);
            PetscInt nen = local[f].nen; // same nen across ranks (uniform p)
            std::vector<PetscInt>  srcCols((size_t)srcNq*nen);
            std::vector<PetscReal> srcBval((size_t)srcNq*nen);
            std::vector<PetscReal> srcBder((size_t)srcNq*nen);
            std::vector<PetscReal> srcX(srcNq), srcY(srcNq), srcZ(srcNq), srcW(srcNq);
            MPI_Recv(srcRows.data(), srcNq, MPIU_INT, src, 200+f, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(srcCols.data(), (size_t)srcNq*nen, MPIU_INT, src, 300+f, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(srcBval.data(), (size_t)srcNq*nen, MPIU_REAL, src, 400+f, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(srcBder.data(), (size_t)srcNq*nen, MPIU_REAL, src, 500+f, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(srcX.data(), srcNq, MPIU_REAL, src, 600+f, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(srcY.data(), srcNq, MPIU_REAL, src, 600+f, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(srcZ.data(), srcNq, MPIU_REAL, src, 600+f, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(srcW.data(), srcNq, MPIU_REAL, src, 700+f, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
            for (PetscInt q = 0; q < srcNq; ++q) {
                std::vector<PetscInt>  cols(nen);
                std::vector<PetscScalar> vrow(nen), drow(nen);
                for (PetscInt a = 0; a < nen; ++a) {
                    cols[a]  = srcCols[(size_t)q*nen + a];
                    vrow[a]  = srcBval[(size_t)q*nen + a];
                    drow[a]  = srcBder[(size_t)q*nen + a];
                }
                PetscCall(MatSetValues(Bval_face[f], 1, &srcRows[q], nen, cols.data(), vrow.data(), INSERT_VALUES));
                PetscCall(MatSetValues(Bder_face[f], 1, &srcRows[q], nen, cols.data(), drow.data(), INSERT_VALUES));
            }
            // append geometry + weights
            faceX[f].insert(faceX[f].end(), srcX.begin(), srcX.end());
            faceY[f].insert(faceY[f].end(), srcY.begin(), srcY.end());
            faceZ[f].insert(faceZ[f].end(), srcZ.begin(), srcZ.end());
            faceW[f].insert(faceW[f].end(), srcW.begin(), srcW.end());
        }

        PetscCall(MatAssemblyBegin(Bval_face[f], MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(Bval_face[f], MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyBegin(Bder_face[f], MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(Bder_face[f], MAT_FINAL_ASSEMBLY));
    }

    // Non-rank-0 ranks: send their local face data to rank 0.
    if (rank != 0) {
        for (PetscInt f = 0; f < 6; ++f) {
            if (!visit[f/2][f%2]) continue;
            FaceData &fd = local[f];
            PetscInt nen = fd.nen;
            PetscInt srcNq = (PetscInt)fd.X.size();
            MPI_Send(&srcNq, 1, MPIU_INT, 0, 100+f, PETSC_COMM_WORLD);
            if (srcNq <= 0) continue;
            MPI_Send(fd.rows.data(), srcNq, MPIU_INT, 0, 200+f, PETSC_COMM_WORLD);
            MPI_Send(fd.cols.data(), (size_t)srcNq*nen, MPIU_INT, 0, 300+f, PETSC_COMM_WORLD);
            MPI_Send(fd.Bval.data(), (size_t)srcNq*nen, MPIU_REAL, 0, 400+f, PETSC_COMM_WORLD);
            MPI_Send(fd.Bder.data(), (size_t)srcNq*nen, MPIU_REAL, 0, 500+f, PETSC_COMM_WORLD);
            MPI_Send(fd.X.data(), srcNq, MPIU_REAL, 0, 600+f, PETSC_COMM_WORLD);
            MPI_Send(fd.Y.data(), srcNq, MPIU_REAL, 0, 600+f, PETSC_COMM_WORLD);
            MPI_Send(fd.Z.data(), srcNq, MPIU_REAL, 0, 600+f, PETSC_COMM_WORLD);
            MPI_Send(fd.W.data(), srcNq, MPIU_REAL, 0, 700+f, PETSC_COMM_WORLD);
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


// Per-step accumulation.  Called from the TS monitor every accepted step.
PetscErrorCode TDSEZManager::AccumulateSurff(Vec psi, PetscReal t, PetscReal dt, PetscInt step)
{
    PetscFunctionBeginUser;
    if (!TDSEZParser::out_tsurff || !surffInitialized) PetscFunctionReturn(PETSC_SUCCESS);

    // ---- 1. Running vector potential A(t) (trapezoidal, every step, causal). ----
    PetscReal Ex_now = TDSEZParser::Ex(t);
    PetscReal Ey_now = TDSEZParser::Ey(t);
    PetscReal Ez_now = TDSEZParser::Ez(t);
    Ax += -0.5*(surffExPrev + Ex_now)*dt;
    Ay += -0.5*(surffEyPrev + Ey_now)*dt;
    Az += -0.5*(surffEzPrev + Ez_now)*dt;
    surffExPrev = Ex_now; surffEyPrev = Ey_now; surffEzPrev = Ez_now;
    surffTprev = t;

    const PetscReal s = TDSEZParser::SurffCouplingSign;
    const PetscReal qA_x = s * charge * Ax;
    const PetscReal qA_y = s * charge * Ay;
    const PetscReal qA_z = s * charge * Az;
    const PetscReal mass = surffMass;

    // ---- 2. Gather psi to rank 0 (collective on ALL ranks) via the scatter,
    //        then evaluate boundary data with the SELF face matrices. ----
    PetscCall(VecScatterBegin(surffScatter, psi, psiZero, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(surffScatter, psi, psiZero, INSERT_VALUES, SCATTER_FORWARD));

    // DIAGNOSTIC: dump psiZero ordering at the final step so we can compare the
    // face-matrix column indices against the true dof layout on equal footing.
    {
        PetscInt nG = 0; VecGetSize(psiZero, &nG);
        PetscScalar *pz = NULL; VecGetArray(psiZero, &pz);
        VecRestoreArray(psiZero, &pz);
    }

    // Everything past this point only matters on rank 0 (face mats are SELF, owned
    // by rank 0; psiZero is the seq copy). Non-zero ranks may return now, but the
    // scatter above already ran collectively so there is no deadlock.
    if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);

    for (int f = 0; f < 6; ++f) {
        if (Bval_face[f] == PETSC_NULLPTR) continue;
        PetscCall(MatMult(Bval_face[f], psiZero, faceValVec[f]));
        PetscCall(MatMult(Bder_face[f], psiZero, faceDerVec[f]));
    }

    const PetscInt dim = TDSEZParser::Dimension;
    const PetscInt Nk = surffNk;
    // Dimension-dependent t-SURFF prefactor (2π)^(-dim/2) / (2m): in 1D this is
    // (2π)^(-1/2), 2D (2π)^(-1), 3D (2π)^(-3/2). (Previously hardcoded to 3D.)
    const PetscReal norm3 = 1.0/std::pow(2.0*PETSC_PI, 0.5*dim) / (2.0*mass);

    // ---- 3. Running Phi(k,t): O(Nk^dim) elementwise, exact every step. ----
    if (dim == 1) {
        for (PetscInt ix = 0; ix < Nk; ++ix) {
            PetscReal kxv = kAxis[ix] + qA_x;
            PetscInt ik = ix;
            surffPhi[ik] += dt/(2.0*mass) * (kxv*kxv);
        }
    } else if (dim == 2) {
        for (PetscInt ix = 0; ix < Nk; ++ix) {
            PetscReal kxv = kAxis[ix] + qA_x;
            for (PetscInt iy = 0; iy < Nk; ++iy) {
                PetscReal kyv = kAxis[iy] + qA_y;
                PetscInt ik = ix*Nk + iy;
                surffPhi[ik] += dt/(2.0*mass) * (kxv*kxv + kyv*kyv);
            }
        }
    } else {
        for (PetscInt ix = 0; ix < Nk; ++ix) {
            PetscReal kxv = kAxis[ix] + qA_x;
            for (PetscInt iy = 0; iy < Nk; ++iy) {
                PetscReal kyv = kAxis[iy] + qA_y;
                for (PetscInt iz = 0; iz < Nk; ++iz) {
                    PetscReal kzv = kAxis[iz] + qA_z;
                    PetscInt ik = (ix*Nk + iy)*Nk + iz;
                    surffPhi[ik] += dt/(2.0*mass) * (kxv*kxv + kyv*kyv + kzv*kzv);
                }
            }
        }
    }

    // Fold the boundary flux into b(k). The expensive flux operators (Af/Bf) are
    // rebuilt only every OutputStrideSurff steps (see rebuildFlux below); the fold
    // itself runs EVERY step using the cached Af/Bf with the current phase and
    // weight dt. The Volkov phase surffPhi is advanced every step in StepSurff(),
    // so the time integral of the oscillatory phase stays exact, and striding only
    // the smooth flux rebuild keeps t-SURFF accurate for OutputStrideSurff > 1
    // (naively skipping steps aliases the fast Volkov oscillation).
    const PetscInt surffStride = TDSEZParser::OutputStrideSurff;
    const bool rebuildFlux = (surffStride <= 1) || ((step % surffStride) == 0);

    // ---- 4. Face-separable flux fold (EXACT, O(Nk^(dim-1)*Nq + Nk^dim)). ----
    // tdim = number of TANGENT axes (axes lying IN the face) = dim-1.
    const PetscInt tdim = (dim <= 1) ? 0 : ((dim == 2) ? 1 : 2);
    // Tangent axes must be computed from the ACTIVE dimension, NOT hardcoded for
    // 3D: in 2D the y-faces have tangent axis x (0), not z (2) — the old
    // hardcoded {{1,2},{1,2},{0,2},{0,2},{0,1},{0,1}} table pointed y-faces at a
    // non-existent z-axis, indexing the empty faceZ[] and corrupting the tangent
    // sum. Active axes are {0,1} for 2D, {0,1,2} for 3D.

    for (int f = 0; f < 6; ++f) {
        if (Bval_face[f] == PETSC_NULLPTR) continue;
        PetscInt Nq = faceNquad[f];

        // Per-face constants (cheap; needed by the fold every step).
        PetscInt axis = f / 2, side = f % 2;
        (void)side;   // only 'axis' (the normal direction) is needed here
        PetscReal nsign = faceSign[f];
        // tangent axes = active axes excluding the face normal 'axis'
        PetscInt u0 = -1, u1 = -1;
        {
            // build the list of axes != axis, up to 'dim' active axes
            PetscInt t[2], nt = 0;
            for (PetscInt a = 0; a < dim; ++a) if (a != axis) t[nt++] = a;
            // nt should equal tdim
            u0 = (nt >= 1) ? t[0] : -1;
            u1 = (nt >= 2) ? t[1] : -1;
        }

        PetscInt tanSize = 1;
        for (PetscInt d = 0; d < tdim; ++d) tanSize *= Nk;
        std::vector<PetscScalar> &Af = surffAf[f];
        std::vector<PetscScalar> &Bf = surffBf[f];

        // Rebuild the expensive boundary-flux operators (Af/Bf) only every
        // OutputStrideSurff steps; reuse the cached values otherwise. The flux
        // J is smooth, so a piecewise-constant approximation is accurate, while
        // the volatile Volkov phase is still integrated every step below.
        if (rebuildFlux) {
        PetscScalar *faceVal = NULL, *faceDer = NULL;
        PetscCall(VecGetArray(faceValVec[f], &faceVal));
        PetscCall(VecGetArray(faceDerVec[f], &faceDer));
        for (PetscInt i = 0; i < tanSize; ++i) { Af[i] = 0.0; Bf[i] = 0.0; }

        if (tdim == 0) {
            // 1D: the only face is x+/-; no tangent axes, so
            //   A_f = sum_q w_q dpsi/dn_q ,  B_f = sum_q w_q psi_q .
            for (PetscInt q = 0; q < Nq; ++q) {
                PetscScalar eik = PetscScalar(1.0, 0.0); // exp(0): no tangent r in 1D
                Af[0] += faceW[f][q] * eik * faceDer[q];
                Bf[0] += faceW[f][q] * eik * faceVal[q];
            }
        } else if (tdim == 1) {
            // 2D (or a 3D degenerate case): exactly ONE tangent axis u0.
            for (PetscInt iu0 = 0; iu0 < Nk; ++iu0) {
                PetscReal ku0 = kAxis[iu0];
                PetscScalar a = 0.0, b = 0.0;
                for (PetscInt q = 0; q < Nq; ++q) {
                    PetscReal ru0 = (u0==0)?faceX[f][q] : (u0==1)?faceY[f][q] : faceZ[f][q];
                    PetscReal phase = ku0*ru0;
                    PetscScalar eik = PetscScalar(std::cos(phase), -std::sin(phase));
                    a += faceW[f][q] * eik * faceDer[q];
                    b += faceW[f][q] * eik * faceVal[q];
                }
                Af[iu0] = a; Bf[iu0] = b;   // tidx == iu0 (single tangent axis)
            }
        } else {
            // 3D: two tangent axes u0,u1 (tdim==2).
            for (PetscInt iu0 = 0; iu0 < Nk; ++iu0) {
                PetscReal ku0 = kAxis[iu0];
                for (PetscInt iu1 = 0; iu1 < Nk; ++iu1) {
                    PetscReal ku1 = kAxis[iu1];
                    PetscScalar a = 0.0, b = 0.0;
                    for (PetscInt q = 0; q < Nq; ++q) {
                        PetscReal ru0 = (u0==0)?faceX[f][q] : (u0==1)?faceY[f][q] : faceZ[f][q];
                        PetscReal ru1 = (u1==0)?faceX[f][q] : (u1==1)?faceY[f][q] : faceZ[f][q];
                        PetscReal phase = ku0*ru0 + ku1*ru1;
                        PetscScalar eik = PetscScalar(std::cos(phase), -std::sin(phase));
                        a += faceW[f][q] * eik * faceDer[q];
                        b += faceW[f][q] * eik * faceVal[q];
                    }
                    PetscInt tidx = iu0*Nk + iu1;
                    Af[tidx] = a; Bf[tidx] = b;
                }
            }
        }

        PetscCall(VecRestoreArray(faceValVec[f], &faceVal));
        PetscCall(VecRestoreArray(faceDerVec[f], &faceDer));
        } // end if(rebuildFlux)

        // helpers to recover tangent axis indices from the 1-D tangent index.
        // For tdim==1 the single tangent axis carries the full index t.
        // For tdim==2 the flat tangent index t in [0,Nk^2) must be decomposed
        // into its two components (t/Nk, t%Nk); treating t as BOTH tangent
        // components (as the old lambdas did) overflowed the k-grid index ik
        // (size Nk^dim) and SEGV'd in 3D.
        auto iu0_of = [&](PetscInt t) -> PetscInt { return (tdim >= 2) ? (t / Nk) : t; };
        auto iu1_of = [&](PetscInt t) -> PetscInt { return (tdim >= 2) ? (t % Nk) : 0; };

        // Now fold J_f(k) into the full k-grid accumulator, applying the normal-k
        // part analytically (exact, no approximation). Done EVERY step with the
        // cached Af/Bf and the current phase, so the time integral is exact.
        auto fold = [&](PetscInt tidx, PetscInt idn, PetscReal kdn) {
            PetscScalar J = Af[tidx] + PetscScalar(0.0,1.0)*nsign*kdn*Bf[tidx];
            // full k index: place k_d at axis 'axis', tangent comp(s) at u0/u1.
            PetscInt ik;
            if (dim == 1) {
                ik = idn; // k_x
            } else if (dim == 2) {
                // tdim==1: the single tangent axis IS tidx; place it at u0.
                PetscInt c[2] = {0,0};
                c[axis] = idn;
                c[u0]   = tidx;
                ik = c[0]*Nk + c[1];
            } else {
                // 3D: tdim==2, tangent axes u0,u1; k_d sits at 'axis'
                PetscInt c[3] = {0,0,0};
                c[axis] = idn;
                c[u0]   = iu0_of(tidx);
                c[u1]   = iu1_of(tidx);
                ik = (c[0]*Nk + c[1])*Nk + c[2];
            }
            // normal-face phase factor e^{-i k_d X_d} (X_d = constant face position)
            PetscReal Xd_val = (axis==0)?faceX[f][0] : (axis==1)?faceY[f][0] : faceZ[f][0];
            PetscScalar eX = PetscScalar(std::cos(-kdn*Xd_val), std::sin(-kdn*Xd_val)); // e^{-i k_d X_d}
            PetscReal ph = surffPhi[ik];
            PetscScalar phase_t = PetscScalar(std::cos(ph), std::sin(ph)); // e^{i Phi(k,t)}
            // Weight dt every step: the phase is integrated exactly, the flux J
            // is the cached piecewise-constant value from the last rebuild.
            surffAmp[ik] += dt * norm3 * phase_t * eX * J;
            if (surffFaceSplit) surffAmpFace[f][ik] += dt * norm3 * phase_t * eX * J;
        };

        // iterate tangent grid x normal grid, building the tangent index
        for (PetscInt t = 0; t < tanSize; ++t) {
            PetscReal kdn_axis = (axis==0)?qA_x : (axis==1)?qA_y : qA_z;
            for (PetscInt idn = 0; idn < Nk; ++idn) {
                PetscReal kdn = kAxis[idn] + kdn_axis;
                fold(t, idn, kdn);
            }
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


// Write |b(k)|^2 (the photoelectron momentum spectrum) + the k-axes to a
// dedicated HDF5 file mirroring the existing WriteHDF5() low-level style.
PetscErrorCode TDSEZManager::FinalizeSurff()
{
    PetscFunctionBeginUser;
    if (!TDSEZParser::out_tsurff || !surffInitialized) PetscFunctionReturn(PETSC_SUCCESS);
    if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);

    // ---- DEBUG: dump final psi (rank-0 seq copy) for independent reference ----
    if (std::getenv("TSURFF_DUMP_PSI")) {
        PetscInt nGlobal = 0;
        PetscCall(VecGetSize(psiZero, &nGlobal));
        const PetscScalar *pz = NULL;
        PetscCall(VecGetArrayRead(psiZero, &pz));
        std::string pname = "td/psi_final_" + std::filesystem::path(inputFile).filename().string() + ".dat";
        FILE *pf = fopen(pname.c_str(), "wb");
        if (pf) {
            PetscInt dim = TDSEZParser::Dimension;
            fwrite(&dim, sizeof(PetscInt), 1, pf);
            fwrite(&nGlobal, sizeof(PetscInt), 1, pf);
            for (PetscInt i = 0; i < nGlobal; ++i) {
                PetscReal re = PetscRealPart(pz[i]), im = PetscImaginaryPart(pz[i]);
                fwrite(&re, sizeof(PetscReal), 1, pf);
                fwrite(&im, sizeof(PetscReal), 1, pf);
            }
            fclose(pf);
        }
        PetscCall(VecRestoreArrayRead(psiZero, &pz));
    }

    PetscInt Nk = surffNk;
    PetscInt dim = TDSEZParser::Dimension;
    PetscInt Nprod = 1;
    for (PetscInt d = 0; d < dim; ++d) Nprod *= Nk;
    std::vector<PetscReal> pes(Nprod);
    for (PetscInt i = 0; i < Nprod; ++i)
        pes[i] = PetscRealPart(PetscConj(surffAmp[i])*surffAmp[i]);

    std::string base = std::filesystem::path(inputFile).filename().string();
    std::string fname = "td/surff_" + base + ".h5";
    std::string absPath = std::filesystem::absolute(fname).string();

    hid_t file = H5Fcreate(absPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "t-SURFF: could not create %s", absPath.c_str());

    auto write1d = [&](const char* name, const std::vector<PetscReal>& v) {
        hsize_t dims[1] = {(hsize_t)v.size()};
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dset  = H5Dcreate(file, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
        H5Dclose(dset); H5Sclose(space);
    };
    auto writeNd = [&](const char* name, const std::vector<PetscReal>& v) {
        hsize_t dims[3] = {1,1,1};
        for (PetscInt d = 0; d < dim; ++d) dims[d] = (hsize_t)Nk;
        hid_t space = H5Screate_simple(3, dims, NULL);
        hid_t dset  = H5Dcreate(file, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
        H5Dclose(dset); H5Sclose(space);
    };

    write1d("kaxis", kAxis);
    writeNd("pes",   pes);

    // Per-face-family diagnostic PES (x/y/z face sums) when TSURFF_FACE_SPLIT=1.
    if (surffFaceSplit) {
        auto writeFace = [&](const char* name, int f0, int f1) {
            std::vector<PetscReal> pf(Nprod, 0.0);
            for (PetscInt f = f0; f <= f1; ++f)
                for (PetscInt i = 0; i < Nprod; ++i)
                    pf[i] += PetscRealPart(PetscConj(surffAmpFace[f][i])*surffAmpFace[f][i]);
            writeNd(name, pf);
        };
        writeFace("pes_xfaces", 0, 1);
        writeFace("pes_yfaces", 2, 3);
        writeFace("pes_zfaces", 4, 5);
    }

    // Also expose the factorization metadata for diagnostics.
    PetscReal kmax_meta = surffKmax;
    PetscInt  nk_meta   = surffNk;
    PetscInt  dim_meta  = dim;
    PetscInt  stride_meta = TDSEZParser::OutputStrideSurff;
    hid_t sspace = H5Screate(H5S_SCALAR);
    hid_t d_nk  = H5Dcreate(file, "Nk",    H5T_NATIVE_INT,    sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t d_km  = H5Dcreate(file, "Kmax",  H5T_NATIVE_DOUBLE, sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t d_dim = H5Dcreate(file, "Dim",   H5T_NATIVE_INT,    sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t d_st  = H5Dcreate(file, "Stride",H5T_NATIVE_INT,    sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(d_nk, H5T_NATIVE_INT,    H5S_ALL, H5S_ALL, H5P_DEFAULT, &nk_meta);
    H5Dwrite(d_km, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &kmax_meta);
    H5Dwrite(d_dim,H5T_NATIVE_INT,    H5S_ALL, H5S_ALL, H5P_DEFAULT, &dim_meta);
    H5Dwrite(d_st, H5T_NATIVE_INT,    H5S_ALL, H5S_ALL, H5P_DEFAULT, &stride_meta);
    H5Dclose(d_nk); H5Dclose(d_km); H5Dclose(d_dim); H5Dclose(d_st); H5Sclose(sspace);

    H5Fclose(file);

    PetscPrintf(PETSC_COMM_SELF,
        "  t-SURFF: wrote PES to %s  (dim=%d, Nk=%d, Kmax=%.3f, size=%d)\n",
        absPath.c_str(), (int)dim, (int)Nk, (double)surffKmax, (int)Nprod);

    PetscFunctionReturn(PETSC_SUCCESS);
}




// ---- TDSEZManager constructor / destructor (out-of-line) ----

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
            // during propagation (TimeEvolutionData, wfs_*, ac_*, ts_*, surff_*)
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

            // Free the buffers
            if (dipBuffer)    PetscFree(dipBuffer);
            if (popBuffer)    PetscFree(popBuffer);
            if (energyBuffer) PetscFree(energyBuffer);
            if (currBuffer)   PetscFree(currBuffer);
            if (acBuffer)     PetscFree(acBuffer);
            if (m_psi0)        VecDestroy(&m_psi0);

            // t-SURFF: destroy face evaluation matrices + MatMult work vectors
            // (owned by the manager; not aliased from the assembler).
            for (int f = 0; f < 6; ++f) {
                if (Bval_face[f]) MatDestroy(&Bval_face[f]);
                if (Bder_face[f]) MatDestroy(&Bder_face[f]);
                if (faceValVec[f]) VecDestroy(&faceValVec[f]);
                if (faceDerVec[f]) VecDestroy(&faceDerVec[f]);
            }
            // psi gather scatter + seq copy
            if (psiZero)       VecDestroy(&psiZero);
            if (surffScatter)  VecScatterDestroy(&surffScatter);

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