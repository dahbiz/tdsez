#ifndef TDSEZ_PATHS_HPP
#define TDSEZ_PATHS_HPP

#include <filesystem>
#include <string>

/**
 * @brief Output file names: static/<name>_<what> and td/<name>_<what>.
 * @details <name> is the input file name without directory and extension
 *          (runs/h2p.inp -> h2p), so every file of a run starts with it:
 *
 *            static/h2p_eigen.h5        spectrum and eigenstates
 *            static/h2p_tdm.txt         transition-dipole summary
 *            static/h2p_Dx.bin          dipole matrix (SaveDipoleAxes)
 *            td/h2p_observables.h5      time series (dipole, populations, ...)
 *            td/h2p_wfs.h5              wavefunction snapshots
 *            td/h2p_surff.h5            t-SURFF spectrum
 *            td/h2p_ts.h5               time-stepper snapshots
 *
 *          Readers fall back to the names written before (EigenData_h2p.inp.h5,
 *          TimeEvolutionData_h2p.inp.h5, ...) so earlier runs stay readable.
 */
namespace TDSEZPaths {

/// Run name: the input file name without directory and extension.
inline std::string Name(const std::string &input)
{
    const std::filesystem::path p = std::filesystem::path(input).filename();
    return p.has_extension() ? p.stem().string() : p.string();
}

/// static/<name>_<what>, e.g. Static("h2p.inp", "eigen.h5") = static/h2p_eigen.h5.
inline std::string Static(const std::string &input, const std::string &what)
{
    return "static/" + Name(input) + "_" + what;
}

/// td/<name>_<what>, e.g. TD("h2p.inp", "observables.h5") = td/h2p_observables.h5.
inline std::string TD(const std::string &input, const std::string &what)
{
    return "td/" + Name(input) + "_" + what;
}

/// Name used before: <dir>/<prefix><input file name><suffix>, e.g.
/// Legacy("static", "EigenData_", "h2p.inp", ".h5") = static/EigenData_h2p.inp.h5.
inline std::string Legacy(const std::string &dir, const std::string &prefix, const std::string &input,
                          const std::string &suffix)
{
    return dir + "/" + prefix + std::filesystem::path(input).filename().string() + suffix;
}

/// The file to read: @p current if it exists, else the first existing legacy
/// name (the input given as "h2p.inp" or "h2p"), else @p current.
inline std::string Existing(const std::string &current, const std::string &dir, const std::string &prefix,
                            const std::string &input, const std::string &suffix)
{
    namespace fs = std::filesystem;
    if (fs::exists(current)) return current;
    for (const std::string &in : {input, Name(input) + ".inp"}) {
        const std::string old = Legacy(dir, prefix, in, suffix);
        if (fs::exists(old)) return old;
    }
    return current;
}

} // namespace TDSEZPaths

#endif // TDSEZ_PATHS_HPP
