#ifndef TDSEZ_INFO_HPP
#define TDSEZ_INFO_HPP

#include <petscsys.h>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

// ── TDSE-Z output-formatting helpers (quality bar) ────────────────────────
// fmtSci2 : 2-significant-digit scientific notation, used ONLY for eV.
// fmtAu   : fixed %+02.5e a.u. (NEVER fmtSci2) per quality bar.
// fmtEv   : eV NUMBER via fmtSci2 (2 sig digits); caller appends " eV".
/**
 * @brief Format a double in 2-significant-digit scientific notation.
 * @param x Value to format.
 * @return std::string in "%.2e" form (used only for eV values).
 */
static inline std::string TDSEZ_fmtSci2(double x) {
    char b[32];
    snprintf(b, sizeof(b), "%.2e", x);
    return std::string(b);
}
/**
 * @brief Format a double in fixed %+02.5e atomic-unit notation.
 * @param x Value to format.
 * @return std::string in "%+02.5e" form (a.u. values, never scientific-2).
 */
static inline std::string TDSEZ_fmtAu(double x) {
    char b[48];
    snprintf(b, sizeof(b), "%+02.5e", x);
    return std::string(b);
}
/**
 * @brief Convert an atomic-unit value to eV (2-significant-digit form).
 * @param au Value in atomic units.
 * @return std::string with the eV number (caller appends " eV").
 */
static inline std::string TDSEZ_fmtEv(double au) {
    return TDSEZ_fmtSci2(au * 27.211386245988);
}
/**
 * @brief Format a PetscScalar for terminal display (complex-aware).
 * @param z Scalar to format.
 * @return Real builds: "%.3f". Complex builds: "a+bi" decimal parts only
 *         (imaginary part omitted when zero), at most 1 char wider than the
 *         real format so KV rows stay within their fixed-width box.
 * @throws never
 */
static inline std::string TDSEZ_fmtZ(PetscScalar z) {
    char b[64];
#if defined(PETSC_USE_COMPLEX)
    const double re = (double)PetscRealPart(z);
    const double im = (double)PetscImaginaryPart(z);
    if (std::abs(im) < 1e-12)
        snprintf(b, sizeof(b), "%.3f", re);
    else
        snprintf(b, sizeof(b), "%.3f%+.3fi", re, im);
    return std::string(b);
#else
    snprintf(b, sizeof(b), "%.3f", (double)z);
    return std::string(b);
#endif
}


// ============================================================================
//  TDSEZInfo  —  centralised terminal-log printer
//  ----------------------------------------------------------------------------
//  Every ═══ rule, ▸ section, KV row, banner, and footer goes through this
//  class so that bar width, indentation, and centering are defined in exactly
//  one place.  To change the box width, edit W below; every call site updates
//  automatically.
//
//  Layout constants
//    INDENT  = 2 spaces before every line
//    W       = 70  (inner content width; rule = INDENT + W ═ chars)
//    Rule    = "  " + 70×═
//
//  Usage
//    TDSEZInfo info;          // default: PETSC_COMM_WORLD
//    info.banner(logoLines);  // print logo + title + credit
//    info.header("EIGENSOLVER SETUP");
//    info.section("PROBLEM");
//    info.kv("DOFs", "%d", n);
//    info.blank();
//    info.footer();
// ============================================================================
/**
 * @brief Centralised terminal-log printer for TDSE-Z.
 * @details All printed rules, section headers, key-value rows, banners, and
 * footers are routed through this class so that box width, indentation, and
 * centering are defined in exactly one place. Adjusting W updates every
 * call site automatically.
 */
class TDSEZInfo
{
public:
    /// Left indentation (spaces) applied to every printed line.
    static constexpr int INDENT = 2;
    /// Default inner content width used when stdout is redirected or the
    /// terminal size cannot be queried.
    static constexpr int W      = 70;

    /**
     * @brief Construct the printer bound to an MPI communicator.
     * @param comm MPI communicator for PetscPrintf (default PETSC_COMM_WORLD).
     */
    explicit TDSEZInfo(MPI_Comm comm = PETSC_COMM_WORLD)
        : m_comm(comm) {}

    /**
     * @brief Print a heavy horizontal rule (INDENT spaces + W box chars).
     */
    // ── heavy rule: "  " + W×═ + "\n" ──────────────────────────────────
    void rule() const
    {
        const int w = width();
        PetscPrintf(m_comm, "%*s", INDENT, "");
        for (int i = 0; i < w; ++i) PetscPrintf(m_comm, "═");
        PetscPrintf(m_comm, "\n");
    }

    /**
     * @brief Print a centered title between two heavy rules.
     * @param title Header text to center.
     */
    // ── centered header between two rules ──────────────────────────────
    void header(const char* title) const
    {
        PetscPrintf(m_comm, "\n");
        rule();
        center(title);
        rule();
    }

    /**
     * @brief Print a "▸ label" section line followed by a trailing rule.
     * @param label Section label text.
     */
    // ── ▸ section label + trailing rule (no leading rule) ──────────────
    void section(const char* label) const
    {
        PetscPrintf(m_comm, "%*s  ▸ %s\n", INDENT, "", label);
        rule();
    }

    /**
     * @brief Print a key-value row: left-aligned label, right-aligned value.
     * @param label Left-aligned label text.
     * @param fmt   printf-style format string for the value.
     * @param ...   Arguments matching @p fmt.
     */
    // ── KV row: compact PETSc-style "key = value" ────────────────────
    void kv(const char* label, const char* fmt, ...) const
    {
        const int w = width();
        char val[256];
        va_list args;
        va_start(args, fmt);
        vsnprintf(val, sizeof(val), fmt, args);
        va_end(args);

        const int labelWidth = 28;
        const int maxValue = PetscMax(8, w - 4 - labelWidth - 3);
        std::string shown(val);
        if ((int)shown.size() > maxValue) {
            shown.resize((size_t)PetscMax(0, maxValue - 3));
            shown += "...";
        }
        PetscPrintf(m_comm, "%*s    %-*s = %s\n",
                    INDENT, "", labelWidth, label, shown.c_str());
    }

    /**
     * @brief Print a blank line padded to the full box width.
     */
    // ── blank line matching box width ─────────────────────────────────
    void blank() const
    {
        PetscPrintf(m_comm, "\n");
    }

    /**
     * @brief Print a footer: single heavy rule followed by a blank line.
     */
    // ── footer: single rule + blank line ────────────────────────────────
    void footer() const
    {
        rule();
        PetscPrintf(m_comm, "\n");
    }

    /**
     * @brief Print a status row: icon + message, padded to the box width.
     * @param icon Status icon string (e.g. "✔", "✘").
     * @param fmt  printf-style format string for the message.
     * @param ...  Arguments matching @p fmt.
     */
    // ── status row: icon + message, padded to box width ────────────────
    void status(const char* icon, const char* fmt, ...) const
    {
        char msg[256];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);

        PetscPrintf(m_comm, "%*s    %s %s\n", INDENT, "", icon, msg);
    }

    /**
     * @brief Print a key-value row showing current and peak PETSc memory usage.
     * @param label Left-aligned label for the memory row.
     */
    // ── memory KV row ──────────────────────────────────────────────────
    void memKV(const char* label) const
    {
        PetscLogDouble mem_curr, mem_max;
        PetscMemoryGetCurrentUsage(&mem_curr);
        PetscMemoryGetMaximumUsage(&mem_max);
        char val[128];
        snprintf(val, sizeof(val), "%.3e MB  (peak %.3e MB)",
                 mem_curr / 1e6, mem_max / 1e6);
        kv(label, "%s", val);
    }

    /**
     * @brief Print a banner: centered logo lines followed by centered text lines.
     * @param logoLines Vector of pre-stripped logo lines (no trailing newline).
     * @param textLines Vector of title/credit lines, each centered in turn.
     */
    // ── banner: logo lines (centered) + text lines (each centered) ──────
    //  logoLines: vector of pre-stripped lines (no trailing \n)
    //  textLines: title + credit lines, each centered one under the other
    void banner(const std::vector<std::string>& logoLines,
                const std::vector<std::string>& textLines) const
    {
        rule();
        for (const auto& line : logoLines)
            centerStr(line.c_str(), displayWidth(line));
        PetscPrintf(m_comm, "\n");
        for (const auto& line : textLines)
            centerStr(line.c_str(), (int)line.size());
        PetscPrintf(m_comm, "\n");
    }

    /**
     * @brief Print a plain printf line (for content that does not fit the box model).
     * @param fmt printf-style format string.
     * @param ... Arguments matching @p fmt.
     */
    // ── plain printf (for lines that don't fit the box model) ──────────
    void raw(const char* fmt, ...) const
    {
        va_list args;
        va_start(args, fmt);
        va_end(args);
        // PetscVPrintf doesn't exist; format into a buffer and use PetscPrintf
        char buf[1024];
        va_list args2;
        va_start(args2, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args2);
        va_end(args2);
        PetscPrintf(m_comm, "%s", buf);
    }

    /**
     * @brief Center a string within the W-wide field (public for custom blocks).
     * @param s String to center.
     */
    // ── center a string in the W-wide field (public for custom blocks) ─
    void center(const char* s) const { centerStr(s, (int)strlen(s)); }

private:
    /// MPI communicator used for all PetscPrintf output.
    MPI_Comm m_comm;

    /// Return the usable inner width of the current terminal.  PETSc output
    /// is often redirected under batch systems, so ioctl is deliberately
    /// optional and the fixed width remains the deterministic fallback.
    static int width()
    {
        static const int cached = [] {
            int columns = 0;
#if defined(__unix__) || defined(__APPLE__)
            struct winsize ws{};
            if (isatty(STDOUT_FILENO) && ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0)
                columns = (int)ws.ws_col;
#endif
            if (columns <= 0) {
                const char *env = std::getenv("COLUMNS");
                if (env) columns = std::atoi(env);
            }
            if (columns <= 0) return W;
            // Keep a small readable margin and avoid pathological scheduler
            // values making the report wider than the display.
            const int inner = columns - INDENT;
            return inner < 70 ? 70 : (inner > 94 ? 94 : inner);
        }();
        return cached;
    }

    /**
     * @brief Print @p s centered within the W-wide field using a display width.
     * @param s  String to print.
     * @param dw Display width to use for centering (may differ from byte length).
     */
    void centerStr(const char* s, int dw) const
    {
        int pad = (dw > 0) ? (width() - dw) / 2 : 0;
        if (pad < 0) pad = 0;
        PetscPrintf(m_comm, "%*s%*s%s\n", INDENT, "", pad, "", s);
    }

    /**
     * @brief Compute the UTF-8 display width (code-point count) of a string.
     * @details Assumes every glyph occupies one column; counts code points
     * by stepping over UTF-8 lead/continuation bytes.
     * @param s Input string.
     * @return Number of code points (display columns).
     */
    // ── UTF-8 display width (code-point count; assumes every glyph = 1 col)
    static int displayWidth(const std::string& s)
    {
        int dw = 0;
        for (size_t i = 0; i < s.size(); ) {
            unsigned char c = (unsigned char)s[i];
            if      (c < 0x80) i += 1;
            else if (c < 0xC0) i += 1;     // continuation byte
            else if (c < 0xE0) i += 2;
            else if (c < 0xF0) i += 3;
            else               i += 4;
            ++dw;
        }
        return dw;
    }
};


#endif // TDSEZ_INFO_HPP
