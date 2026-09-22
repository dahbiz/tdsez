#ifndef TDSEZ_INFO_HPP
#define TDSEZ_INFO_HPP

#include <petscsys.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
    /// Inner content width; a full rule line is INDENT + W box-drawing chars.
    static constexpr int W      = 70;   // inner width (rule = INDENT + W ═)

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
        PetscPrintf(m_comm, "%*s", INDENT, "");
        for (int i = 0; i < W; ++i) PetscPrintf(m_comm, "═");
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
        int llen = (int)strlen(label) + 4;   // "  ▸ " prefix
        int pad  = W - llen;
        if (pad < 0) pad = 0;
        PetscPrintf(m_comm, "%*s  ▸ %s%*s\n", INDENT, "", label, pad, "");
        rule();
    }

    /**
     * @brief Print a key-value row: left-aligned label, right-aligned value.
     * @param label Left-aligned label text.
     * @param fmt   printf-style format string for the value.
     * @param ...   Arguments matching @p fmt.
     */
    // ── KV row: left-aligned label, right-aligned value ────────────────
    //    "  " + 4-space inner indent + label + gap + value + 2 trailing
    void kv(const char* label, const char* fmt, ...) const
    {
        char val[256];
        va_list args;
        va_start(args, fmt);
        vsnprintf(val, sizeof(val), fmt, args);
        va_end(args);

        int llen = (int)strlen(label);
        int vlen = (int)strlen(val);
        int gap  = W - 4 - llen - vlen;
        if (gap < 1) gap = 1;
        PetscPrintf(m_comm, "%*s    %-*s%*s  \n",
                     INDENT, "", llen, label, gap + vlen, val);
    }

    /**
     * @brief Print a blank line padded to the full box width.
     */
    // ── blank line matching box width ─────────────────────────────────
    void blank() const
    {
        PetscPrintf(m_comm, "%*s", INDENT, "");
        for (int i = 0; i < W; ++i) PetscPrintf(m_comm, " ");
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

        int mlen = (int)strlen(icon) + 1 + (int)strlen(msg);
        int pad  = W - 4 - mlen;
        if (pad < 0) pad = 0;
        PetscPrintf(m_comm, "%*s    %s %s%*s\n",
                     INDENT, "", icon, msg, pad, "");
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

    /**
     * @brief Print @p s centered within the W-wide field using a display width.
     * @param s  String to print.
     * @param dw Display width to use for centering (may differ from byte length).
     */
    void centerStr(const char* s, int dw) const
    {
        int pad = (dw > 0) ? (W - dw) / 2 : 0;
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
