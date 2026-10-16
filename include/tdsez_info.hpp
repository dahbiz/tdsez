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

// ── TDSE-Z number formats ─────────────────────────────────────────────────
/// Atomic units: "%+.5e".
static inline std::string TDSEZ_fmtAu(double x) {
    char b[48];
    snprintf(b, sizeof(b), "%+.5e", x);
    return std::string(b);
}
/// eV number of an atomic-unit energy, fixed with two decimals ("-13.60");
/// the caller appends " eV".
static inline std::string TDSEZ_fmtEv(double au) {
    char b[48];
    snprintf(b, sizeof(b), "%+.2f", au * 27.211386245988);
    return std::string(b);
}
/**
 * @brief Format a PetscScalar for terminal display (complex-aware).
 * @return Real builds: "%.3f". Complex builds: "a+bi" (imaginary part omitted
 *         when zero).
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
//  TDSEZInfo  —  the terminal log
//  ----------------------------------------------------------------------------
//  Every line of the run log goes through this class, so the layout is
//  defined once:
//
//    ══════════════════════════════════════   rule()      full width
//                   TITLE                     header()    stage of the run
//    ══════════════════════════════════════
//
//        ─────────── LABEL ───────────        section()   part of a stage
//        Key                    = value       kv()
//        text, word-wrapped                   line()
//        ⵣ message                            status(), warn()
//
//  The body is a column of at most BODY characters centred under the rules;
//  the rules span the terminal (70 to 94 columns; 80 when not a terminal).
// ============================================================================
class TDSEZInfo
{
public:
    /// Left indentation (spaces) of the rules.
    static constexpr int INDENT = 2;
    /// Rule width when stdout is not a terminal (batch logs).
    static constexpr int W      = 80;
    /// Widest body column.
    static constexpr int BODY   = 78;
    /// Label width of kv() rows.
    static constexpr int LABEL  = 30;

    explicit TDSEZInfo(MPI_Comm comm = PETSC_COMM_WORLD) : m_comm(comm) {}

    /// Heavy rule across the full width.
    void rule() const
    {
        std::string r((size_t)INDENT, ' ');
        for (int i = 0; i < width(); ++i) r += "═";
        PetscPrintf(m_comm, "%s\n", r.c_str());
    }

    /// Stage of the run: centred title between two rules.
    void header(const char* title) const
    {
        PetscPrintf(m_comm, "\n");
        rule();
        center(title);
        rule();
    }

    /// Part of a stage: "──── LABEL ────" across the body, centred.
    void section(const char* label) const
    {
        const int lw = displayWidth(label) + 2, bw = body();
        const int left = PetscMax(3, (bw - lw) / 2), right = PetscMax(3, bw - lw - left);
        std::string s;
        for (int i = 0; i < left; ++i) s += "─";
        s += " "; s += label; s += " ";
        for (int i = 0; i < right; ++i) s += "─";
        PetscPrintf(m_comm, "\n%*s%s\n", col(), "", s.c_str());
    }

    /// Thin rule across the body.
    void thin() const
    {
        std::string s;
        for (int i = 0; i < body(); ++i) s += "─";
        PetscPrintf(m_comm, "%*s%s\n", col(), "", s.c_str());
    }

    /// "Label = value" row; a long value wraps under itself.
    void kv(const char* label, const char* fmt, ...) const
    {
        char val[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(val, sizeof(val), fmt, args);
        va_end(args);
        std::string lab(label);
        const int lw = displayWidth(lab);
        if (lw < LABEL) lab.append((size_t)(LABEL - lw), ' ');
        emit(lab + " = ", val, LABEL + 3);
    }

    /// Body text, word-wrapped to the body width.
    void line(const char* fmt, ...) const
    {
        char msg[2048];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);
        emit("", msg, 2);
    }

    /// One preformatted body line, never wrapped (tables, progress).
    void row(const char* fmt, ...) const
    {
        char msg[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);
        PetscPrintf(m_comm, "%*s%s\n", col(), "", msg);
    }

    /// Icon and message; continuation lines align after the icon.
    void status(const char* icon, const char* fmt, ...) const
    {
        char msg[2048];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);
        const std::string head = std::string(icon) + " ";
        emit(head, msg, displayWidth(head));
    }

    /// "ⵣ WARNING: message".
    void warn(const char* fmt, ...) const
    {
        char msg[2048];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);
        emit("ⵣ WARNING: ", msg, 2);
    }

    /// "✖ TDSEZ FATAL: message" (the caller aborts).
    void fatal(const char* fmt, ...) const
    {
        char msg[2048];
        va_list args;
        va_start(args, fmt);
        vsnprintf(msg, sizeof(msg), fmt, args);
        va_end(args);
        PetscPrintf(m_comm, "\n");
        emit("✖ TDSEZ FATAL: ", msg, 2);
    }

    /// Empty line.
    void blank() const { PetscPrintf(m_comm, "\n"); }

    /// Closing rule of a stage.
    void footer() const
    {
        rule();
    }

    /// Resident memory of this process.
    void memKV(const char* label) const
    {
        PetscLogDouble mem_curr;
        PetscMemoryGetCurrentUsage(&mem_curr);
        kv(label, "%.1f MB", mem_curr / 1048576.0);
    }

    /// Opening banner: the logo as a centred block, then each group of text
    /// lines centred (groups separated by an empty line).
    void banner(const std::vector<std::string>& logoLines,
                const std::vector<std::vector<std::string>>& groups) const
    {
        std::vector<std::string> logo;
        for (std::string l : logoLines) {
            while (!l.empty() && (l.back() == ' ' || l.back() == '\t')) l.pop_back();
            logo.push_back(l);
        }
        while (!logo.empty() && logo.back().empty()) logo.pop_back();
        int lw = 0;
        for (const auto& l : logo) lw = PetscMax(lw, displayWidth(l));
        const int pad = PetscMax(0, (width() - lw) / 2);
        rule();
        PetscPrintf(m_comm, "\n");
        for (const auto& l : logo) PetscPrintf(m_comm, "%*s%s\n", INDENT + pad, "", l.c_str());
        for (const auto& g : groups) {
            PetscPrintf(m_comm, "\n");
            for (const auto& t : g) center(t.c_str(), width());
        }
    }

    /// Centre a string under the rules; a string wider than the body is
    /// word-wrapped and every line centred.
    void center(const char* s, int w = 0) const
    {
        for (const auto& l : wrap(s, w > 0 ? w : body()))
            PetscPrintf(m_comm, "%*s%s\n", INDENT + PetscMax(0, (width() - displayWidth(l)) / 2), "", l.c_str());
    }

    /// Width of the rules.
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
            const int inner = columns - 2 * INDENT;
            return inner < 70 ? 70 : (inner > 94 ? 94 : inner);
        }();
        return cached;
    }
    /// Width of the body column.
    static int body() { return PetscMin(BODY, width() - 8); }
    /// First column of the body.
    static int col() { return INDENT + (width() - body()) / 2; }

    /// Display width: code points (every glyph used here is one column).
    static int displayWidth(const std::string& s)
    {
        int dw = 0;
        for (unsigned char c : s) if ((c & 0xC0) != 0x80) ++dw;
        return dw;
    }

private:
    MPI_Comm m_comm;

    /// Greedy word wrap of @p s to @p w columns (explicit '\n' kept).
    static std::vector<std::string> wrap(const std::string& s, int w)
    {
        std::vector<std::string> out;
        size_t i = 0;
        std::string cur;
        int cw = 0;
        auto flush = [&]() { out.push_back(cur); cur.clear(); cw = 0; };
        while (i <= s.size()) {
            if (i == s.size() || s[i] == '\n') {
                flush();
                ++i;
                if (i > s.size()) break;
                continue;
            }
            if (s[i] == ' ') { ++i; continue; }
            size_t j = i;
            while (j < s.size() && s[j] != ' ' && s[j] != '\n') ++j;
            const std::string word = s.substr(i, j - i);
            const int ww = displayWidth(word);
            if (cw > 0 && cw + 1 + ww > w) flush();
            if (cw > 0) { cur += ' '; ++cw; }
            cur += word; cw += ww;
            i = j;
        }
        while (!out.empty() && out.back().empty()) out.pop_back();
        if (out.empty()) out.push_back("");
        return out;
    }

    /// Print @p head + @p text wrapped to the body; continuation lines are
    /// indented by @p hang columns.
    void emit(const std::string& head, const std::string& text, int hang) const
    {
        const auto lines = wrap(text, PetscMax(20, body() - PetscMax(displayWidth(head), hang)));
        for (size_t k = 0; k < lines.size(); ++k) {
            if (k == 0) PetscPrintf(m_comm, "%*s%s%s\n", col(), "", head.c_str(), lines[k].c_str());
            else PetscPrintf(m_comm, "%*s%s\n", col() + hang, "", lines[k].c_str());
        }
    }
};


#endif // TDSEZ_INFO_HPP
