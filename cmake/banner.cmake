# =============================================================================
#  TDSEZ banner generator for CMake configure-time output.
#
#  Reproduces the runtime TDSEZInfo::banner() format:
#    - 2-space indent
#    - 70-char heavy rule (═)
#    - ASCII-art logo, centered in the 70-char field
#    - blank line
#    - title + credit lines, centered
#    - 70-char heavy rule
#
#  CMake's string(LENGTH) counts BYTES, not UTF-8 code points. The TDSE-Z
#  logo uses Tifinagh glyphs (3 bytes each), so we use Python3 (already found
#  by the main CMakeLists.txt) to compute code-point widths at configure time.
#
#  Usage from CMakeLists.txt:
#    include(cmake/banner.cmake)
#    tdsez_banner()
# =============================================================================

# ── Layout constants (must match TDSEZInfo: INDENT=2, W=70) ──────────────────
set(_TDSEZ_INDENT 2)
set(_TDSEZ_W 70)

# ── The TDSE-Z ASCII-art logo (must match src/TDSE-Z_logo.txt exactly) ───────
set(_TDSEZ_LOGO_LINES
    "▐▆▆       ▆▆▋    "
    "▐ⵥⵥ  ▅▅▅  ⵥⵥ▋    "
    "▐ⵥⵥ▇▇ⵥⵥⵥ▇▇ⵥⵥ▋    "
    "▝▀▀▀▀ⵥⵥⵥ▀▀▀▀▘    "
    "▗▃▃▃▃ⵥⵥⵥ▃▃▃▃▖    "
    "▐ⵥⵥ▛▜ⵥⵥⵥ▛▜ⵥⵥ▌    "
    "▐ⵥⵥ  ▀▀▀  ⵥⵥ▌    "
    "▐ⵥⵥ       ⵥⵥ▌    "
)

# ── Credit / title lines (must match src/tdsez.cpp banner() call) ────────────
set(_TDSEZ_TITLE_LINES
    "TDSE-Z - Time-Dependent Schrodinger Equation solver (B-spline / IGA)"
    "Dr. Zakaria Dahbi  |  zdahbi@outlook.es"
    "Attosecond Quantum Physics Lab, King's College London, UK"
)

# ── Python script to compute code-point widths ──────────────────────────────
# Reads lines from stdin (one per line), prints code-point count per line.
# This avoids shell-escaping issues with apostrophes and special chars.
set(_TDSEZ_CP_SCRIPT
"import sys
for line in sys.stdin:
    print(len(line.rstrip(chr(10))))
"
)

# ── Helper: center a string in W-wide field, with INDENT prefix ──────────────
function(_tdsez_center_line line cp_count)
    math(EXPR _pad "(${_TDSEZ_W} - ${cp_count}) / 2")
    if(_pad LESS 0)
        set(_pad 0)
    endif()
    set(_prefix "  ")
    set(_center "")
    if(_pad GREATER 0)
        foreach(i RANGE 1 ${_pad})
            string(APPEND _center " ")
        endforeach()
    endif()
    message(STATUS "${_prefix}${_center}${line}")
endfunction()

# ── Main banner macro ───────────────────────────────────────────────────────
# Prints the full TDSE-Z banner at CMake configure time, matching the
# runtime output format exactly:
#   rule → logo (centered) → blank → title (centered) → rule
macro(tdsez_banner)
    # Build the input for the Python script: logo lines + title lines,
    # separated by newlines, written to a temp file.
    set(_banner_input "")
    foreach(_line ${_TDSEZ_LOGO_LINES})
        string(APPEND _banner_input "${_line}\n")
    endforeach()
    foreach(_line ${_TDSEZ_TITLE_LINES})
        string(APPEND _banner_input "${_line}\n")
    endforeach()

    # Write input to a temp file (CMake can't pipe multi-line strings to
    # stdin reliably with execute_process).
    set(_banner_in_file "${CMAKE_CURRENT_BINARY_DIR}/_tdsez_banner_in.txt")
    file(WRITE "${_banner_in_file}" "${_banner_input}")

    # Write the Python script to a temp file
    set(_banner_py_file "${CMAKE_CURRENT_BINARY_DIR}/_tdsez_banner_cp.py")
    file(WRITE "${_banner_py_file}" "${_TDSEZ_CP_SCRIPT}")

    # Run Python to get code-point counts
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "${_banner_py_file}"
        INPUT_FILE "${_banner_in_file}"
        OUTPUT_VARIABLE _cp_widths
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REPLACE "\n" ";" _cp_list "${_cp_widths}")

    # Clean up temp files
    file(REMOVE "${_banner_in_file}" "${_banner_py_file}")

    # Heavy rule: 2 spaces + 70 × ═
    set(_rule "  ")
    foreach(i RANGE 1 ${_TDSEZ_W})
        string(APPEND _rule "═")
    endforeach()

    # Print banner
    message(STATUS "${_rule}")

    # Logo lines (first 8 entries in _cp_list)
    set(_idx 0)
    foreach(_line ${_TDSEZ_LOGO_LINES})
        list(GET _cp_list ${_idx} _cp)
        _tdsez_center_line("${_line}" ${_cp})
        math(EXPR _idx "${_idx} + 1")
    endforeach()

    # Blank line
    message(STATUS "")

    # Title + credit lines (entries 8-10)
    set(_tidx 8)
    foreach(_line ${_TDSEZ_TITLE_LINES})
        list(GET _cp_list ${_tidx} _cp)
        _tdsez_center_line("${_line}" ${_cp})
        math(EXPR _tidx "${_tidx} + 1")
    endforeach()

    # Closing rule
    message(STATUS "${_rule}")
endmacro()
