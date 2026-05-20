from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

from .utils import (
    normalize, latex_escape, parse_time_to_seconds,
    is_solved, is_completed, avg, fmt_seconds,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

GROUPS = [
    (
        "Classical Models",
        {
            "foo",
            "fooeligibility",
            "kas2",
            "kas2eck",
            "naxos",
            "naxoseck",
            "naxoseckpfs",
            "tutorial",
            "yubikey",
            "yubikeyhsm",
            "umpfs",
        },
    ),
    (
        "Complex Models",
        {
            "wireguard",
            "5ghandover5gstoeps",
            "5gstoepsovern26handover",
            "gcm",
            "pkcs11aead",
            "signal",
            "signalloopingforgood",
            "signalrevealing",
        },
    ),
    (
        "State-of-the-Art Models",
        {
            "5gaka",
            "5gakafix",
            "5ghandoverxn",
            "5gxnhandover",
            "wpa2",
            "wpa2fourwayhandshake",
            "spdm",
            "compositionspdm",
        },
    ),
]

PRETTY_PROTOCOL_NAMES = {
    "5gakafix": "5G AKA",
    "5gxnhandover": "5G Handover XN",
    "5ghandoverxn": "5G Handover XN",
    "5gstoepsovern26handover": "5G Handover EPS N26",
    "fooeligibility": "FOO Eligibility",
    "kas2eck": "KAS2 eCK",
    "naxoseck": "NAXOS eCK",
    "naxoseckpfs": "NAXOS eCK PFS",
    "pkcs11aead": "PKCS11 AEAD",
    "signalrevealing": "Signal",
    "tutorial": "Tutorial",
    "umpfs": "UM PFS",
    "yubikey": "YubiKey",
    "yubikeyhsm": "YubiKey HSM",
    "compositionspdm": "SPDM",
    "wireguard": "Wireguard",
    "wpa2fourwayhandshake": "WPA2",
}


PROTOCOL_ORDER = [
    "fooeligibility", "kas2eck", "naxoseck", "naxoseckpfs",
    "tutorial", "umpfs", "yubikey", "yubikeyhsm",
    "5gstoepsovern26handover", "pkcs11aead", "signalrevealing", "wireguard",
    "5gakafix", "5gxnhandover", "compositionspdm", "wpa2fourwayhandshake",
]

HYPERPARAM_KEYS = [
    ("search/budget", "$B$"),
    ("search/budget_increase_factor", "$B_{\\text{inc}}$"),
    ("search/num_searches", "$S$"),
    ("search/expand_top_n", "$N$"),
    ("optimizer/learning_rate", "LR"),
    ("ucb/heuristic_weight", "$\\lambda$"),
    ("ucb/value_penalty", "$\\delta$"),
    ("ucb/temperature", "Temp"),
    ("ucb/c_and", "$c_{\\text{and}}$"),
    ("ucb/pb_c_base", "$c_{\\text{base}}$"),
    ("ucb/pb_c_init", "$c_{\\text{init}}$"),
    ("reward/time_penalty", "$r_{\\text{time}}$"),
]

LEMMA_COLUMNS = [
    "Protocol",
    "Total",
    "ML (any)",
    "ML (best run)",
    "Tamarin (c)",
    "Tamarin (s)",
    "Original",
]
LEMMA_COLUMNS_BY_TYPE = [
    "Protocol",
    "Type",
    "Total",
    "ML (any)",
    "ML (best run)",
    "Tamarin (c)",
    "Tamarin (s)",
    "Original",
]

PROOFSIZE_COLUMNS = [
    "Protocol",
    "ML (any)",
    "Tamarin (c)",
    "Tamarin (s)",
    "Both ML",
    "Both Tamarin",
    "Original",
    "Both ML",
    "Both Original",
]

TIME_COLUMNS = [
    "Protocol",
    "ML",
    "Tamarin (c)",
    "Tamarin (s)",
    "Both ML",
    "Both Tamarin",
    "Original",
    "Both ML",
    "Both Original",
]

INPUT_FILES = {
    "lemma-completion": "lemma_completion_by_type.md",
    "proofsize": "proofsize_overview.md",
    "time": "protocol_time_minmax_overview.md",
}

OUTPUT_FILES = {
    "lemma-completion": "experiment1_table.tex",
    "proofsize": "proofsize_table.tex",
    "proofsize-full": "full_proofsize_table.tex",
    "time": "time_table.tex",
    "hyperparameters": "hyperparameters_table.tex",
    "proofsize-by-type": "proofsize_by_type.tex",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extract_caption(text: str) -> tuple[int, int] | None:
    """Return (start, end) of \\caption{...} with nested-brace matching."""
    idx = text.find("\\caption{")
    if idx == -1:
        return None
    depth = 0
    i = idx + len("\\caption")
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return (idx, i + 1)
        i += 1
    return None


def write_preserving_caption(content: str, output_path: Path) -> None:
    if output_path.exists():
        old = output_path.read_text(encoding="utf-8")
        old_span = _extract_caption(old)
        new_span = _extract_caption(content)
        if old_span and new_span:
            old_caption = old[old_span[0]:old_span[1]]
            content = content[:new_span[0]] + old_caption + content[new_span[1]:]
    output_path.write_text(content, encoding="utf-8")


def format_protocol_name(raw: str) -> str:
    key = normalize(raw)
    pretty = PRETTY_PROTOCOL_NAMES.get(key)
    if pretty:
        return pretty
    tokens = raw.replace("_", " ").replace("-", " ").split()
    acronyms = {
        "5g": "5G",
        "xn": "XN",
        "aka": "AKA",
        "spdm": "SPDM",
        "hsm": "HSM",
        "pfs": "PFS",
        "wpa2": "WPA2",
        "pkcs11": "PKCS11",
        "eck": "eCK",
    }
    return " ".join(acronyms.get(tok.lower(), tok.capitalize()) for tok in tokens)


def _format_protocol_cell(raw: str) -> str:
    return latex_escape(format_protocol_name(raw))


def _fallback_if_missing(value: str, fallback: str) -> str:
    missing = {"", "n/a", "na", "none", "null", "---", "-"}
    if value.strip().lower() in missing:
        return fallback
    return value


def _bold_smaller_pair(left: str, right: str) -> tuple[str, str]:
    try:
        left_num = float(left)
        right_num = float(right)
    except ValueError:
        return left, right
    if left_num < right_num:
        return f"\\textbf{{{left}}}", right
    if right_num < left_num:
        return left, f"\\textbf{{{right}}}"
    return left, right


def _parse_markdown_rows(
    markdown_path: Path, expected_columns: list[str]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    table_lines = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError("No markdown table found or table is too short.")
    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    if headers != expected_columns:
        raise ValueError(
            f"Unexpected header columns. Expected {expected_columns}, got {headers}."
        )
    for ln in table_lines[2:]:
        cells = [cell.strip() for cell in ln.strip("|").split("|")]
        if len(cells) != len(expected_columns):
            continue
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def assign_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = {name: [] for name, _ in GROUPS}
    alias_to_group: dict[str, str] = {}
    for name, aliases in GROUPS:
        for alias in aliases:
            alias_to_group[alias] = name
    unassigned: list[str] = []
    for row in rows:
        key = normalize(row["Protocol"])
        group = alias_to_group.get(key)
        if not group:
            unassigned.append(row["Protocol"])
            continue
        grouped[group].append(row)
    if unassigned:
        print(
            f"Warning: Skipping protocols not assigned to any group:"
            f" {', '.join(sorted(unassigned))}"
        )
    return grouped


# ---------------------------------------------------------------------------
# Lemma completion
# ---------------------------------------------------------------------------


def _format_lemma_type(raw: str) -> str:
    value = raw.strip().lower()
    if value == "forall":
        return r"$\forall$"
    if value == "exists":
        return r"$\exists$"
    return latex_escape(raw)


def _format_ml_value(ml_val: str, c_val: str, s_val: str, total: str = "") -> str:
    try:
        val_n = int(ml_val)
        baseline = max(int(c_val), int(s_val))
        total_n = int(total) if total else None
    except ValueError:
        return ml_val
    if val_n > baseline or (total_n is not None and val_n == total_n):
        return f"\\textbf{{{ml_val}}}"
    return ml_val


def parse_lemma_completion(markdown_path: Path) -> list[dict[str, str]]:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    table_lines = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError("No markdown table found or table is too short.")
    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    is_by_type = headers == LEMMA_COLUMNS_BY_TYPE
    if headers != LEMMA_COLUMNS and not is_by_type:
        raise ValueError(
            f"Unexpected columns. Expected {LEMMA_COLUMNS} or "
            f"{LEMMA_COLUMNS_BY_TYPE}, got {headers}."
        )
    expected_len = len(LEMMA_COLUMNS_BY_TYPE) if is_by_type else len(LEMMA_COLUMNS)
    rows: list[dict[str, str]] = []
    for ln in table_lines[2:]:
        cells = [cell.strip() for cell in ln.strip("|").split("|")]
        if len(cells) != expected_len:
            continue
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def write_lemma_completion(
    grouped: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    has_type = any("Type" in row for rows in grouped.values() for row in rows)
    f = io.StringIO()
    if has_type:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Overview of how many lemmas our RL system"
            " and Tamarin's\n"
            "\\z{c} and \\z{s} heuristics solve in each case study.\n"
            "The Total column shows the number of total lemmas in this"
            " category, with the \\emph{original} heuristics in"
            " parentheses when fewer than the total.}\n"
            "\\label{tab:lemma-completion}\n"
        )
        f.write("\\begin{tabular}{@{}l c c c c c@{}}\n")
        f.write("\\toprule\n")
        f.write(
            "Protocol & Type & Total & \\z{c} & \\z{s} & RL \\\\\n" "\\midrule\n"
        )
    else:
        f.write("\\begin{table*}[htbp]\n  \\centering\n")
        f.write(
            "  \\caption{Overview of how many lemmas our RL model, Tamarin's\n"
            "  \\z{c} and \\z{s} heuristics,\n"
            "  and the original heuristics solve in each case study.}\n"
            "  \\label{tab:lemma-completion}\n"
        )
        f.write("  \\begin{tabular}{@{}llrrrrrr@{}}\n    \\toprule\n")
        f.write(
            "    & & & & \\multicolumn{2}{c}{RL}"
            " & \\multicolumn{2}{c}{Tamarin} \\\\\n"
        )
        f.write("    \\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n")
        f.write(
            "    Category & Protocol & Total & Original"
            " & Any & Best Config & C & S \\\\\n"
            "    \\midrule\n"
        )

    groups_with_rows = [name for name, _ in GROUPS if grouped.get(name)]
    for group_idx, group_name in enumerate(groups_with_rows):
        group_rows = grouped[group_name]

        if has_type:
            group_label = latex_escape(group_name)
            f.write(f"\\multicolumn{{6}}{{c}}{{\\textbf{{{group_label}}}}} \\\\\n")
            f.write("\\midrule\n")

            protocol_rows: dict[str, list[dict[str, str]]] = {}
            for row in group_rows:
                protocol_rows.setdefault(row["Protocol"], []).append(row)

            protocol_items = list(protocol_rows.items())
            for pidx, (protocol_raw, rows_for_proto) in enumerate(protocol_items):
                protocol_cell = _format_protocol_cell(protocol_raw)
                type_values = {
                    r.get("Type", "").strip().lower() for r in rows_for_proto
                }
                has_both = {"forall", "exists"}.issubset(type_values)
                for ridx, row in enumerate(rows_for_proto):
                    if has_both:
                        pcol = (
                            f"\\multirow{{{len(rows_for_proto)}}}{{*}}"
                            f"{{{protocol_cell}}}"
                            if ridx == 0
                            else ""
                        )
                    else:
                        pcol = protocol_cell if ridx == 0 else ""
                    lemma_type = _format_lemma_type(row.get("Type", ""))
                    total = row["Total"]
                    original = row["Original"]
                    c_val = row["Tamarin (c)"]
                    s_val = _fallback_if_missing(row["Tamarin (s)"], original)
                    ml_best = row["ML (best run)"]
                    ml_fmt = _format_ml_value(ml_best, c_val, s_val, total)
                    try:
                        total_n = int(total)
                        orig_n = int(original)
                        total_cell = (
                            f"{total} ({original})" if orig_n < total_n else total
                        )
                    except ValueError:
                        total_cell = total
                    f.write(
                        f"{pcol} & {lemma_type} & {total_cell}"
                        f" & {c_val} & {s_val}"
                        f" & {ml_fmt} \\\\\n"
                    )
                if pidx < len(protocol_items) - 1:
                    f.write("\\lightrule\n")

            if group_idx < len(groups_with_rows) - 1:
                f.write("\\midrule\n")
        else:
            group_label = latex_escape(group_name)
            for idx, row in enumerate(group_rows):
                set_cell = (
                    f"\\multirow{{{len(group_rows)}}}{{*}}{{{group_label}}}"
                    if idx == 0
                    else ""
                )
                protocol = _format_protocol_cell(row["Protocol"])
                total = row["Total"]
                ml_any = row["ML (any)"]
                ml_best = row["ML (best run)"]
                c_val = row["Tamarin (c)"]
                original = row["Original"]
                s_val = _fallback_if_missing(row["Tamarin (s)"], original)
                f.write(
                    f"    {set_cell} & {protocol} & {total} & {original}"
                    f" & {ml_any} & {ml_best} & {c_val} & {s_val} \\\\\n"
                )
            f.write("    \\midrule\n")

    if has_type:
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    else:
        f.write("    \\bottomrule\n  \\end{tabular}\n")
        f.write("\\end{table*}\n")
    output_path.write_text(f.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Proof size
# ---------------------------------------------------------------------------


def parse_proofsize(markdown_path: Path) -> list[dict[str, str]]:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    table_lines = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError("No markdown table found or table is too short.")
    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    if headers != PROOFSIZE_COLUMNS:
        raise ValueError(
            f"Unexpected columns. Expected {PROOFSIZE_COLUMNS}, got {headers}."
        )
    rows: list[dict[str, str]] = []
    for ln in table_lines[2:]:
        cells = [cell.strip() for cell in ln.strip("|").split("|")]
        if len(cells) != len(PROOFSIZE_COLUMNS):
            continue
        rows.append(
            {
                "Protocol": cells[0],
                "ML (any)": cells[1],
                "Tamarin (c)": cells[2],
                "Tamarin (s)": cells[3],
                "Both ML (tam)": cells[4],
                "Both Tamarin": cells[5],
                "Original": cells[6],
                "Both ML (orig)": cells[7],
                "Both Original": cells[8],
            }
        )
    return rows


def write_proofsize_full(
    grouped: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    f = io.StringIO()
    f.write("\\begin{table*}[htbp]\n  \\centering\n")
    f.write(
        "  \\caption{ Average proof size for the lemmas solved by the"
        " different approaches, i.e., RL, \\z{c}, \\z{s}, and Orig."
        " Intersection columns restrict to lemmas solved by both"
        " approaches. Reports smallest proof size per lemma when"
        " \\z{c} or \\z{s} are aggregated together.}\n"
        "  \\label{tab:full-proofsize-overview}\n"
    )
    f.write("  \\begin{tabular}{@{}llrrrrrrrr@{}}\n    \\toprule\n")
    f.write(
        "    & & & \\multicolumn{3}{c}{Tamarin}"
        " & \\multicolumn{2}{c}{\\z{c}/\\z{s} $\\cap$ RL}"
        " & \\multicolumn{2}{c}{Orig.\\ $\\cap$ RL} \\\\\n"
    )
    f.write("    \\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n")
    f.write(
        "    Category & Protocol"
        " & \\multicolumn{1}{c}{RL}"
        " & \\multicolumn{1}{c}{\\z{c}}"
        " & \\multicolumn{1}{c}{\\z{s}}"
        " & \\multicolumn{1}{c}{Orig.}"
        " & \\multicolumn{1}{c}{\\z{c}/\\z{s}}"
        " & \\multicolumn{1}{c}{RL}"
        " & \\multicolumn{1}{c}{Orig.}"
        " & \\multicolumn{1}{c}{RL} \\\\\n"
        "    \\midrule\n"
    )

    groups_with_rows = [
        (name, grouped[name]) for name, _ in GROUPS if grouped.get(name)
    ]
    for group_idx, (group_name, group_rows) in enumerate(groups_with_rows):
        group_label = latex_escape(group_name)
        for idx, row in enumerate(group_rows):
            set_cell = (
                f"\\multirow{{{len(group_rows)}}}{{*}}{{{group_label}}}"
                if idx == 0
                else ""
            )
            protocol = _format_protocol_cell(row["Protocol"])
            ml_any = row["ML (any)"]
            c_val = row["Tamarin (c)"]
            original = row["Original"]
            s_val = _fallback_if_missing(row["Tamarin (s)"], original)
            both_ml_tam = row["Both ML (tam)"]
            both_tam = row["Both Tamarin"]
            both_ml_orig = row["Both ML (orig)"]
            both_orig = row["Both Original"]
            both_tam_fmt, both_ml_tam_fmt = _bold_smaller_pair(
                both_tam, both_ml_tam
            )
            both_orig_fmt, both_ml_orig_fmt = _bold_smaller_pair(
                both_orig, both_ml_orig
            )
            f.write(
                f"    {set_cell} & {protocol} & {ml_any} & {c_val}"
                f" & {s_val} & {original}"
                f" & {both_tam_fmt} & {both_ml_tam_fmt}"
                f" & {both_orig_fmt} & {both_ml_orig_fmt} \\\\\n"
            )
        if group_idx < len(groups_with_rows) - 1:
            f.write("    \\midrule\n")

    f.write("    \\bottomrule\n  \\end{tabular}\n")
    f.write("\\end{table*}\n")
    output_path.write_text(f.getvalue(), encoding="utf-8")


def write_proofsize_compact(
    grouped: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    f = io.StringIO()
    f.write("\\begin{table}[htbp]\n  \\centering\n")
    f.write(
        "  \\caption{Average proof size for the lemmas solved by both"
        " the respective baseline\n"
        "  and our approach (RL).}\n"
        "  \\label{tab:proofsize-overview}\n"
    )
    f.write("  \\begin{tabular}{@{}lrrrr@{}}\n    \\toprule\n")
    f.write(
        "    & \\multicolumn{2}{c}{\\z{c}/\\z{s} $\\cap$ RL}"
        " & \\multicolumn{2}{c}{Orig.\\ $\\cap$ RL} \\\\\n"
    )
    f.write("    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n")
    f.write(
        "    Protocol"
        " & \\multicolumn{1}{c}{\\z{c}/\\z{s}}"
        " & \\multicolumn{1}{c}{RL}"
        " & \\multicolumn{1}{c}{Orig.}"
        " & \\multicolumn{1}{c}{RL} \\\\\n"
        "    \\midrule\n"
    )

    groups_with_rows = [name for name, _ in GROUPS if grouped.get(name)]
    for group_idx, group_name in enumerate(groups_with_rows):
        group_rows = grouped[group_name]
        group_label = latex_escape(group_name)
        f.write(
            f"    \\multicolumn{{5}}{{c}}" f"{{\\textbf{{{group_label}}}}} \\\\\n"
        )
        f.write("    \\midrule\n")

        for row in group_rows:
            protocol = latex_escape(format_protocol_name(row["Protocol"]))
            both_ml_tam = row["Both ML (tam)"]
            both_tam = row["Both Tamarin"]
            both_ml_orig = row["Both ML (orig)"]
            both_orig = row["Both Original"]
            both_tam_fmt, both_ml_tam_fmt = _bold_smaller_pair(
                both_tam, both_ml_tam
            )
            both_orig_fmt, both_ml_orig_fmt = _bold_smaller_pair(
                both_orig, both_ml_orig
            )
            f.write(
                f"    {protocol} & {both_tam_fmt} & {both_ml_tam_fmt}"
                f" & {both_orig_fmt} & {both_ml_orig_fmt} \\\\\n"
            )

        if group_idx < len(groups_with_rows) - 1:
            f.write("    \\midrule\n")

    f.write("    \\bottomrule\n  \\end{tabular}\n")
    f.write("\\end{table}\n")
    output_path.write_text(f.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def _shorten_time(time_str: str) -> str:
    total = parse_time_to_seconds(time_str.strip())
    if total is None:
        return time_str.strip()
    if total >= 3600:
        total_min = int(round(total / 60))
        h, m = divmod(total_min, 60)
        return f"{h}h {m}m" if m else f"{h}h"
    if total >= 60:
        m = int(total // 60)
        s = int(round(total - m * 60))
        if s == 60:
            m += 1
            s = 0
        return f"{m}m {s}s" if s else f"{m}m"
    if total >= 10:
        return f"{int(round(total))}s"
    return time_str.strip()


def _bold_shorter_time(left: str, right: str) -> tuple[str, str]:
    left_sec = parse_time_to_seconds(left)
    right_sec = parse_time_to_seconds(right)
    if left_sec is None or right_sec is None:
        return left, right
    if left_sec < right_sec:
        return f"\\textbf{{{left}}}", right
    if right_sec < left_sec:
        return left, f"\\textbf{{{right}}}"
    return left, right


def parse_time(markdown_path: Path) -> list[dict[str, str]]:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    table_lines = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError("No markdown table found or table is too short.")
    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    if headers != TIME_COLUMNS:
        raise ValueError(f"Unexpected columns. Expected {TIME_COLUMNS}, got {headers}.")
    rows: list[dict[str, str]] = []
    for ln in table_lines[2:]:
        cells = [cell.strip() for cell in ln.strip("|").split("|")]
        if len(cells) != len(TIME_COLUMNS):
            continue
        rows.append(
            {
                "Protocol": cells[0],
                "ML": cells[1],
                "Tamarin (c)": cells[2],
                "Tamarin (s)": cells[3],
                "Both ML (tam)": cells[4],
                "Both Tamarin": cells[5],
                "Original": cells[6],
                "Both ML (orig)": cells[7],
                "Both Original": cells[8],
            }
        )
    return rows


def write_time(grouped: dict[str, list[dict[str, str]]], output_path: Path) -> None:
    f = io.StringIO()
    f.write("\\begin{table*}[htbp]\n  \\centering\n")
    f.write(
        "  \\caption{Wall-clock time to solve all lemmas of a protocol,"
        " assuming as much parallelization across lemmas"
        " as each approach allows.\n"
        "  RL times include both training and proof search.\n"
        "  Intersection columns pick the shortest time per lemma.\n"
        "  Times are rounded to integer seconds (${\\geq}10$s)"
        " or nearest minute (${\\geq}1$h).}\n"
        "  \\label{tab:protocol-time-minmax}\n"
    )
    f.write("  \\resizebox{\\linewidth}{!}{%\n")
    f.write("  \\begin{tabular}{@{}llrrrrrrrr@{}}\n    \\toprule\n")
    f.write(
        "    & & & \\multicolumn{3}{c}{Tamarin}"
        " & \\multicolumn{2}{c}{\\z{c}/\\z{s} $\\cap$ RL}"
        " & \\multicolumn{2}{c}{Orig.\\ $\\cap$ RL} \\\\\n"
    )
    f.write("    \\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n")
    f.write(
        "    Category & Protocol"
        " & \\multicolumn{1}{c}{RL}"
        " & \\multicolumn{1}{c}{\\z{c}}"
        " & \\multicolumn{1}{c}{\\z{s}}"
        " & \\multicolumn{1}{c}{Orig.}"
        " & \\multicolumn{1}{c}{\\z{c}/\\z{s}}"
        " & \\multicolumn{1}{c}{RL}"
        " & \\multicolumn{1}{c}{Orig.}"
        " & \\multicolumn{1}{c}{RL} \\\\\n"
        "    \\midrule\n"
    )

    groups_with_rows = [
        (name, grouped[name]) for name, _ in GROUPS if grouped.get(name)
    ]
    for group_idx, (group_name, group_rows) in enumerate(groups_with_rows):
        group_label = latex_escape(group_name)
        for idx, row in enumerate(group_rows):
            set_cell = (
                f"\\multirow{{{len(group_rows)}}}{{*}}{{{group_label}}}"
                if idx == 0
                else ""
            )
            protocol = _format_protocol_cell(row["Protocol"])
            ml = _shorten_time(row["ML"])
            c_val = _shorten_time(row["Tamarin (c)"])
            original = _shorten_time(row["Original"])
            s_val = _shorten_time(
                _fallback_if_missing(row["Tamarin (s)"], row["Original"])
            )
            both_ml_tam = _shorten_time(row["Both ML (tam)"])
            both_tam = _shorten_time(row["Both Tamarin"])
            both_ml_orig = _shorten_time(row["Both ML (orig)"])
            both_orig = _shorten_time(row["Both Original"])
            both_tam_fmt, both_ml_tam_fmt = _bold_shorter_time(
                both_tam, both_ml_tam
            )
            both_orig_fmt, both_ml_orig_fmt = _bold_shorter_time(
                both_orig, both_ml_orig
            )
            f.write(
                f"    {set_cell} & {protocol} & {ml} & {c_val} & {s_val}"
                f" & {original} & {both_tam_fmt} & {both_ml_tam_fmt}"
                f" & {both_orig_fmt} & {both_ml_orig_fmt} \\\\\n"
            )
        if group_idx < len(groups_with_rows) - 1:
            f.write("    \\midrule\n")

    f.write("    \\bottomrule\n  \\end{tabular}}\n")
    f.write("\\end{table*}\n")
    output_path.write_text(f.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------



def find_best_and_extra_runs(
    all_runs: dict[str, dict],
) -> list[tuple[str, str, dict, bool, int]]:
    protocols: dict[str, list[tuple[str, dict]]] = {}
    for run_id, run_data in all_runs.items():
        for key in run_data:
            if key == "_meta":
                continue
            protocols.setdefault(key, []).append((run_id, run_data[key]))

    results: list[tuple[str, str, dict, bool, int]] = []
    for proto, runs in protocols.items():
        best_id = None
        best_count = -1
        for run_id, lemmas in runs:
            completed = sum(1 for m in lemmas.values() if is_completed(m))
            if completed > best_count:
                best_count = completed
                best_id = run_id

        if best_id:
            best_solved = {
                l for l, m in all_runs[best_id][proto].items()
                if is_completed(m)
            }
            extra_id, extra_count = None, -1
            for run_id, lemmas in runs:
                if run_id == best_id:
                    continue
                solved = {
                    l for l, m in lemmas.items() if is_completed(m)
                }
                if solved - best_solved and len(solved) > extra_count:
                    extra_id, extra_count = run_id, len(solved)

            both_dagger = extra_id is not None and extra_count == best_count
            config = all_runs[best_id].get("_meta", {}).get("config", {})
            results.append((proto, best_id, config, both_dagger, best_count))
            if extra_id:
                cfg = all_runs[extra_id].get("_meta", {}).get("config", {})
                results.append((proto, extra_id, cfg, True, extra_count))

    def _sort_key(item: tuple[str, str, dict, bool, int]) -> tuple[int, int, str]:
        key = normalize(item[0])
        try:
            order = PROTOCOL_ORDER.index(key)
        except ValueError:
            order = 999
        return (order, int(item[3]), item[1])

    results.sort(key=_sort_key)
    return results


def _fmt_val(v: str) -> str:
    if v in ("", "None"):
        return "---"
    try:
        fv = float(v)
        if fv == int(fv) and fv >= 1:
            return str(int(fv))
    except ValueError:
        pass
    return v


def write_hyperparameters(
    results: list[tuple[str, str, dict, bool, int]], output_path: Path
) -> None:
    col_spec = "@{}ll r" + " r" * len(HYPERPARAM_KEYS) + "@{}"

    groups_for_results: list[tuple[str, list[tuple[str, str, dict, bool]]]] = []
    current_group = None
    for item in results:
        key = normalize(item[0])
        for group_name, aliases in GROUPS:
            if key in aliases:
                if group_name != current_group:
                    current_group = group_name
                    groups_for_results.append((group_name, []))
                groups_for_results[-1][1].append(item)
                break

    f = io.StringIO()
    f.write("\\begin{table*}[htbp]\n  \\centering\n")
    f.write(
        "\\caption{Hyperparameters per protocol. "
        "$B$: search budget, "
        "$B_{\\text{inc}}$: budget increase factor, "
        "$S$: number of searches per lemma, "
        "$N$: expansion breadth, LR: learning rate, "
        "$\\lambda$: heuristic influence, $\\delta$: unvisited penalty, "
        "Temp: UCB temperature, "
        "$c_{\\text{and}}$/$c_{\\text{base}}$/$c_{\\text{init}}$: "
        "UCB exploration constants, "
        "$r_{\\text{time}}$: time penalty. "
        "$\\dagger$: both runs solve the same amount of lemmas but each run solves one additional lemma that the other run does not solve.}\n"
        "\\label{tab:hyperparameters}\n"
    )
    f.write("  \\resizebox{\\linewidth}{!}{%\n")
    f.write(f"  \\begin{{tabular}}{{{col_spec}}}\n    \\toprule\n")

    param_headers = " & ".join(h for _, h in HYPERPARAM_KEYS)
    f.write(f"    Category & Protocol & Solved & {param_headers} \\\\\n")
    f.write("    \\midrule\n")

    for group_idx, (group_name, group_items) in enumerate(groups_for_results):
        group_label = latex_escape(group_name)
        for idx, (proto_raw, _run_id, config, is_extra, solved) in enumerate(
            group_items
        ):
            set_cell = (
                f"\\multirow{{{len(group_items)}}}{{*}}{{{group_label}}}"
                if idx == 0
                else ""
            )
            pretty = format_protocol_name(proto_raw)
            if is_extra:
                pretty = f"{pretty}$^\\dagger$"
            vals = " & ".join(
                _fmt_val(str(config.get(k, "")))
                for k, _ in HYPERPARAM_KEYS
            )
            f.write(f"    {set_cell} & {pretty} & {solved} & {vals} \\\\\n")
        if group_idx < len(groups_for_results) - 1:
            f.write("    \\midrule\n")

    f.write("    \\bottomrule\n  \\end{tabular}}\n")
    f.write("\\end{table*}\n")
    output_path.write_text(f.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown writers (counterparts of the LaTeX paper tables)
# ---------------------------------------------------------------------------


def write_lemma_completion_md(
    grouped: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    has_type = any("Type" in row for rows in grouped.values() for row in rows)
    f = io.StringIO()
    f.write("# Lemma Completion\n\n")
    if has_type:
        f.write("| Group | Protocol | Type | Total | c | s | RL |\n")
        f.write("|-------|----------|------|-------|---|---|----|\n")
        for group_name, _ in GROUPS:
            for row in grouped.get(group_name, []):
                proto = format_protocol_name(row["Protocol"])
                f.write(
                    f"| {group_name} | {proto} | {row.get('Type', '')}"
                    f" | {row['Total']} | {row['Tamarin (c)']}"
                    f" | {row['Tamarin (s)']} | {row['ML (best run)']} |\n"
                )
    else:
        f.write(
            "| Group | Protocol | Total | Original"
            " | ML (any) | ML (best) | c | s |\n"
        )
        f.write(
            "|-------|----------|-------|----------"
            "|----------|-----------|---|---|\n"
        )
        for group_name, _ in GROUPS:
            for row in grouped.get(group_name, []):
                proto = format_protocol_name(row["Protocol"])
                f.write(
                    f"| {group_name} | {proto} | {row['Total']}"
                    f" | {row['Original']} | {row['ML (any)']}"
                    f" | {row['ML (best run)']}"
                    f" | {row['Tamarin (c)']} | {row['Tamarin (s)']} |\n"
                )
    output_path.write_text(f.getvalue(), encoding="utf-8")


def write_proofsize_md(
    grouped: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    f = io.StringIO()
    f.write("# Proof Size\n\n")
    f.write(
        "| Group | Protocol | RL | c | s | Orig."
        " | c/s∩RL c/s | c/s∩RL RL | Orig.∩RL Orig. | Orig.∩RL RL |\n"
    )
    f.write(
        "|-------|----------|----|---|---|------"
        "|-----------|-----------|----------------|-------------|\n"
    )
    for group_name, _ in GROUPS:
        for row in grouped.get(group_name, []):
            proto = format_protocol_name(row["Protocol"])
            f.write(
                f"| {group_name} | {proto}"
                f" | {row['ML (any)']} | {row['Tamarin (c)']}"
                f" | {row['Tamarin (s)']} | {row['Original']}"
                f" | {row['Both Tamarin']} | {row['Both ML (tam)']}"
                f" | {row['Both Original']} | {row['Both ML (orig)']} |\n"
            )
    output_path.write_text(f.getvalue(), encoding="utf-8")


def write_time_md(
    grouped: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    f = io.StringIO()
    f.write("# Protocol Time\n\n")
    f.write(
        "| Group | Protocol | RL | c | s | Orig."
        " | c/s∩RL c/s | c/s∩RL RL | Orig.∩RL Orig. | Orig.∩RL RL |\n"
    )
    f.write(
        "|-------|----------|----|---|---|------"
        "|-----------|-----------|----------------|-------------|\n"
    )
    for group_name, _ in GROUPS:
        for row in grouped.get(group_name, []):
            proto = format_protocol_name(row["Protocol"])
            f.write(
                f"| {group_name} | {proto}"
                f" | {row['ML']} | {row['Tamarin (c)']}"
                f" | {row['Tamarin (s)']} | {row['Original']}"
                f" | {row['Both Tamarin']} | {row['Both ML (tam)']}"
                f" | {row['Both Original']} | {row['Both ML (orig)']} |\n"
            )
    output_path.write_text(f.getvalue(), encoding="utf-8")


def write_hyperparameters_md(
    results: list[tuple[str, str, dict, bool, int]], output_path: Path
) -> None:
    f = io.StringIO()
    f.write("# Hyperparameters\n\n")
    param_headers = " | ".join(label for _, label in HYPERPARAM_KEYS)
    f.write(f"| Group | Protocol | Solved | {param_headers} |\n")
    f.write(
        "|-------|----------|--------|" + "|".join("---" for _ in HYPERPARAM_KEYS) + "|\n"
    )
    for proto_raw, _run_id, config, is_extra, solved in results:
        key = normalize(proto_raw)
        group_name = ""
        for gn, aliases in GROUPS:
            if key in aliases:
                group_name = gn
                break
        pretty = format_protocol_name(proto_raw)
        if is_extra:
            pretty = f"{pretty} (extra)"
        vals = " | ".join(
            _fmt_val(str(config.get(k, ""))) for k, _ in HYPERPARAM_KEYS
        )
        f.write(f"| {group_name} | {pretty} | {solved} | {vals} |\n")
    output_path.write_text(f.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Combined standalone PDF
# ---------------------------------------------------------------------------

_COMBINED_PREAMBLE = r"""\documentclass{article}

\usepackage[a4paper, margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{graphicx}
\usepackage{adjustbox}
\usepackage{pdflscape}
\usepackage{amsmath}
\usepackage{multirow}
\usepackage{cleveref}

\newcommand{\z}[1]{\texttt{#1}}
\newcommand{\lightrule}{\midrule[\lightrulewidth]}

\title{Evaluation Tables}
\date{}

\begin{document}

\begin{landscape}
"""

_COMBINED_POSTAMBLE = r"""
\end{landscape}

\end{document}
"""

def build_combined_main_tex(tables_dir: str | Path) -> str:
    tables_dir = Path(tables_dir)
    skip = {"main.tex", "paper_main.tex"}
    inputs = []
    for tex_file in sorted(tables_dir.glob("*.tex")):
        if tex_file.name in skip:
            continue
        inputs.append(f"\\input{{{tex_file.stem}}}")
    body = "\n\n".join(inputs)
    return f"{_COMBINED_PREAMBLE}\n{body}\n{_COMBINED_POSTAMBLE}"


# ---------------------------------------------------------------------------
# Row builders (data → list[dict] for paper tables and markdown)
# ---------------------------------------------------------------------------


def _all_models(yaml_data: dict, csv_data: dict) -> list[str]:
    return sorted(set(yaml_data) | set(csv_data))


def _extract_solved_sizes(
    version_dict: dict, lemma_set: set[str] | None = None,
) -> list[int]:
    return [
        d["tree_size"]
        for ln, d in version_dict.items()
        if (lemma_set is None or ln in lemma_set)
        and is_solved(d["result"])
        and d.get("tree_size") not in (None, -1)
    ]


def _fmt_avg_size(sizes: list, fallback: str = "---") -> str:
    return f"{avg(sizes):.1f}" if sizes else fallback


def _fmt_time_or(t: float | None, fallback: str = "N/A") -> str:
    return fmt_seconds(t) if t is not None else fallback


def _compute_both_size(
    yl: dict, cv: dict, lemma_set: set[str] | None = None
) -> tuple[list, list, list, list]:
    tam_ml: list = []
    tam_tam: list = []
    orig_ml: list = []
    orig_orig: list = []
    for ln, ld in yl.items():
        if lemma_set is not None and ln not in lemma_set:
            continue
        if ld.get("min_time") in (None, "null"):
            continue
        ms = ld.get("min_tree_size")
        if ms is None:
            continue
        tam = [
            cv[ver][ln]["tree_size"]
            for ver in ("base_c", "base_s")
            if ver in cv
            and ln in cv[ver]
            and is_solved(cv[ver][ln]["result"])
            and cv[ver][ln].get("tree_size") not in (None, -1)
        ]
        if tam:
            tam_ml.append(ms)
            tam_tam.append(min(tam))
        if "original" in cv and ln in cv["original"]:
            if is_solved(cv["original"][ln]["result"]):
                sz = cv["original"][ln].get("tree_size")
                if sz not in (None, -1):
                    orig_ml.append(ms)
                    orig_orig.append(sz)
    return tam_ml, tam_tam, orig_ml, orig_orig


def _compute_protocol_time_minmax(
    yl: dict, cv: dict
) -> tuple[float | None, float | None, float | None,
           float | None, float | None, float | None,
           float | None, float | None]:
    ml_time: float | None = None
    for ld in yl.values():
        if ld.get("min_time") in (None, "null"):
            continue
        mt = parse_time_to_seconds(ld["min_time"])
        if mt is not None:
            ml_time = max(ml_time or 0.0, mt)

    c_t = [d["time"] for d in cv.get("base_c", {}).values() if is_solved(d["result"])]
    s_t = [d["time"] for d in cv.get("base_s", {}).values() if is_solved(d["result"])]
    orig_t = [d["time"] for d in cv.get("original", {}).values() if is_solved(d["result"])]

    tam_ml: float | None = None
    tam_tam: float | None = None
    orig_ml: float | None = None
    orig_orig: float | None = None

    for ln, ld in yl.items():
        if ld.get("min_time") in (None, "null"):
            continue
        mt = parse_time_to_seconds(ld["min_time"])
        if mt is None:
            continue
        tam = [
            cv[ver][ln]["time"]
            for ver in ("base_c", "base_s")
            if ver in cv and ln in cv[ver] and is_solved(cv[ver][ln]["result"])
        ]
        if tam:
            tam_ml = max(tam_ml or 0.0, mt)
            tam_tam = max(tam_tam or 0.0, min(tam))
        if "original" in cv and ln in cv["original"]:
            if is_solved(cv["original"][ln]["result"]):
                orig_ml = max(orig_ml or 0.0, mt)
                orig_orig = max(orig_orig or 0.0, cv["original"][ln]["time"])

    return (
        ml_time,
        max(c_t) if c_t else None,
        max(s_t) if s_t else None,
        tam_ml, tam_tam,
        max(orig_t) if orig_t else None,
        orig_ml, orig_orig,
    )


def _split_lemmas_by_type(
    yl: dict, cv: dict
) -> dict[str, dict[str, list[str]]]:
    type_map: dict[str, str] = {}
    for lemma, data in yl.items():
        type_map[lemma] = data.get("type", "unknown")

    for ver in ("base_c", "base_s", "original"):
        for lemma in cv.get(ver, {}):
            if lemma not in type_map:
                type_map[lemma] = "unknown"

    result: dict[str, dict[str, list[str]]] = {}
    for lemma, ltype in type_map.items():
        if ltype not in result:
            result[ltype] = {"ml": [], "base_c": [], "base_s": [], "original": []}
        if lemma in yl:
            result[ltype]["ml"].append(lemma)
        for ver in ("base_c", "base_s", "original"):
            if lemma in cv.get(ver, {}):
                result[ltype][ver].append(lemma)

    return result


def _best_run_completions_by_type(
    raw_data: dict, theory: str
) -> dict[str, int]:
    best_run_id = None
    best_completed = -1

    for run_id, run_data in raw_data.items():
        meta = run_data.get("_meta")
        if meta and "baseline" in meta.get("name", ""):
            continue
        if theory not in run_data:
            continue
        lemmas = run_data[theory]
        completed = sum(1 for m in lemmas.values() if is_completed(m))
        if completed > best_completed:
            best_completed = completed
            best_run_id = run_id

    by_type: dict[str, int] = {}
    if best_run_id is not None and theory in raw_data[best_run_id]:
        for lemma, metrics in raw_data[best_run_id][theory].items():
            if not isinstance(metrics, dict):
                continue
            ltype = metrics.get("type", "unknown")
            if ltype not in by_type:
                by_type[ltype] = 0
            if metrics.get("completes", 0) > 0:
                by_type[ltype] += 1

    return by_type


def _type_counts(
    by_type: dict, yl: dict, cv: dict, best_by_type: dict
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for ltype, groups in by_type.items():
        total = len(set(groups["ml"]) | set(groups["base_c"])
                    | set(groups["base_s"]) | set(groups["original"]))
        ml_any = sum(
            1 for ln in groups["ml"]
            if yl[ln].get("min_time") not in (None, "null")
        )
        ml_best = best_by_type.get(ltype, 0)
        c_s = sum(
            1 for ln in groups["base_c"]
            if is_solved(cv["base_c"][ln]["result"])
        )
        s_s = sum(
            1 for ln in groups["base_s"]
            if is_solved(cv["base_s"][ln]["result"])
        )
        o_s = sum(
            1 for ln in groups["original"]
            if is_solved(cv["original"][ln]["result"])
        )
        counts[ltype] = {
            "total": total, "ml_any": ml_any, "ml_best": ml_best,
            "c": c_s, "s": s_s, "orig": o_s,
        }
    return counts


def build_lemma_completion_by_type_rows(
    yaml_data: dict, csv_data: dict, raw_data: dict,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in sorted(set(yaml_data) | set(csv_data)):
        ym = yaml_data.get(model, {})
        cv = csv_data.get(model, {})
        yl = ym.get("lemmas", {})
        by_type = _split_lemmas_by_type(yl, cv)
        best_by_type = _best_run_completions_by_type(raw_data, model)
        counts = _type_counts(by_type, yl, cv, best_by_type)
        for ltype in sorted(counts.keys()):
            c = counts[ltype]
            type_label = (
                "exists" if ltype == "exists-trace"
                else "forall" if ltype == "all-traces"
                else ltype
            )
            rows.append({
                "Protocol": model, "Type": type_label,
                "Total": str(c["total"]), "ML (any)": str(c["ml_any"]),
                "ML (best run)": str(c["ml_best"]),
                "Tamarin (c)": str(c["c"]), "Tamarin (s)": str(c["s"]),
                "Original": str(c["orig"]),
            })
    return rows


def build_protocol_time_minmax_rows(
    yaml_data: dict, csv_data: dict,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in _all_models(yaml_data, csv_data):
        ym = yaml_data.get(model, {})
        cv = csv_data.get(model, {})
        yl = ym.get("lemmas", {})
        (ml, c, s, tam_ml, tam_tam, orig, orig_ml, orig_orig
         ) = _compute_protocol_time_minmax(yl, cv)
        rows.append({
            "Protocol": model, "ML": _fmt_time_or(ml),
            "Tamarin (c)": _fmt_time_or(c), "Tamarin (s)": _fmt_time_or(s),
            "Both ML (tam)": _fmt_time_or(tam_ml),
            "Both Tamarin": _fmt_time_or(tam_tam),
            "Original": _fmt_time_or(orig),
            "Both ML (orig)": _fmt_time_or(orig_ml),
            "Both Original": _fmt_time_or(orig_orig),
        })
    return rows


def build_proofsize_overview_rows(
    yaml_data: dict, csv_data: dict,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in _all_models(yaml_data, csv_data):
        ym = yaml_data.get(model, {})
        cv = csv_data.get(model, {})
        yl = ym.get("lemmas", {})
        ml_sz = [
            d["min_tree_size"]
            for d in yl.values()
            if d.get("min_time") not in (None, "null")
            and d.get("min_tree_size") is not None
        ]
        c_sz = _extract_solved_sizes(cv.get("base_c", {}))
        s_sz = _extract_solved_sizes(cv.get("base_s", {}))
        orig_sz = _extract_solved_sizes(cv.get("original", {}))
        tam_ml, tam_tam, orig_ml, orig_orig = _compute_both_size(yl, cv)
        rows.append({
            "Protocol": model,
            "ML (any)": _fmt_avg_size(ml_sz),
            "Tamarin (c)": _fmt_avg_size(c_sz),
            "Tamarin (s)": _fmt_avg_size(s_sz),
            "Both ML (tam)": _fmt_avg_size(tam_ml),
            "Both Tamarin": _fmt_avg_size(tam_tam),
            "Original": _fmt_avg_size(orig_sz),
            "Both ML (orig)": _fmt_avg_size(orig_ml),
            "Both Original": _fmt_avg_size(orig_orig),
        })
    return rows


def build_proofsize_by_type_rows(
    yaml_data: dict, csv_data: dict,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in _all_models(yaml_data, csv_data):
        ym = yaml_data.get(model, {})
        cv = csv_data.get(model, {})
        yl = ym.get("lemmas", {})
        by_type = _split_lemmas_by_type(yl, cv)
        for ltype in sorted(by_type.keys()):
            lemma_names = set(by_type[ltype]["ml"])
            ml_sz = [
                d["min_tree_size"]
                for ln, d in yl.items()
                if ln in lemma_names
                and d.get("min_time") not in (None, "null")
                and d.get("min_tree_size") is not None
            ]
            type_lemmas_c = set(by_type[ltype]["base_c"])
            type_lemmas_s = set(by_type[ltype]["base_s"])
            type_lemmas_orig = set(by_type[ltype]["original"])
            c_sz = _extract_solved_sizes(cv.get("base_c", {}), type_lemmas_c)
            s_sz = _extract_solved_sizes(cv.get("base_s", {}), type_lemmas_s)
            orig_sz = _extract_solved_sizes(cv.get("original", {}), type_lemmas_orig)
            all_type_lemmas = (
                set(by_type[ltype]["ml"])
                | set(by_type[ltype]["base_c"])
                | set(by_type[ltype]["base_s"])
                | set(by_type[ltype]["original"])
            )
            tam_ml, tam_tam, orig_ml, orig_orig = _compute_both_size(
                yl, cv, lemma_set=all_type_lemmas,
            )
            type_label = (
                "exists" if ltype == "exists-trace"
                else "forall" if ltype == "all-traces"
                else ltype
            )
            rows.append({
                "Protocol": model, "Type": type_label,
                "ML (any)": _fmt_avg_size(ml_sz),
                "Tamarin (c)": _fmt_avg_size(c_sz),
                "Tamarin (s)": _fmt_avg_size(s_sz),
                "Both ML (tam)": _fmt_avg_size(tam_ml),
                "Both Tamarin": _fmt_avg_size(tam_tam),
                "Original": _fmt_avg_size(orig_sz),
                "Both ML (orig)": _fmt_avg_size(orig_ml),
                "Both Original": _fmt_avg_size(orig_orig),
            })
    return rows




def write_proofsize_by_type(
    grouped: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    f = io.StringIO()
    f.write("\\begin{table*}[htbp]\n  \\centering\n")
    f.write(
        "  \\caption{ Average proof size for the lemmas solved by the"
        " different approaches, i.e., RL, \\z{c}, \\z{s}, and Orig."
        " Intersection columns restrict to lemmas solved by both"
        " approaches. Reports smallest proof size per lemma when"
        " \\z{c} or \\z{s} are aggregated together."
        " $\\exists$: exists-trace; $\\forall$: all-traces.}\n"
        "  \\label{tab:proofsize-by-type}\n"
    )
    f.write("  \\begin{tabular}{@{}lllrrrrrrrr@{}}\n    \\toprule\n")
    f.write(
        "    & & & & \\multicolumn{3}{c}{Tamarin}"
        " & \\multicolumn{2}{c}{\\z{c}/\\z{s} $\\cap$ RL}"
        " & \\multicolumn{2}{c}{Orig.\\ $\\cap$ RL} \\\\\n"
    )
    f.write(
        "    \\cmidrule(lr){5-7}\\cmidrule(lr){8-9}"
        "\\cmidrule(lr){10-11}\n"
    )
    f.write(
        "    Category & Protocol & Type"
        " & \\multicolumn{1}{c}{RL}"
        " & \\multicolumn{1}{c}{\\z{c}}"
        " & \\multicolumn{1}{c}{\\z{s}}"
        " & \\multicolumn{1}{c}{Orig.}"
        " & \\multicolumn{1}{c}{\\z{c}/\\z{s}}"
        " & \\multicolumn{1}{c}{RL}"
        " & \\multicolumn{1}{c}{Orig.}"
        " & \\multicolumn{1}{c}{RL} \\\\\n"
        "    \\midrule\n"
    )

    groups_with_rows = [
        (name, grouped[name]) for name, _ in GROUPS if grouped.get(name)
    ]
    for group_idx, (group_name, group_rows) in enumerate(groups_with_rows):
        group_label = latex_escape(group_name)
        prev_protocol = None
        for idx, row in enumerate(group_rows):
            set_cell = (
                f"\\multirow{{{len(group_rows)}}}{{*}}{{{group_label}}}"
                if idx == 0
                else ""
            )
            cur_proto = row["Protocol"]
            if prev_protocol is not None and cur_proto != prev_protocol:
                f.write("    \\cmidrule(lr){2-11}\n")
            prev_protocol = cur_proto

            protocol = _format_protocol_cell(cur_proto)
            type_label = _format_lemma_type(row["Type"])
            ml_any = row["ML (any)"]
            c_val = row["Tamarin (c)"]
            s_val = row["Tamarin (s)"]
            original = row["Original"]
            both_tam_fmt, both_ml_tam_fmt = _bold_smaller_pair(
                row["Both Tamarin"], row["Both ML (tam)"]
            )
            both_orig_fmt, both_ml_orig_fmt = _bold_smaller_pair(
                row["Both Original"], row["Both ML (orig)"]
            )
            f.write(
                f"    {set_cell} & {protocol} & {type_label}"
                f" & {ml_any} & {c_val} & {s_val}"
                f" & {original}"
                f" & {both_tam_fmt} & {both_ml_tam_fmt}"
                f" & {both_orig_fmt} & {both_ml_orig_fmt}"
                f" \\\\\n"
            )
        if group_idx < len(groups_with_rows) - 1:
            f.write("    \\midrule\n")

    f.write("    \\bottomrule\n  \\end{tabular}\n")
    f.write("\\end{table*}\n")
    output_path.write_text(f.getvalue(), encoding="utf-8")


def generate_paper_tables(
    all_yaml: dict, all_raw: dict, csv_data: dict, tables_dir: Path,
    best_raw: dict | None = None,
) -> None:
    """Write paper-quality tex + md tables and compile to PDF."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    lc_rows = build_lemma_completion_by_type_rows(all_yaml, csv_data, all_raw)
    if lc_rows:
        grouped = assign_groups(lc_rows)
        write_lemma_completion(
            grouped, tables_dir / OUTPUT_FILES["lemma-completion"],
        )
        print(f"  Wrote {tables_dir / OUTPUT_FILES['lemma-completion']}")
        write_lemma_completion_md(
            grouped,
            tables_dir / OUTPUT_FILES["lemma-completion"].replace(
                ".tex", ".md",
            ),
        )

    ps_rows = build_proofsize_overview_rows(all_yaml, csv_data)
    if ps_rows:
        grouped = assign_groups(ps_rows)
        write_proofsize_compact(
            grouped, tables_dir / OUTPUT_FILES["proofsize"],
        )
        write_proofsize_full(
            grouped, tables_dir / OUTPUT_FILES["proofsize-full"],
        )
        print(f"  Wrote {tables_dir / OUTPUT_FILES['proofsize']}")
        print(f"  Wrote {tables_dir / OUTPUT_FILES['proofsize-full']}")
        write_proofsize_md(
            grouped,
            tables_dir / OUTPUT_FILES["proofsize"].replace(".tex", ".md"),
        )

    ps_type_rows = build_proofsize_by_type_rows(all_yaml, csv_data)
    if ps_type_rows:
        grouped = assign_groups(ps_type_rows)
        write_proofsize_by_type(
            grouped, tables_dir / OUTPUT_FILES["proofsize-by-type"],
        )
        print(
            f"  Wrote {tables_dir / OUTPUT_FILES['proofsize-by-type']}"
        )

    time_rows = build_protocol_time_minmax_rows(all_yaml, csv_data)
    if time_rows:
        grouped = assign_groups(time_rows)
        write_time(grouped, tables_dir / OUTPUT_FILES["time"])
        print(f"  Wrote {tables_dir / OUTPUT_FILES['time']}")
        write_time_md(
            grouped,
            tables_dir / OUTPUT_FILES["time"].replace(".tex", ".md"),
        )

    hp_raw = best_raw if best_raw is not None else all_raw
    if hp_raw:
        results = find_best_and_extra_runs(hp_raw)
        write_hyperparameters(
            results, tables_dir / OUTPUT_FILES["hyperparameters"],
        )
        print(f"  Wrote {tables_dir / OUTPUT_FILES['hyperparameters']}")
        write_hyperparameters_md(
            results,
            tables_dir / OUTPUT_FILES["hyperparameters"].replace(
                ".tex", ".md",
            ),
        )

    paper_main = tables_dir / "main.tex"
    paper_main.write_text(build_combined_main_tex(tables_dir))
    if shutil.which("pdflatex"):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=tables_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"  Paper tables PDF -> {tables_dir / 'main.pdf'}")
        else:
            print(f"  pdflatex failed (see {tables_dir / 'main.log'})")
