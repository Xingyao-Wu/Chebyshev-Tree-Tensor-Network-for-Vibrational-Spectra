import argparse
import math
import re
from typing import List, Optional, Tuple
from pathlib import Path

def _is_int(text: str) -> bool:
    try:
        int(text)
        return True
    except Exception:
        return False


def _extract_numeric(token: str) -> Tuple[float, str]:
    """
    Return (numeric_value, original_display_text_without_macro).
    If the token already contains \\best{...}, unwrap it first.
    """
    token_stripped = token.strip()
    best_match = re.fullmatch(r"\\best\{(.+)\}", token_stripped)
    inner = best_match.group(1) if best_match else token_stripped
    # Keep original text (without macro) for re-wrapping
    original_display = inner
    # Extract number (could be like -0.000)
    try:
        value = float(inner)
    except ValueError:
        # In case of unexpected formatting, try to find a number substring
        num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", inner)
        if not num_match:
            raise
        value = float(num_match.group(0))
        original_display = num_match.group(0)
    return value, original_display


def _wrap_best(token_display: str, annotate_macro: str) -> str:
    return f"{annotate_macro}{{{token_display}}}"


def _process_data_row_tokens(
    tokens: List[str],
    comparison: str,
    annotate_macro: str,
    num_last_columns: int,
) -> List[str]:
    """
    tokens: already stripped tokens split by '&'
    We expect tokens layout:
      [index, ref_energy, col3, col4, col5, col6, col7, col8]
    """
    if len(tokens) < 2 + num_last_columns:
        return tokens

    data_start_idx = len(tokens) - num_last_columns
    metrics: List[Tuple[float, float]] = []  # (metric, value)
    displays: List[str] = []

    for i in range(data_start_idx, len(tokens)):
        value, display = _extract_numeric(tokens[i])
        metric = abs(value) if comparison == "abs" else value
        metrics.append((metric, value))
        displays.append(display)

    # Determine minima (support ties using isclose)
    metric_values = [m for (m, _) in metrics]
    min_metric = min(metric_values)
    minima_indices = [
        i for i, m in enumerate(metric_values) if math.isclose(m, min_metric, rel_tol=1e-12, abs_tol=1e-12)
    ]

    # Apply wrapping to minima
    for local_idx, display in enumerate(displays):
        global_idx = data_start_idx + local_idx
        if local_idx in minima_indices:
            tokens[global_idx] = _wrap_best(display, annotate_macro)
        else:
            tokens[global_idx] = display

    return tokens


def annotate_best_in_table(
    input_path: str,
    output_path: Optional[str] = None,
    comparison: str = "abs",
    num_last_columns: int = 6,
    annotate_macro: str = r"\best",
) -> Optional[str]:
    """
    Annotate the smallest entries in the last `num_last_columns` columns of each data row by wrapping
    them with `annotate_macro{...}`.

    - comparison: "abs" (default) compares by absolute value, "raw" compares by raw numeric value
    - If output_path is None, returns the annotated content as a string
    """
    if comparison not in {"abs", "raw"}:
        raise ValueError("comparison must be 'abs' or 'raw'")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    annotated_lines: List[str] = []
    for line in lines:
        # Only consider rows that look like data rows with '&'
        if "&" not in line:
            annotated_lines.append(line)
            continue

        # Preserve leading indentation
        indent_match = re.match(r"^(\s*)", line)
        indent = indent_match.group(1) if indent_match else ""

        # Detect if this row had trailing '\\'
        backslashes_match = re.search(r"(\\+)\s*$", line)
        trailing_backslashes = backslashes_match.group(1) if backslashes_match else ""

        # Split and strip tokens
        tokens = [t.strip() for t in line.strip().rstrip("\\").split("&")]
        if len(tokens) == 0:
            annotated_lines.append(line)
            continue

        # Identify data rows: first token should be an integer index
        first_token = tokens[0]
        if not _is_int(first_token):
            annotated_lines.append(line)
            continue

        # Process tokens: wrap minima among the last num_last_columns
        tokens = _process_data_row_tokens(
            tokens=tokens,
            comparison=comparison,
            annotate_macro=annotate_macro,
            num_last_columns=num_last_columns,
        )

        # Rebuild the line, preserve indent and whether it had trailing '\\'
        rebuilt = indent + " & ".join(tokens)
        if trailing_backslashes:
            rebuilt += f" {trailing_backslashes}\n"
        else:
            rebuilt += "\n"
        annotated_lines.append(rebuilt)

    result_text = "".join(annotated_lines)
    if output_path is None:
        return result_text

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    return None


def main() -> None:
    SCRIPT_DIR = Path(__file__).resolve().parent
    input_path = SCRIPT_DIR / 'table_ii.txt'
    output_path = SCRIPT_DIR / 'table_ii_annotated.tex'
    annotate_best_in_table(
        input_path=input_path,
        output_path=output_path,
        comparison="abs",
        num_last_columns=6,
        annotate_macro=r"\best",
    )

if __name__ == "__main__":
    main()


