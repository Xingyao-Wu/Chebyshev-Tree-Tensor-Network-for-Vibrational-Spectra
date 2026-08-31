from __future__ import annotations

import pathlib
import re
import math
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

from Experiment.utils import get_harmonic_oscillator_orbitals, single_voxel_block_diag
from pytreenet.core.node import Node
from pytreenet.random.random_node import random_tensor_node
from pytreenet.ttns import TTNS
from pytreenet.util.tensor_splitting import SplitMode


CH3CN_DEFAULT_N = [9, 7, 9, 9, 9, 9, 7, 7, 9, 9, 27, 27]
CH3CN_DEFAULT_NODE_ORDER = list(range(12))
CH3CN_ORCA_VIB_OUT = pathlib.Path(__file__).parent / "ch3cn" / "vib.out"
CH3CN_ORCA_VPT2 = pathlib.Path(__file__).parent / "ch3cn" / "vib.vpt2"
CH3CN_ORCA_HESS = pathlib.Path(__file__).parent / "ch3cn" / "vib.hess"
CH3CN_ORCA_OPT_OUT = pathlib.Path(__file__).parent / "ch3cn" / "opt.out"

# Site order:
# [nu1, nu2, nu3, nu4, nu5a, nu5b, nu6a, nu6b, nu7a, nu7b, nu8a, nu8b]
#
# ORCA's final VPT2 IR table is in ascending-frequency vibrational order.
# This maps the site order above to ORCA's zero-based vibrational mode indices.
CH3CN_SITE_TO_ORCA_MODES = [9, 8, 5, 2, 10, 11, 6, 7, 3, 4, 0, 1]


def _resolve_orca_vib_out(path: pathlib.Path | str) -> pathlib.Path:
    p = pathlib.Path(path)
    candidates = []
    if p.is_dir():
        candidates.append(p / "vib.out")
    else:
        candidates.extend([p, p.with_suffix(".out"), p.parent / f"{p.name}.out"])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate ORCA vib.out. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _resolve_orca_vpt2(path: pathlib.Path | str = CH3CN_ORCA_VPT2) -> pathlib.Path:
    p = pathlib.Path(path)
    candidates = []
    if p.is_dir():
        candidates.append(p / "vib.vpt2")
    else:
        candidates.extend([p, p.with_suffix(".vpt2"), p.parent / f"{p.name}.vpt2"])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate ORCA vib.vpt2. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _resolve_orca_hess(path: pathlib.Path | str = CH3CN_ORCA_HESS) -> pathlib.Path:
    p = pathlib.Path(path)
    candidates = []
    if p.is_dir():
        candidates.append(p / "vib.hess")
    else:
        candidates.extend([p, p.with_suffix(".hess"), p.parent / f"{p.name}.hess"])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate ORCA vib.hess. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _resolve_orca_opt_out(path: pathlib.Path | str = CH3CN_ORCA_OPT_OUT) -> pathlib.Path:
    p = pathlib.Path(path)
    candidates = []
    if p.is_dir():
        candidates.append(p / "opt.out")
    else:
        candidates.extend([p, p.with_suffix(".out"), p.parent / f"{p.name}.out"])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate ORCA opt.out. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def get_ch3cn_orca_permanent_dipole(
    path: pathlib.Path | str = CH3CN_ORCA_OPT_OUT,
) -> np.ndarray:
    """
    Return the Cartesian permanent dipole [x, y, z] from ORCA opt.out.
    """
    opt_out = _resolve_orca_opt_out(path)
    line_re = re.compile(
        r"^\s*Total Dipole Moment\s*:\s+"
        r"(?P<x>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
        r"(?P<y>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
        r"(?P<z>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$"
    )
    for line in opt_out.read_text(encoding="utf-8").splitlines():
        match = line_re.match(line)
        if match is not None:
            return np.array(
                [float(match.group("x")), float(match.group("y")), float(match.group("z"))],
                dtype=np.float64,
            )
    raise ValueError(f"Could not find Total Dipole Moment in {opt_out}.")


def _parse_final_orca_ir_rows(path: pathlib.Path | str = CH3CN_ORCA_VIB_OUT) -> list[dict[str, float]]:
    vib_out = _resolve_orca_vib_out(path)
    lines = vib_out.read_text(encoding="utf-8").splitlines()

    fermi_idx = next(
        (i for i, line in enumerate(lines) if "Analysis of possible Fermi resonances" in line),
        len(lines),
    )
    ir_header_indices = [
        i for i, line in enumerate(lines[:fermi_idx])
        if "IR Intensities" in line
    ]
    if not ir_header_indices:
        raise ValueError(f"Could not find an IR Intensities block in {vib_out}.")

    start_idx = ir_header_indices[-1]
    row_re = re.compile(
        r"^\s*(?P<mode>\d+)\s+"
        r"(?P<freq>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
        r"(?P<intensity>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
        r"(?P<t2>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
        r"\(\s*"
        r"(?P<tx>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
        r"(?P<ty>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
        r"(?P<tz>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*"
        r"\)\s*$"
    )

    rows = []
    for line in lines[start_idx:fermi_idx]:
        match = row_re.match(line)
        if match is None:
            continue
        rows.append({
            "raw_mode": int(match.group("mode")),
            "freq_cm": float(match.group("freq")),
            "intensity_km_mol": float(match.group("intensity")),
            "t2": float(match.group("t2")),
            "tx": float(match.group("tx")),
            "ty": float(match.group("ty")),
            "tz": float(match.group("tz")),
        })

    if len(rows) < 12:
        raise ValueError(f"Expected at least 12 numeric IR rows in {vib_out}, found {len(rows)}.")
    return rows


def _parse_orca_fundamental_rows(
    path: pathlib.Path | str = CH3CN_ORCA_VIB_OUT,
) -> list[dict[str, float]]:
    vib_out = _resolve_orca_vib_out(path)
    lines = vib_out.read_text(encoding="utf-8").splitlines()

    start_idx = next(
        (i for i, line in enumerate(lines) if "Fundamental transitions [1/cm]" in line),
        None,
    )
    if start_idx is None:
        raise ValueError(f"Could not find the Fundamental transitions block in {vib_out}.")

    row_re = re.compile(
        r"^\s*(?P<mode>\d+)\s+"
        r"(?P<harm>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
        r"(?P<fund>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s+"
        r"(?P<diff>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$"
    )

    rows = []
    for line in lines[start_idx:]:
        if rows and line.strip().startswith("-"):
            break
        match = row_re.match(line)
        if match is None:
            continue
        rows.append({
            "orca_mode": int(match.group("mode")),
            "harmonic_freq_cm": float(match.group("harm")),
            "fundamental_freq_cm": float(match.group("fund")),
            "diff_cm": float(match.group("diff")),
        })

    if len(rows) < 12:
        raise ValueError(
            f"Expected at least 12 fundamental transition rows in {vib_out}, found {len(rows)}."
        )
    return rows[:12]


def _parse_orca_vpt2_tensor_block(
    path: pathlib.Path | str,
    header: str,
    rank: int,
) -> np.ndarray:
    vpt2 = _resolve_orca_vpt2(path)
    lines = vpt2.read_text(encoding="utf-8").splitlines()
    start_idx = next((i for i, line in enumerate(lines) if header in line), None)
    if start_idx is None:
        raise ValueError(f"Could not find {header!r} in {vpt2}.")

    dims = tuple(int(value) for value in lines[start_idx + 1].split())
    if len(dims) != rank:
        raise ValueError(f"Expected {rank} dimensions after {header!r}, got {dims}.")

    tensor = np.zeros(dims, dtype=np.float64)
    row_re = re.compile(
        r"^\s*"
        + r"\s+".join([r"(\d+)"] * rank)
        + r"\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*$"
    )
    for line in lines[start_idx + 2:]:
        if line.startswith("#"):
            break
        match = row_re.match(line)
        if match is None:
            if line.strip():
                break
            continue
        indices = tuple(int(match.group(i + 1)) for i in range(rank))
        tensor[indices] = float(match.group(rank + 1))

    return tensor


def _parse_orca_hess_matrix_block(
    path: pathlib.Path | str,
    marker: str,
) -> np.ndarray:
    hess = _resolve_orca_hess(path)
    lines = hess.read_text(encoding="utf-8").splitlines()
    start_idx = next((i for i, line in enumerate(lines) if line.strip() == marker), None)
    if start_idx is None:
        raise ValueError(f"Could not find {marker!r} in {hess}.")

    dims = tuple(int(value) for value in lines[start_idx + 1].split())
    if len(dims) == 1:
        nrows = ncols = dims[0]
    elif len(dims) == 2:
        nrows, ncols = dims
    else:
        raise ValueError(f"Expected one or two dimensions after {marker!r}, got {dims}.")

    matrix = np.zeros((nrows, ncols), dtype=np.float64)
    int_re = re.compile(r"^\d+$")
    i = start_idx + 2
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("$") or stripped.startswith("#"):
            break
        if not stripped:
            i += 1
            continue

        parts = stripped.split()
        if all(int_re.fullmatch(part) for part in parts):
            columns = [int(part) for part in parts]
            i += 1
            while i < len(lines):
                row_stripped = lines[i].strip()
                if not row_stripped:
                    break
                if row_stripped.startswith("$") or row_stripped.startswith("#"):
                    return matrix

                row_parts = row_stripped.split()
                if all(int_re.fullmatch(part) for part in row_parts):
                    break

                row = int(row_parts[0])
                values = [float(value) for value in row_parts[1:]]
                for column, value in zip(columns, values):
                    matrix[row, column] = value
                i += 1
            continue

        i += 1

    return matrix


def _polynomial_denominator(indices: Sequence[int]) -> int:
    counts = Counter(int(index) for index in indices)
    denom = 1
    for count in counts.values():
        denom *= math.factorial(count)
    return denom


def get_ch3cn_orca_ir_table(
    path: pathlib.Path | str = CH3CN_ORCA_VIB_OUT,
    map_modes: Sequence[int] = CH3CN_SITE_TO_ORCA_MODES,
) -> dict[str, np.ndarray]:
    """
    Return CH3CN ORCA IR data in site order.

    The returned arrays follow:
        [nu1, nu2, nu3, nu4, nu5a, nu5b, nu6a, nu6b, nu7a, nu7b, nu8a, nu8b]
    """
    mapping = [int(i) for i in map_modes]
    if len(mapping) != 12 or sorted(mapping) != list(range(12)):
        raise ValueError("map_modes must be a permutation of 0..11.")

    # The final VPT2 IR block contains 6 translation/rotation rows followed by
    # the 12 vibrational modes. Use the last 12 numeric rows to avoid relying
    # on ORCA's raw row labels.
    vibrational_rows = _parse_final_orca_ir_rows(path)[-12:]
    rows = [vibrational_rows[i] for i in mapping]

    return {
        "orca_mode": np.array(mapping, dtype=np.int64),
        "raw_mode": np.array([row["raw_mode"] for row in rows], dtype=np.int64),
        "freq_cm": np.array([row["freq_cm"] for row in rows], dtype=np.float64),
        "intensity_km_mol": np.array([row["intensity_km_mol"] for row in rows], dtype=np.float64),
        "t2": np.array([row["t2"] for row in rows], dtype=np.float64),
        "tx": np.array([row["tx"] for row in rows], dtype=np.float64),
        "ty": np.array([row["ty"] for row in rows], dtype=np.float64),
        "tz": np.array([row["tz"] for row in rows], dtype=np.float64),
    }


CH3CN_ORCA_IR_TABLE = get_ch3cn_orca_ir_table()
CH3CN_IR_FREQ_CM_12 = CH3CN_ORCA_IR_TABLE["freq_cm"].copy()
CH3CN_IR_INTENSITIES_12 = CH3CN_ORCA_IR_TABLE["intensity_km_mol"].copy()
CH3CN_IR_T2_12 = CH3CN_ORCA_IR_TABLE["t2"].copy()
CH3CN_IR_TRANSITION_DIPOLE_12 = np.column_stack(
    (
        CH3CN_ORCA_IR_TABLE["tx"],
        CH3CN_ORCA_IR_TABLE["ty"],
        CH3CN_ORCA_IR_TABLE["tz"],
    )
)


def get_ch3cn_orca_fundamental_table(
    path: pathlib.Path | str = CH3CN_ORCA_VIB_OUT,
    map_modes: Sequence[int] = CH3CN_SITE_TO_ORCA_MODES,
) -> dict[str, np.ndarray]:
    """
    Return ORCA VPT2 fundamental transition frequencies in site order.
    """
    mapping = [int(i) for i in map_modes]
    if len(mapping) != 12 or sorted(mapping) != list(range(12)):
        raise ValueError("map_modes must be a permutation of 0..11.")

    rows_orca_order = _parse_orca_fundamental_rows(path)
    rows = [rows_orca_order[i] for i in mapping]
    return {
        "orca_mode": np.array(mapping, dtype=np.int64),
        "harmonic_freq_cm": np.array([row["harmonic_freq_cm"] for row in rows], dtype=np.float64),
        "fundamental_freq_cm": np.array([row["fundamental_freq_cm"] for row in rows], dtype=np.float64),
        "diff_cm": np.array([row["diff_cm"] for row in rows], dtype=np.float64),
    }


CH3CN_ORCA_FUNDAMENTAL_TABLE = get_ch3cn_orca_fundamental_table()
CH3CN_ORCA_FUNDAMENTAL_FREQ_CM_12 = CH3CN_ORCA_FUNDAMENTAL_TABLE[
    "fundamental_freq_cm"
].copy()


def get_ch3cn_orca_vpt2_force_field(
    path: pathlib.Path | str = CH3CN_ORCA_VPT2,
    map_modes: Sequence[int] = CH3CN_SITE_TO_ORCA_MODES,
    force_constant_cutoff_cm: float = 0.0,
    as_polynomial_coefficients: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ORCA cubic and semi-quartic force fields in CH3CN site order.

    ORCA's VPT2 file prints a full symmetric cubic force field and a
    semi-diagonal quartic force field F[i,j,k,k].  When
    as_polynomial_coefficients is True, the returned tensors contain one
    canonical monomial coefficient per sorted index tuple, matching the
    convention used by Experiment.utils.get_name_list.
    """
    mapping = [int(i) for i in map_modes]
    if len(mapping) != 12 or sorted(mapping) != list(range(12)):
        raise ValueError("map_modes must be a permutation of 0..11.")

    cubic_orca = _parse_orca_vpt2_tensor_block(path, "Cubic[i][j][k]", rank=3)
    semi_quartic_orca = _parse_orca_vpt2_tensor_block(path, "Semi-quartic[i][j][k][k]", rank=3)

    cutoff = float(force_constant_cutoff_cm)
    k3 = np.zeros((12, 12, 12), dtype=np.float64)
    for i in range(12):
        for j in range(i, 12):
            for k in range(j, 12):
                force_constant = cubic_orca[mapping[i], mapping[j], mapping[k]]
                coeff = force_constant
                if as_polynomial_coefficients:
                    coeff = coeff / _polynomial_denominator((i, j, k))
                if abs(coeff) > cutoff:
                    k3[i, j, k] = coeff

    inverse_mapping = {orca_mode: site for site, orca_mode in enumerate(mapping)}
    quartic_buckets: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    for oi in range(12):
        for oj in range(12):
            for ok in range(12):
                site_i = inverse_mapping[oi]
                site_j = inverse_mapping[oj]
                site_k = inverse_mapping[ok]
                key = tuple(sorted((site_i, site_j, site_k, site_k)))
                quartic_buckets[key].append(semi_quartic_orca[oi, oj, ok])

    k4 = np.zeros((12, 12, 12, 12), dtype=np.float64)
    for key, values in quartic_buckets.items():
        force_constant = float(np.mean(values))
        coeff = force_constant
        if as_polynomial_coefficients:
            coeff = coeff / _polynomial_denominator(key)
        if abs(coeff) > cutoff:
            k4[key] = coeff

    return k3, k4


def get_ch3cn_orca_normal_modes(
    path: pathlib.Path | str = CH3CN_ORCA_HESS,
    map_modes: Sequence[int] = CH3CN_SITE_TO_ORCA_MODES,
    vibrational_start: int = 6,
) -> np.ndarray:
    """
    Return ORCA normal-mode vectors in CH3CN site order.

    The returned array has shape ``(3N, 12)``.  Columns follow the CH3CN site
    order used throughout this file.  ORCA prints the six translation/rotation
    columns before the vibrational columns, so ``vibrational_start`` defaults
    to 6.
    """
    mapping = [int(i) for i in map_modes]
    if len(mapping) != 12 or sorted(mapping) != list(range(12)):
        raise ValueError("map_modes must be a permutation of 0..11.")

    normal_modes = _parse_orca_hess_matrix_block(path, "$normal_modes")
    mode_columns = [vibrational_start + mode for mode in mapping]
    if max(mode_columns) >= normal_modes.shape[1]:
        raise ValueError(
            f"Normal-mode matrix has only {normal_modes.shape[1]} columns; "
            f"cannot select columns {mode_columns}."
        )
    return normal_modes[:, mode_columns].copy()


def get_ch3cn_orca_vpt2_dipole_derivative_blocks(
    path: pathlib.Path | str = CH3CN_ORCA_VPT2,
) -> dict[str, np.ndarray]:
    """
    Return raw dipole derivative blocks from ORCA's ``vib.vpt2`` file.

    The raw high-order blocks are mixed-coordinate arrays printed by ORCA:

    - ``linear_cartesian``: ``(3N, xyz)``
    - ``second_mixed``: ``(NVib, 3N, xyz)``
    - ``third_mixed``: ``(NVib, 3N, xyz)``

    Use :func:`get_ch3cn_orca_vpt2_dipole_expansion` for the site-order
    tensors used by the TTNO dipole builder.
    """
    return {
        "linear_cartesian": _parse_orca_vpt2_tensor_block(
            path,
            "Dipole Derivatives[i][j]",
            rank=2,
        ),
        "second_mixed": _parse_orca_vpt2_tensor_block(
            path,
            "2nd Dipole Derivatives[NVib][threeN][xyz]",
            rank=3,
        ),
        "third_mixed": _parse_orca_vpt2_tensor_block(
            path,
            "3rd Dipole Derivatives[NVib][threeN][xyz]",
            rank=3,
        ),
    }


def _component_axis(component: str | None) -> int | None:
    if component is None:
        return None
    component_map = {"x": 0, "y": 1, "z": 2}
    try:
        return component_map[component]
    except KeyError as exc:
        raise ValueError('component must be None, "x", "y", or "z".') from exc


def _semidiagonal_cubic_to_canonical(semi_cubic: np.ndarray) -> np.ndarray:
    """
    Convert ORCA's projected ``d3mu/dq_i^2 dq_j`` data to canonical triples.

    The output stores values only at sorted index triples.  This matches the
    polynomial-name-list convention used by the TTNO dipole builder.
    """
    semi_cubic = np.asarray(semi_cubic, dtype=np.float64)
    if semi_cubic.ndim not in (2, 3) or semi_cubic.shape[0:2] != (12, 12):
        raise ValueError(
            "semi_cubic must have shape (12, 12) or (12, 12, 3), "
            f"got {semi_cubic.shape}."
        )

    if semi_cubic.ndim == 2:
        cubic = np.zeros((12, 12, 12), dtype=np.float64)
        for repeated_mode in range(12):
            for other_mode in range(12):
                key = tuple(sorted((repeated_mode, repeated_mode, other_mode)))
                cubic[key] = semi_cubic[repeated_mode, other_mode]
        return cubic

    cubic = np.zeros((12, 12, 12, semi_cubic.shape[2]), dtype=np.float64)
    for repeated_mode in range(12):
        for other_mode in range(12):
            key = tuple(sorted((repeated_mode, repeated_mode, other_mode)))
            cubic[key] = semi_cubic[repeated_mode, other_mode]
    return cubic


def get_ch3cn_orca_vpt2_dipole_expansion(
    component: str | None = None,
    path: pathlib.Path | str = CH3CN_ORCA_VPT2,
    normal_modes_path: pathlib.Path | str = CH3CN_ORCA_HESS,
    map_modes: Sequence[int] = CH3CN_SITE_TO_ORCA_MODES,
    linear_source: str = "ir_table",
    symmetrize_quadratic: bool = True,
    include_cubic: bool = True,
) -> dict[str, np.ndarray]:
    """
    Return site-order CH3CN dipole Taylor tensors from ORCA VPT2 data.

    The returned tensors follow the CH3CN site order and can be passed to
    ``Dipole_moment.get_mu_ttno_polynomial``:

    - ``linear``: shape ``(12,)`` or ``(12, 3)``
    - ``quadratic``: shape ``(12, 12)`` or ``(12, 12, 3)``
    - ``cubic``: shape ``(12, 12, 12)`` or ``(12, 12, 12, 3)``

    ``linear_source="ir_table"`` keeps the same first-order coefficients used
    by the current notebooks.  ``linear_source="projected_cartesian"`` instead
    projects ORCA's raw Cartesian dipole derivatives through the normal modes.

    ORCA's printed third-derivative block has shape ``NVib x 3N x xyz``.  After
    projection this gives semi-diagonal ``d3mu/dq_i^2 dq_j`` entries, not a
    complete arbitrary ``d3mu/dq_i dq_j dq_k`` tensor.  The returned ``cubic``
    tensor therefore contains only those canonical ``q_i^2 q_j`` terms.
    """
    component_index = _component_axis(component)
    mapping = [int(i) for i in map_modes]
    if len(mapping) != 12 or sorted(mapping) != list(range(12)):
        raise ValueError("map_modes must be a permutation of 0..11.")

    blocks = get_ch3cn_orca_vpt2_dipole_derivative_blocks(path)
    normal_modes = get_ch3cn_orca_normal_modes(
        path=normal_modes_path,
        map_modes=mapping,
    )

    linear_source = str(linear_source)
    if linear_source == "ir_table":
        linear = CH3CN_IR_TRANSITION_DIPOLE_12.copy()
    elif linear_source == "projected_cartesian":
        linear = normal_modes.T @ blocks["linear_cartesian"]
    elif linear_source == "zero":
        linear = np.zeros((12, 3), dtype=np.float64)
    else:
        raise ValueError(
            'linear_source must be "ir_table", "projected_cartesian", or "zero".'
        )

    second_site_first = blocks["second_mixed"][mapping, :, :]
    quadratic = np.einsum("icp,cj->ijp", second_site_first, normal_modes)
    if symmetrize_quadratic:
        quadratic = 0.5 * (quadratic + np.swapaxes(quadratic, 0, 1))

    if include_cubic:
        third_site_first = blocks["third_mixed"][mapping, :, :]
        semi_cubic = np.einsum("icp,cj->ijp", third_site_first, normal_modes)
        cubic = _semidiagonal_cubic_to_canonical(semi_cubic)
    else:
        cubic = np.zeros((12, 12, 12, 3), dtype=np.float64)

    if component_index is not None:
        linear = linear[:, component_index]
        quadratic = quadratic[:, :, component_index]
        cubic = cubic[:, :, :, component_index]

    return {
        "linear": linear,
        "quadratic": quadratic,
        "cubic": cubic,
    }


def get_ch3cn_orca_potential_energy(
    alpha: float = 1000,
    path: pathlib.Path | str = CH3CN_ORCA_VPT2,
    force_constant_cutoff_cm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (w, k3, k4) from ORCA in the same scaled units as Experiment potentials.
    """
    w = CH3CN_IR_FREQ_CM_12.astype(np.float64) / float(alpha)
    k3_cm, k4_cm = get_ch3cn_orca_vpt2_force_field(
        path=path,
        force_constant_cutoff_cm=force_constant_cutoff_cm,
        as_polynomial_coefficients=True,
    )
    return w, k3_cm / float(alpha), k4_cm / float(alpha)


def create_node_mapping(node_order: Sequence[int]) -> dict[int, int]:
    if len(node_order) != 12:
        raise ValueError("node_order must contain exactly 12 values.")

    order = [int(i) for i in node_order]
    if sorted(order) != list(range(12)):
        raise ValueError("node_order must be a permutation of 0..11.")

    sort_idx = np.argsort(order)
    inv = np.empty_like(sort_idx)
    inv[sort_idx] = np.arange(12)
    return {i: int(inv[i]) for i in range(12)}


def _prepare_inputs(
    physical_dim: Sequence[int],
    omega: Sequence[float],
    orb_state: np.ndarray,
    node_order: Sequence[int],
):
    physical_dim = [int(n) for n in physical_dim]
    omega = np.asarray(omega, dtype=np.float64)
    orb_state = np.asarray(orb_state, dtype=np.int64)

    if orb_state.ndim == 1:
        orb_state = orb_state.reshape(1, -1)

    if len(physical_dim) != 12:
        raise ValueError("CH3CN should have exactly 12 vibrational modes.")
    if omega.shape != (12,):
        raise ValueError(f"omega must have shape (12,), got {omega.shape}.")
    if orb_state.shape[1] != 12:
        raise ValueError(f"orb_state must have shape (m, 12), got {orb_state.shape}.")

    node_mapping = create_node_mapping(node_order)
    sort_idx = np.argsort([int(i) for i in node_order])
    return physical_dim, omega, orb_state, sort_idx, node_mapping


def _harmonic_tensors(
    physical_dim: Sequence[int],
    omega: Sequence[float],
    orb_state: np.ndarray,
    node_order: Sequence[int],
    nneighbour: Sequence[int],
    dtype: np.dtype,
):
    physical_dim, omega, orb_state, sort_idx, _ = _prepare_inputs(
        physical_dim,
        omega,
        orb_state,
        node_order,
    )

    ho_tensors = get_harmonic_oscillator_orbitals(
        physical_dim,
        omega,
        orb_state,
    )

    if orb_state.shape[0] == 1:
        for i in range(12):
            ho_tensors[i] = ho_tensors[i].reshape(-1, 1)

    nneighbour = [int(nneighbour[i]) for i in sort_idx]
    ho_tensors = [ho_tensors[i] for i in sort_idx]

    for i in range(12):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])
        ho_tensors[i] = ho_tensors[i].astype(dtype)

    return ho_tensors


def random_mps_ch3cn_harmonic_oscillator_0(
    physical_dim: Sequence[int],
    omega: Sequence[float],
    orb_state: np.ndarray,
    node_order: Sequence[int] = CH3CN_DEFAULT_NODE_ORDER,
    dtype: np.dtype = np.float64,
) -> TTNS:
    nneighbour = [1] + [2] * 10 + [1]
    ho_tensors = _harmonic_tensors(
        physical_dim,
        omega,
        orb_state,
        node_order,
        nneighbour,
        dtype,
    )
    node_mapping = create_node_mapping(node_order)

    nodes = [
        (Node(tensor=ho_tensors[i], identifier=f"site{i}"), ho_tensors[i])
        for i in range(12)
    ]

    state = TTNS()
    state.add_root(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1])
    state.add_child_to_parent(
        nodes[node_mapping[1]][0],
        nodes[node_mapping[1]][1],
        0,
        f"site{node_mapping[0]}",
        0,
    )

    for i in range(10):
        state.add_child_to_parent(
            nodes[node_mapping[i + 2]][0],
            nodes[node_mapping[i + 2]][1],
            0,
            f"site{node_mapping[i + 1]}",
            1,
        )

    state.canonical_form(f"site{node_mapping[11]}", mode=SplitMode.KEEP)
    state.normalize()
    return state


def random_leafonly_ch3cn_harmonic_oscillator_0(
    physical_dim: Sequence[int],
    omega: Sequence[float],
    orb_state: np.ndarray,
    node_order: Sequence[int] = CH3CN_DEFAULT_NODE_ORDER,
    dtype: np.dtype = np.float64,
) -> TTNS:
    orb_state = np.asarray(orb_state, dtype=np.int64)
    if orb_state.ndim == 1:
        orb_state = orb_state.reshape(1, -1)

    m = orb_state.shape[0]
    ho_tensors = _harmonic_tensors(
        physical_dim,
        omega,
        orb_state,
        node_order,
        [1] * 12,
        dtype,
    )
    node_mapping = create_node_mapping(node_order)

    nodes = [
        (Node(tensor=ho_tensors[i], identifier=f"site{i}"), ho_tensors[i])
        for i in range(12)
    ]

    shapes_ancillary = [[m, m, 1]] + [[m, m, m, 1]] * 10
    nodes.extend(
        random_tensor_node(shape, identifier=f"site{i + 12}", dtype=dtype)
        for i, shape in enumerate(shapes_ancillary)
    )

    state = TTNS()
    state.add_root(nodes[12][0], nodes[12][1])
    state.add_child_to_parent(nodes[13][0], nodes[13][1], 0, "site12", 0)
    state.add_child_to_parent(nodes[14][0], nodes[14][1], 0, "site13", 1)
    state.add_child_to_parent(nodes[15][0], nodes[15][1], 0, "site14", 1)
    state.add_child_to_parent(nodes[16][0], nodes[16][1], 0, "site13", 2)
    state.add_child_to_parent(nodes[17][0], nodes[17][1], 0, "site16", 1)
    state.add_child_to_parent(nodes[18][0], nodes[18][1], 0, "site16", 2)
    state.add_child_to_parent(nodes[19][0], nodes[19][1], 0, "site12", 1)
    state.add_child_to_parent(nodes[20][0], nodes[20][1], 0, "site19", 1)
    state.add_child_to_parent(nodes[21][0], nodes[21][1], 0, "site20", 1)
    state.add_child_to_parent(nodes[22][0], nodes[22][1], 0, "site19", 2)
    state.add_child_to_parent(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1], 0, "site14", 2)
    state.add_child_to_parent(nodes[node_mapping[4]][0], nodes[node_mapping[4]][1], 0, "site15", 1)
    state.add_child_to_parent(nodes[node_mapping[5]][0], nodes[node_mapping[5]][1], 0, "site15", 2)
    state.add_child_to_parent(nodes[node_mapping[6]][0], nodes[node_mapping[6]][1], 0, "site17", 1)
    state.add_child_to_parent(nodes[node_mapping[7]][0], nodes[node_mapping[7]][1], 0, "site17", 2)
    state.add_child_to_parent(nodes[node_mapping[8]][0], nodes[node_mapping[8]][1], 0, "site18", 1)
    state.add_child_to_parent(nodes[node_mapping[9]][0], nodes[node_mapping[9]][1], 0, "site18", 2)
    state.add_child_to_parent(nodes[node_mapping[2]][0], nodes[node_mapping[2]][1], 0, "site20", 2)
    state.add_child_to_parent(nodes[node_mapping[1]][0], nodes[node_mapping[1]][1], 0, "site21", 1)
    state.add_child_to_parent(nodes[node_mapping[3]][0], nodes[node_mapping[3]][1], 0, "site21", 2)
    state.add_child_to_parent(nodes[node_mapping[10]][0], nodes[node_mapping[10]][1], 0, "site22", 1)
    state.add_child_to_parent(nodes[node_mapping[11]][0], nodes[node_mapping[11]][1], 0, "site22", 2)
    state.canonical_form(state.root_id, mode=SplitMode.KEEP)
    state.normalize()
    return state


def random_threetree_ch3cn_harmonic_oscillator_0(
    physical_dim: Sequence[int],
    omega: Sequence[float],
    orb_state: np.ndarray,
    node_order: Sequence[int] = CH3CN_DEFAULT_NODE_ORDER,
    dtype: np.dtype = np.float64,
) -> TTNS:
    orb_state = np.asarray(orb_state, dtype=np.int64)
    if orb_state.ndim == 1:
        orb_state = orb_state.reshape(1, -1)

    m = orb_state.shape[0]
    nneighbour = [1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1]
    ho_tensors = _harmonic_tensors(
        physical_dim,
        omega,
        orb_state,
        node_order,
        nneighbour,
        dtype,
    )
    node_mapping = create_node_mapping(node_order)

    nodes = [
        (Node(tensor=ho_tensors[i], identifier=f"site{i}"), ho_tensors[i])
        for i in range(12)
    ]
    nodes.append(
        random_tensor_node([m, m, m, 1], identifier="site12", dtype=dtype)
    )

    state = TTNS()
    state.add_root(nodes[12][0], nodes[12][1])
    state.add_child_to_parent(nodes[node_mapping[3]][0], nodes[node_mapping[3]][1], 0, "site12", 0)
    state.add_child_to_parent(nodes[node_mapping[2]][0], nodes[node_mapping[2]][1], 0, f"site{node_mapping[3]}", 1)
    state.add_child_to_parent(nodes[node_mapping[1]][0], nodes[node_mapping[1]][1], 0, f"site{node_mapping[2]}", 1)
    state.add_child_to_parent(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1], 0, f"site{node_mapping[1]}", 1)
    state.add_child_to_parent(nodes[node_mapping[4]][0], nodes[node_mapping[4]][1], 0, "site12", 1)
    state.add_child_to_parent(nodes[node_mapping[5]][0], nodes[node_mapping[5]][1], 0, f"site{node_mapping[4]}", 1)
    state.add_child_to_parent(nodes[node_mapping[6]][0], nodes[node_mapping[6]][1], 0, f"site{node_mapping[5]}", 1)
    state.add_child_to_parent(nodes[node_mapping[7]][0], nodes[node_mapping[7]][1], 0, f"site{node_mapping[6]}", 1)
    state.add_child_to_parent(nodes[node_mapping[8]][0], nodes[node_mapping[8]][1], 0, "site12", 2)
    state.add_child_to_parent(nodes[node_mapping[9]][0], nodes[node_mapping[9]][1], 0, f"site{node_mapping[8]}", 1)
    state.add_child_to_parent(nodes[node_mapping[10]][0], nodes[node_mapping[10]][1], 0, f"site{node_mapping[9]}", 1)
    state.add_child_to_parent(nodes[node_mapping[11]][0], nodes[node_mapping[11]][1], 0, f"site{node_mapping[10]}", 1)
    state.canonical_form(state.root_id, mode=SplitMode.KEEP)
    state.normalize()
    return state


def get_ch3cn_ir_dipole_derivative(
    normalize: bool = True,
    component: str | None = None,
    path: pathlib.Path | str = CH3CN_ORCA_VIB_OUT,
) -> np.ndarray:
    """
    Return a one-coefficient-per-mode dipole vector from ORCA data.

    component:
        None or "intensity": sqrt(ORCA IR intensity in km/mol).
        "t2": sqrt(ORCA T**2).
        "x", "y", "z": ORCA transition dipole component Tx/Ty/Tz.
    """
    table = get_ch3cn_orca_ir_table(path)
    if component is None or component == "intensity":
        derivative = np.sqrt(np.maximum(table["intensity_km_mol"], 0.0))
    elif component == "t2":
        derivative = np.sqrt(np.maximum(table["t2"], 0.0))
    elif component in ("x", "y", "z"):
        derivative = table[f"t{component}"].copy()
    else:
        raise ValueError('component must be None, "intensity", "t2", "x", "y", or "z".')

    if normalize:
        norm = np.linalg.norm(derivative)
        if norm > 0:
            derivative = derivative / norm
    return derivative


# Backward-compatible names matching Experiment/utils_ch3cn.py.
random_mps_harmonic_oscillator_0 = random_mps_ch3cn_harmonic_oscillator_0
random_leafonly_harmonic_oscillator_0 = random_leafonly_ch3cn_harmonic_oscillator_0
random_threetree_harmonic_oscillator_0 = random_threetree_ch3cn_harmonic_oscillator_0
