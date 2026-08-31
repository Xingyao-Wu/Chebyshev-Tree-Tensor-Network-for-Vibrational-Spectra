from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import pathlib

import numpy as np
from scipy.special import roots_hermite

from pytreenet.operators import Hamiltonian, TensorProduct
from pytreenet.ttno import TTNOFinder, TreeTensorNetworkOperator
from pytreenet.ttns import TreeTensorNetworkState
from ttn_project.calc.orca_parser import get_anharmonic_constants


CH3CN_VIB_PATH = pathlib.Path(__file__).parent / "ch3cn" / "vib"

# Site order used in the CH3CN notebooks:
# [nu1, nu2, nu3, nu4, nu5a, nu5b, nu6a, nu6b, nu7a, nu7b, nu8a, nu8b]
#
# ORCA's vib.out/vib.vpt2 order is ascending frequency. This maps the site
# order above to ORCA's zero-based mode indices.
CH3CN_SITE_TO_ORCA_MODES = [9, 8, 5, 2, 10, 11, 6, 7, 3, 4, 0, 1]
CH3CN_DEFAULT_SIGNS = [1] * 12


def get_laplacian(xs: np.ndarray) -> np.ndarray:
    N = len(xs)
    lp = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                lp[i, j] = (-1) ** (i - j) * (2 * (xs[i] - xs[j]) ** (-2) - 0.5)
            else:
                lp[i, j] = 1.0 / 6 * (4 * N - 1 - 2 * xs[i] ** 2)

    return 0.5 * lp


def _mode_mapping(map_modes: Optional[Sequence[int]]) -> List[int]:
    if map_modes is None:
        return list(CH3CN_SITE_TO_ORCA_MODES)

    mapping = [int(i) for i in map_modes]
    if len(mapping) != 12:
        raise ValueError(f"CH3CN should have 12 vibrational modes, got {len(mapping)}.")
    if sorted(mapping) != list(range(12)):
        raise ValueError("map_modes must be a permutation of 0..11.")
    return mapping


def _mode_signs(signs: Optional[Sequence[int]]) -> List[int]:
    if signs is None:
        return list(CH3CN_DEFAULT_SIGNS)

    mode_signs = [int(s) for s in signs]
    if len(mode_signs) != 12:
        raise ValueError(f"CH3CN should have 12 mode signs, got {len(mode_signs)}.")
    if any(s not in (-1, 1) for s in mode_signs):
        raise ValueError("signs must contain only -1 or 1.")
    return mode_signs


def _get_mapped_anharmonic_constants(
    filename: pathlib.Path,
    map_modes: Optional[Sequence[int]],
    signs: Optional[Sequence[int]],
):
    mapping = _mode_mapping(map_modes)
    mode_signs = _mode_signs(signs)

    w_orca, cubic, quartic = get_anharmonic_constants(
        filename,
        map_modes=mapping,
        signs=mode_signs,
    )

    # ttn_project.calc.orca_parser maps cubic/quartic indices, but returns
    # frequencies in ORCA order. Reorder frequencies to match the site order.
    w = np.asarray(w_orca, dtype=np.float64)[mapping]
    return w, cubic, quartic


def parse_force_terms(text: str, order: int, alpha: float = 1000.0):
    """
    Parse ORCA force-constant lines into zero-based indices and scaled coeffs.

    ORCA/parser output is one-based. The TTNO Hamiltonian coefficient includes
    the Taylor factorial and the same cm^-1 -> 10^3 cm^-1 scaling as H2O_ttno.
    """
    terms = []

    for raw in text.strip().splitlines():
        parts = raw.split()
        if not parts:
            continue

        idxs = [int(x) - 1 for x in parts[:order]]
        coeff = float(parts[order]) / math.factorial(order) / alpha
        terms.append((idxs, coeff))

    return terms


def get_ch3cn_frequencies(
    filename: pathlib.Path = CH3CN_VIB_PATH,
    map_modes: Optional[Sequence[int]] = None,
    signs: Optional[Sequence[int]] = None,
    alpha: float = 1000.0,
) -> np.ndarray:
    w, _, _ = _get_mapped_anharmonic_constants(
        filename=filename,
        map_modes=map_modes,
        signs=signs,
    )
    return np.asarray(w, dtype=np.float64) / alpha


def _validate_basis(N: Sequence[int], n_modes: int) -> List[int]:
    basis = [int(n) for n in N]
    if len(basis) != n_modes:
        raise ValueError(f"N must have length {n_modes}, got {len(basis)}.")
    return basis


def get_name_list_ch3cn_harmonic(
    N: Sequence[int],
    filename: pathlib.Path = CH3CN_VIB_PATH,
    map_modes: Optional[Sequence[int]] = None,
    signs: Optional[Sequence[int]] = None,
    alpha: float = 1000.0,
) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
    w, _, _ = _get_mapped_anharmonic_constants(
        filename=filename,
        map_modes=map_modes,
        signs=signs,
    )

    w = np.asarray(w, dtype=np.float64) / alpha
    N = _validate_basis(N, len(w))

    name_list = []
    conversion_dict = {}

    for n in np.unique(N):
        x, _ = roots_hermite(n)
        conversion_dict[f"I{n}"] = np.eye(n)
        conversion_dict[f"t{n}"] = get_laplacian(x)
        conversion_dict[f"q{n}^2"] = np.diag(x**2)

    for i in range(len(w)):
        name_list.append({
            f"site{i}": f"t{N[i]}",
            "coeff": w[i],
        })

        name_list.append({
            f"site{i}": f"q{N[i]}^2",
            "coeff": 0.5 * w[i],
        })

    return name_list, conversion_dict


def get_name_list_ch3cn(
    N: Sequence[int],
    filename: pathlib.Path = CH3CN_VIB_PATH,
    map_modes: Optional[Sequence[int]] = None,
    signs: Optional[Sequence[int]] = None,
    alpha: float = 1000.0,
) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
    w, cubic, quartic = _get_mapped_anharmonic_constants(
        filename=filename,
        map_modes=map_modes,
        signs=signs,
    )

    w = np.asarray(w, dtype=np.float64) / alpha
    N = _validate_basis(N, len(w))

    name_list = []
    conversion_dict = {}

    for n in np.unique(N):
        x, _ = roots_hermite(n)

        conversion_dict[f"I{n}"] = np.eye(n)
        conversion_dict[f"t{n}"] = get_laplacian(x)

        conversion_dict[f"q{n}"] = np.diag(x)
        conversion_dict[f"q{n}^2"] = np.diag(x**2)
        conversion_dict[f"q{n}^3"] = np.diag(x**3)
        conversion_dict[f"q{n}^4"] = np.diag(x**4)

    for i in range(len(w)):
        name_list.append({
            f"site{i}": f"t{N[i]}",
            "coeff": w[i],
        })

        name_list.append({
            f"site{i}": f"q{N[i]}^2",
            "coeff": 0.5 * w[i],
        })

    for idxs, coeff in parse_force_terms(cubic, order=3, alpha=alpha):
        count_dict = Counter(idxs)
        name_dict = {}

        for site, power in count_dict.items():
            if power == 1:
                name_dict[f"site{site}"] = f"q{N[site]}"
            else:
                name_dict[f"site{site}"] = f"q{N[site]}^{power}"

        name_dict["coeff"] = coeff
        name_list.append(name_dict)

    for idxs, coeff in parse_force_terms(quartic, order=4, alpha=alpha):
        count_dict = Counter(idxs)
        name_dict = {}

        for site, power in count_dict.items():
            if power == 1:
                name_dict[f"site{site}"] = f"q{N[site]}"
            else:
                name_dict[f"site{site}"] = f"q{N[site]}^{power}"

        name_dict["coeff"] = coeff
        name_list.append(name_dict)

    return name_list, conversion_dict


def _build_ttno(
    name_list: List[Dict[str, Union[str, float]]],
    conversion_dict: Dict[str, np.ndarray],
    state: TreeTensorNetworkState,
    dtype: np.dtype = np.float64,
    hamiltonian: bool = False,
):
    conversion_dict["I1"] = np.eye(1)

    new_name_list = [
        {k: v for k, v in d.items() if k != "coeff"}
        for d in name_list
    ]

    coeffs = [
        (Fraction(str(np.around(d["coeff"], 8))), "1")
        for d in name_list
    ]

    terms = [TensorProduct(d) for d in new_name_list]
    ham_terms = [(x, y, z) for (x, y), z in zip(coeffs, terms)]

    ham = Hamiltonian(
        ham_terms,
        conversion_dictionary=conversion_dict,
    )

    ham_pad = ham.pad_with_identities(state, symbolic=True)

    ttno = TreeTensorNetworkOperator.from_hamiltonian(
        ham_pad,
        state,
        dtype=dtype,
        method=TTNOFinder.SGE,
    )

    if hamiltonian:
        return ttno, ham_pad
    return ttno


def get_ttno_ch3cn_harmonic(
    N: Sequence[int],
    state: TreeTensorNetworkState,
    filename: pathlib.Path = CH3CN_VIB_PATH,
    map_modes: Optional[Sequence[int]] = None,
    signs: Optional[Sequence[int]] = None,
    alpha: float = 1000.0,
    dtype: np.dtype = np.float64,
    hamiltonian: bool = False,
) -> TreeTensorNetworkOperator:
    name_list, conversion_dict = get_name_list_ch3cn_harmonic(
        N=N,
        filename=filename,
        map_modes=map_modes,
        signs=signs,
        alpha=alpha,
    )

    return _build_ttno(
        name_list=name_list,
        conversion_dict=conversion_dict,
        state=state,
        dtype=dtype,
        hamiltonian=hamiltonian,
    )


def get_ttno_ch3cn(
    N: Sequence[int],
    state: TreeTensorNetworkState,
    filename: pathlib.Path = CH3CN_VIB_PATH,
    map_modes: Optional[Sequence[int]] = None,
    signs: Optional[Sequence[int]] = None,
    alpha: float = 1000.0,
    dtype: np.dtype = np.float64,
    hamiltonian: bool = False,
) -> TreeTensorNetworkOperator:
    name_list, conversion_dict = get_name_list_ch3cn(
        N=N,
        filename=filename,
        map_modes=map_modes,
        signs=signs,
        alpha=alpha,
    )

    return _build_ttno(
        name_list=name_list,
        conversion_dict=conversion_dict,
        state=state,
        dtype=dtype,
        hamiltonian=hamiltonian,
    )
