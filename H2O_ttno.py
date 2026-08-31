from typing import List, Dict, Union, Tuple
from collections import Counter
from fractions import Fraction
import numpy as np
from scipy.special import roots_hermite
import math
from pytreenet.ttno import TreeTensorNetworkOperator, TTNOFinder
from pytreenet.operators import Hamiltonian, TensorProduct
from pytreenet.ttns import TreeTensorNetworkState
import pathlib
from ttn_project.calc.orca_parser import get_anharmonic_constants

f = pathlib.Path(__file__).parent / "ttn_project" / "orca" / "h2o" / "vib"

def get_laplacian(xs: np.ndarray) -> np.ndarray:
    N = len(xs)
    lp = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                lp[i, j] = (-1)**(i - j) * (2*(xs[i] - xs[j])**(-2) - 0.5)
            else:
                lp[i, j] = 1.0/6 * (4*N - 1 - 2*xs[i]**2)

    return 0.5 * lp


def parse_force_terms(text: str, order: int):
    """
    把 orca_parser 返回的 cubic/quartic 字符串解析成：
        indices, coeff
    例如：
        "1 1 2  123.0"
    变成：
        [0, 0, 1], 123.0 / factorial(order)
    """
    terms = []

    for raw in text.strip().splitlines():
        parts = raw.split()
        if not parts:
            continue

        # ORCA parser 输出是 1-based，这里转成 0-based site index
        idxs = [int(x) - 1 for x in parts[:order]]
        coeff = float(parts[order]) / math.factorial(order) / 1000.0

        terms.append((idxs, coeff))

    return terms


def get_name_list_h2o_harmonic(
    N: List[int],
    filename,
    map_modes=[0, 1, 2],
    signs=[1, 1, 1],
) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:

    w, cubic, quartic = get_anharmonic_constants(
        filename,
        map_modes=map_modes,
        signs=signs,
    )

    w = np.array(w, dtype=np.float64) / 1000.0

    n_modes = len(w)
    assert len(N) == n_modes

    name_list = []
    conversion_dict = {}

    for n in np.unique(N):
        x, _ = roots_hermite(n)

        conversion_dict[f"I{n}"] = np.eye(n)
        conversion_dict[f"t{n}"] = get_laplacian(x)

        conversion_dict[f"q{n}^2"] = np.diag(x**2)

    for i in range(n_modes):
        name_list.append({
            f"site{i}": f"t{N[i]}",
            "coeff": w[i],
        })

        name_list.append({
            f"site{i}": f"q{N[i]}^2",
            "coeff": 0.5 * w[i],
        })

    return name_list, conversion_dict


def get_name_list_h2o(
    N: List[int],
    filename,
    map_modes=[0, 1, 2],
    signs=[1, 1, 1],
) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:

    w, cubic, quartic = get_anharmonic_constants(
        filename,
        map_modes=map_modes,
        signs=signs,
    )

    w = np.array(w, dtype=np.float64) / 1000.0

    n_modes = len(w)
    assert len(N) == n_modes

    name_list = []
    conversion_dict = {}

    # local operators
    for n in np.unique(N):
        x, _ = roots_hermite(n)

        conversion_dict[f"I{n}"] = np.eye(n)
        conversion_dict[f"t{n}"] = get_laplacian(x)

        q = np.diag(x)
        conversion_dict[f"q{n}"] = q
        conversion_dict[f"q{n}^2"] = np.diag(x**2)
        conversion_dict[f"q{n}^3"] = np.diag(x**3)
        conversion_dict[f"q{n}^4"] = np.diag(x**4)

    # harmonic kinetic + harmonic potential
    for i in range(n_modes):
        name_list.append({
            f"site{i}": f"t{N[i]}",
            "coeff": w[i],
        })

        name_list.append({
            f"site{i}": f"q{N[i]}^2",
            "coeff": 0.5 * w[i],
        })

    # cubic terms
    for idxs, coeff in parse_force_terms(cubic, order=3):
        count_dict = Counter(idxs)
        name_dict = {}

        for site, power in count_dict.items():
            if power == 1:
                name_dict[f"site{site}"] = f"q{N[site]}"
            else:
                name_dict[f"site{site}"] = f"q{N[site]}^{power}"

        name_dict["coeff"] = coeff
        name_list.append(name_dict)

    # quartic terms
    for idxs, coeff in parse_force_terms(quartic, order=4):
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

def get_ttno_h2o_harmonic(
    N: List[int],
    state: TreeTensorNetworkState,
    filename= f,
    map_modes=[0, 1, 2],
    signs=[1, 1, 1],
    hamiltonian: bool = False,
) -> TreeTensorNetworkOperator:

    name_list, conversion_dict = get_name_list_h2o_harmonic(
        N=N,
        filename=filename,
        map_modes=map_modes,
        signs=signs,
    )

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
        dtype=np.float64,
        method=TTNOFinder.SGE,
    )

    if hamiltonian:
        return ttno, ham_pad
    return ttno


def get_ttno_h2o(
    N: List[int],
    state: TreeTensorNetworkState,
    filename= f,
    map_modes=[0, 1, 2],
    signs=[1, 1, 1],
    hamiltonian: bool = False,
) -> TreeTensorNetworkOperator:

    name_list, conversion_dict = get_name_list_h2o(
        N=N,
        filename=filename,
        map_modes=map_modes,
        signs=signs,
    )

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
        dtype=np.float64,
        method=TTNOFinder.SGE,
    )

    if hamiltonian:
        return ttno, ham_pad
    return ttno