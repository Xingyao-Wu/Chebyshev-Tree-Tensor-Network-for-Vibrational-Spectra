from collections import Counter
from math import factorial
from typing import Dict, List, Sequence, Tuple, Union
from fractions import Fraction
import numpy as np
from scipy.special import roots_hermite

from pytreenet.ttno import TreeTensorNetworkOperator, TTNOFinder
from pytreenet.operators import Hamiltonian, TensorProduct
from pytreenet.ttns import TreeTensorNetworkState


def _polynomial_denominator(indices: Sequence[int]) -> int:
    counts = Counter(int(index) for index in indices)
    denom = 1
    for count in counts.values():
        denom *= factorial(count)
    return denom


def _q_label(local_dim: int, power: int) -> str:
    if power == 1:
        return f"q{local_dim}"
    return f"q{local_dim}^{power}"


def _monomial_name_dict(N: Sequence[int], indices: Sequence[int]) -> Dict[str, str]:
    counts = Counter(int(index) for index in indices)
    name_dict = {}
    for site, power in sorted(counts.items()):
        name_dict[f"site{site}"] = _q_label(int(N[site]), power)
    return name_dict


def _build_mu_ttno_from_name_list(
    name_list: List[Dict[str, Union[str, float]]],
    conversion_dict: Dict[str, np.ndarray],
    state: TreeTensorNetworkState,
    dtype: np.dtype = np.float64,
    return_hamiltonian: bool = False,
) -> Union[TreeTensorNetworkOperator, Tuple[TreeTensorNetworkOperator, Hamiltonian]]:
    # Needed because the TTNS tree may contain ancillary physical dimension 1 nodes.
    conversion_dict["I1"] = np.eye(1)

    new_name_list = [{k: v for k, v in d.items() if k != "coeff"} for d in name_list]

    coeffs = [
        (Fraction(str(np.around(d["coeff"], 12))), "1")
        for d in name_list
    ]

    terms = [TensorProduct(term) for term in new_name_list]
    mu_terms = [(coeff, label, term) for (coeff, label), term in zip(coeffs, terms)]

    mu_ham = Hamiltonian(mu_terms, conversion_dictionary=conversion_dict)
    mu_pad = mu_ham.pad_with_identities(state, symbolic=True)

    mu_ttno = TreeTensorNetworkOperator.from_hamiltonian(
        mu_pad,
        state,
        dtype=dtype,
        method=TTNOFinder.SGE,
    )

    if return_hamiltonian:
        return mu_ttno, mu_pad
    return mu_ttno


def get_mu_name_list_linear(
    N: List[int],
    dipole_derivative: np.ndarray,
    *,
    mu0: float = 0.0,
):
    """
    Construct name_list and conversion_dict for linear dipole operator

        mu = mu0 + sum_i d_i q_i

    N:
        local basis sizes, e.g. [9,7,9,...]
    dipole_derivative:
        shape (nmode,), coefficients d_i for one dipole component.
        For example d_mu_dq_x, d_mu_dq_y, or d_mu_dq_z.
    mu0:
        constant dipole offset for this component. Set to 0.0 to omit it.
    """
    nmode = len(N)
    dipole_derivative = np.asarray(dipole_derivative, dtype=np.float64)

    if dipole_derivative.shape != (nmode,):
        raise ValueError(f"dipole_derivative must have shape {(nmode,)}, got {dipole_derivative.shape}")

    unique_N = np.unique(N)
    conversion_dict = {}
    name_list = []

    for n in unique_N:
        x, _ = roots_hermite(n)
        conversion_dict[f"I{n}"] = np.eye(n)
        conversion_dict[f"q{n}"] = np.diag(x)

    if abs(mu0) > 1e-14:
        name_list.append({"coeff": float(mu0)})

    for i, coeff in enumerate(dipole_derivative):
        if abs(coeff) > 1e-14:
            name_list.append({
                f"site{i}": f"q{N[i]}",
                "coeff": float(coeff),
            })

    return name_list, conversion_dict


def get_mu_name_list_polynomial(
    N: List[int],
    linear: np.ndarray | None = None,
    quadratic: np.ndarray | None = None,
    cubic: np.ndarray | None = None,
    derivative_coefficients: bool = True,
    coefficient_cutoff: float = 1e-14,
) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
    """
    Construct name_list and conversion_dict for a polynomial dipole operator.

    With ``derivative_coefficients=True`` the inputs are Taylor derivatives:

        mu = sum_i d_i q_i
             + 1/2 sum_ij d_ij q_i q_j
             + 1/6 sum_ijk d_ijk q_i q_j q_k

    The returned name list contains one canonical monomial per sorted index
    tuple.  Repeated-index factors are handled automatically.  If the input
    arrays already contain direct monomial coefficients, set
    ``derivative_coefficients=False``.
    """
    nmode = len(N)
    N = [int(n) for n in N]
    cutoff = float(coefficient_cutoff)

    if linear is not None:
        linear = np.asarray(linear, dtype=np.float64)
        if linear.shape != (nmode,):
            raise ValueError(f"linear must have shape {(nmode,)}, got {linear.shape}.")

    if quadratic is not None:
        quadratic = np.asarray(quadratic, dtype=np.float64)
        if quadratic.shape != (nmode, nmode):
            raise ValueError(
                f"quadratic must have shape {(nmode, nmode)}, got {quadratic.shape}."
            )
        quadratic = 0.5 * (quadratic + quadratic.T)

    if cubic is not None:
        cubic = np.asarray(cubic, dtype=np.float64)
        if cubic.shape != (nmode, nmode, nmode):
            raise ValueError(
                f"cubic must have shape {(nmode, nmode, nmode)}, got {cubic.shape}."
            )

    conversion_dict = {}
    for n in np.unique(N):
        x, _ = roots_hermite(n)
        conversion_dict[f"I{n}"] = np.eye(n)
        conversion_dict[f"q{n}"] = np.diag(x)
        conversion_dict[f"q{n}^2"] = np.diag(x**2)
        conversion_dict[f"q{n}^3"] = np.diag(x**3)

    name_list: List[Dict[str, Union[str, float]]] = []

    if linear is not None:
        for i, value in enumerate(linear):
            coeff = float(value)
            if abs(coeff) > cutoff:
                term = _monomial_name_dict(N, (i,))
                term["coeff"] = coeff
                name_list.append(term)

    if quadratic is not None:
        for i in range(nmode):
            for j in range(i, nmode):
                value = float(quadratic[i, j])
                coeff = value
                if derivative_coefficients:
                    coeff /= _polynomial_denominator((i, j))
                if abs(coeff) > cutoff:
                    term = _monomial_name_dict(N, (i, j))
                    term["coeff"] = coeff
                    name_list.append(term)

    if cubic is not None:
        for i in range(nmode):
            for j in range(i, nmode):
                for k in range(j, nmode):
                    value = float(cubic[i, j, k])
                    coeff = value
                    if derivative_coefficients:
                        coeff /= _polynomial_denominator((i, j, k))
                    if abs(coeff) > cutoff:
                        term = _monomial_name_dict(N, (i, j, k))
                        term["coeff"] = coeff
                        name_list.append(term)

    return name_list, conversion_dict


def get_mu_ttno_linear(
    N: List[int],
    state: TreeTensorNetworkState,
    dipole_derivative: np.ndarray,
    dtype: np.dtype = np.float64,
    return_hamiltonian: bool = False,
    *,
    mu0: float = 0.0,
) -> Union[TreeTensorNetworkOperator, Tuple[TreeTensorNetworkOperator, Hamiltonian]]:
    """
    Build TTNO for

        mu = mu0 + sum_i d_i q_i

    using the same pytreenet Hamiltonian wrapper as your get_ttno().
    """
    name_list, conversion_dict = get_mu_name_list_linear(
        N,
        dipole_derivative,
        mu0=mu0,
    )
    return _build_mu_ttno_from_name_list(
        name_list,
        conversion_dict,
        state,
        dtype=dtype,
        return_hamiltonian=return_hamiltonian,
    )


def get_mu_ttno_polynomial(
    N: List[int],
    state: TreeTensorNetworkState,
    linear: np.ndarray | None = None,
    quadratic: np.ndarray | None = None,
    cubic: np.ndarray | None = None,
    derivative_coefficients: bool = True,
    coefficient_cutoff: float = 1e-14,
    dtype: np.dtype = np.float64,
    return_hamiltonian: bool = False,
) -> Union[TreeTensorNetworkOperator, Tuple[TreeTensorNetworkOperator, Hamiltonian]]:
    """
    Build a TTNO for a linear/quadratic/cubic dipole expansion.

    See :func:`get_mu_name_list_polynomial` for the coefficient convention.
    """
    name_list, conversion_dict = get_mu_name_list_polynomial(
        N,
        linear=linear,
        quadratic=quadratic,
        cubic=cubic,
        derivative_coefficients=derivative_coefficients,
        coefficient_cutoff=coefficient_cutoff,
    )
    return _build_mu_ttno_from_name_list(
        name_list,
        conversion_dict,
        state,
        dtype=dtype,
        return_hamiltonian=return_hamiltonian,
    )
