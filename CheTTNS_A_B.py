"""Compatibility helpers for older CheTTNS_A_B notebooks.

The main Chebyshev utilities now live in ``CheTTNS.py``.  Some older
notebooks still import ``CheTTNS_A_B`` and also expect the small coCheTTNS
orthogonalization helpers below, so keep this shim local to those notebooks.
"""

import CheTTNS as _CheTTNS
from CheTTNS import *  # noqa: F401,F403

import copy
import numpy as np

from pytreenet.contractions.state_operator_contraction import get_matrix_element

# ``CheTTNS.py`` has an ``__all__`` list for the newer notebooks.  The old
# ``CheTTNS_A_B`` notebooks relied on names such as ``SVDParameters`` and
# ``ApplicationMethod`` being imported by ``*``, so expose all public names from
# the current module, not only its curated ``__all__``.
for _name in dir(_CheTTNS):
    if not _name.startswith("_"):
        globals().setdefault(_name, getattr(_CheTTNS, _name))


def _overlap_matrix_ttns(ttns_list):
    n_states = len(ttns_list)
    ovp = np.zeros((n_states, n_states), dtype=np.complex128)
    for i, ttns_i in enumerate(ttns_list):
        ovp[i, i] = ttns_i.scalar_product()
        for j in range(i + 1, n_states):
            ovp[i, j] = ttns_list[j].scalar_product(ttns_i)
            ovp[j, i] = ovp[i, j].conjugate()
    return 0.5 * (ovp + ovp.conj().T)


def cholesky_ttns(ttns_list, eig_cut=1e-12):
    """Return an overlap-orthogonalizing coefficient matrix and overlap."""
    ovp = _overlap_matrix_ttns(ttns_list)
    eigvals, eigvecs = np.linalg.eigh(ovp)
    keep = eigvals > eig_cut
    if not np.any(keep):
        raise RuntimeError(f"No TTNS overlap eigenvalues survived eig_cut={eig_cut:g}.")
    C = eigvecs[:, keep] @ np.diag(1.0 / np.sqrt(eigvals[keep]))
    return C, ovp


def canonical_ttns(ttns_list, eig_cut=1e-12):
    """Alias kept for notebooks that compare canonical/cholesky labels."""
    return cholesky_ttns(ttns_list, eig_cut=eig_cut)


def heff_ttns(ttns_list, C, H_scaled, shift=0.0):
    """Project ``H_scaled - shift * I`` into the orthogonalized TTNS basis."""
    n_states = len(ttns_list)
    C = np.asarray(C)
    if C.shape[0] != n_states:
        raise AssertionError(f"C must have first dimension {n_states}, got {C.shape}")

    H = np.zeros((n_states, n_states), dtype=np.complex128)
    for i, ttns_i in enumerate(ttns_list):
        H[i, i] = ttns_i.operator_expectation_value(H_scaled)
        for j in range(i + 1, n_states):
            hij = get_matrix_element(
                ttns_i.conjugate(),
                H_scaled,
                ttns_list[j],
            )
            H[i, j] = hij
            H[j, i] = hij.conjugate()

    if shift:
        H = H - shift * _overlap_matrix_ttns(ttns_list)
    return C.conj().T @ H @ C


def psi_eff_ttns(ttns_list, C, ttns0):
    """Project a TTNS into the orthogonalized effective basis."""
    n_states = len(ttns_list)
    C = np.asarray(C)
    if C.shape[0] != n_states:
        raise AssertionError(f"C must have first dimension {n_states}, got {C.shape}")

    ttns_left = copy.deepcopy(ttns0)
    b = np.zeros(n_states, dtype=np.complex128)
    for i, ttns_i in enumerate(ttns_list):
        b[i] = ttns_left.scalar_product(ttns_i)
    return C.conj().T @ b
