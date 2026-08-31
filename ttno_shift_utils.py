from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Sequence

import numpy as np

from pytreenet.core.node import relative_leg_permutation
from pytreenet.ttno.ttno_class import TTNO


def _as_coeff_list(coeffs, n_ttnos: int) -> list[Number]:
    if np.isscalar(coeffs):
        return [coeffs] * n_ttnos

    coeff_list = list(coeffs)
    if len(coeff_list) != n_ttnos:
        raise ValueError(
            f"Expected {n_ttnos} coefficients, got {len(coeff_list)}."
        )
    return coeff_list


def _check_compatible_ttnos(ttnos: Sequence[TTNO]) -> None:
    if len(ttnos) == 0:
        raise ValueError("At least one TTNO is required.")

    ref_ttno = ttnos[0]
    if ref_ttno.root_id is None:
        raise ValueError("Cannot combine an empty TTNO.")

    for ttno in ttnos[1:]:
        if ttno.root_id != ref_ttno.root_id:
            raise ValueError(
                f"All TTNOs must have the same root_id. "
                f"Expected {ref_ttno.root_id}, got {ttno.root_id}."
            )
        if not ref_ttno.same_hierarchy_as(ttno):
            raise ValueError("All TTNOs must have the same tree hierarchy.")

    for node_id, ref_node in ref_ttno.nodes.items():
        ref_open_dims = ref_node.open_dimensions()
        if len(ref_open_dims) != 2:
            raise ValueError(
                f"Node {node_id!r} has {len(ref_open_dims)} open legs; "
                "a TTNO node should have exactly two open legs."
            )

        for ttno in ttnos[1:]:
            open_dims = ttno.nodes[node_id].open_dimensions()
            if open_dims != ref_open_dims:
                raise ValueError(
                    f"Open dimensions differ at node {node_id!r}: "
                    f"{ref_open_dims} vs {open_dims}."
                )


def _block_diag_keep_open(
    tensors: Sequence[np.ndarray],
    num_open_legs: int = 2,
) -> np.ndarray:
    """Block-diagonal direct sum over virtual legs, preserving open legs."""
    if len(tensors) == 0:
        raise ValueError("The tensor list must not be empty.")

    ref_ndim = tensors[0].ndim
    if ref_ndim < num_open_legs:
        raise ValueError(
            f"Tensor rank {ref_ndim} is smaller than num_open_legs={num_open_legs}."
        )

    open_shape = tensors[0].shape[-num_open_legs:]
    for i, tensor in enumerate(tensors):
        if tensor.ndim != ref_ndim:
            raise ValueError(
                f"Tensor {i} has ndim {tensor.ndim}; expected {ref_ndim}."
            )
        if tensor.shape[-num_open_legs:] != open_shape:
            raise ValueError(
                f"Tensor {i} has open shape {tensor.shape[-num_open_legs:]}; "
                f"expected {open_shape}."
            )

    num_virtual_legs = ref_ndim - num_open_legs
    if num_virtual_legs == 0:
        return sum(tensors)

    virtual_shape = tuple(
        sum(tensor.shape[axis] for tensor in tensors)
        for axis in range(num_virtual_legs)
    )
    out = np.zeros(virtual_shape + open_shape, dtype=np.result_type(*tensors))

    offsets = [0] * num_virtual_legs
    for tensor in tensors:
        virtual_slices = []
        for axis in range(num_virtual_legs):
            lower = offsets[axis]
            upper = lower + tensor.shape[axis]
            virtual_slices.append(slice(lower, upper))
            offsets[axis] = upper
        out[tuple(virtual_slices + [slice(None)] * num_open_legs)] = tensor

    return out


def ttno_linear_combination(
    ttnos: Sequence[TTNO],
    coeffs: Number | Sequence[Number] = 1.0,
) -> TTNO:
    """
    Build ``sum_i coeffs[i] * ttnos[i]`` by direct-summing TTNO virtual bonds.

    The input TTNOs must use the same node identifiers, root, hierarchy, and
    local physical dimensions. Coefficients are placed on the root tensor only.
    """
    ttnos = list(ttnos)
    _check_compatible_ttnos(ttnos)
    coeffs = _as_coeff_list(coeffs, len(ttnos))

    if len(ttnos) == 1:
        return ttnos[0].scale(coeffs[0], inplace=False)

    ref_ttno = ttnos[0]
    new_tensors: dict[str, np.ndarray] = {}

    for node_id, ref_node in ref_ttno.nodes.items():
        node_tensors = []
        for coeff, ttno in zip(coeffs, ttnos):
            node = ttno.nodes[node_id]
            tensor = ttno.tensors[node_id]
            tensor = np.transpose(tensor, axes=relative_leg_permutation(ref_node, node))
            if node_id == ref_ttno.root_id:
                tensor = coeff * tensor
            node_tensors.append(tensor)

        new_tensors[node_id] = _block_diag_keep_open(node_tensors, num_open_legs=2)

    return TTNO.from_tensors(ref_ttno, new_tensors)


def add_ttnos(
    ttno1: TTNO,
    ttno2: TTNO,
    c1: Number = 1.0,
    c2: Number = 1.0,
) -> TTNO:
    """Return ``c1 * ttno1 + c2 * ttno2``."""
    return ttno_linear_combination([ttno1, ttno2], [c1, c2])


def identity_ttno_like(ttno: TTNO, dtype: np.dtype | None = None) -> TTNO:
    """Construct an identity TTNO with the same tree and local dimensions."""
    if ttno.root_id is None:
        raise ValueError("Cannot build identity for an empty TTNO.")

    if dtype is None:
        dtype = np.result_type(*(tensor.dtype for tensor in ttno.tensors.values()))

    tensors: dict[str, np.ndarray] = {}
    for node_id, node in ttno.nodes.items():
        open_dims = node.open_dimensions()
        if len(open_dims) != 2:
            raise ValueError(
                f"Node {node_id!r} has {len(open_dims)} open legs; "
                "a TTNO node should have exactly two open legs."
            )
        if open_dims[0] != open_dims[1]:
            raise ValueError(
                f"Node {node_id!r} has non-square local dimensions {open_dims}."
            )

        shape = (1,) * node.nvirt_legs() + tuple(open_dims)
        tensor = np.zeros(shape, dtype=dtype)
        tensor[(0,) * node.nvirt_legs() + (slice(None), slice(None))] = np.eye(
            open_dims[0],
            dtype=dtype,
        )
        tensors[node_id] = tensor

    return TTNO.from_tensors(ttno, tensors)


def shift_ttno_by_identity(ttno: TTNO, shift: Number) -> TTNO:
    """Return ``ttno + shift * I`` as a TTNO."""
    if shift == 0:
        return deepcopy(ttno)
    identity = identity_ttno_like(ttno)
    return add_ttnos(ttno, identity, c1=1.0, c2=shift)


def subtract_identity_shift(ttno: TTNO, shift: Number) -> TTNO:
    """Return ``ttno - shift * I`` as a TTNO."""
    return shift_ttno_by_identity(ttno, -shift)


def scale_ttno_with_energy_window_shifted(
    hamiltonian: TTNO,
    E_min: float,
    E_max: float,
    W_prime: float = 0.9875,
    safety_factor: float = 1.10,
) -> tuple[TTNO, float, float]:
    """
    Scale a TTNO and directly subtract the Chebyshev identity shift.

    Returns ``((1 / a) * H - shift * I, a, shift)`` with
    ``a = safety_factor * (E_max - E_min) / (2 * W_prime)`` and
    ``shift = E_min / a + W_prime``.
    """
    if not np.isfinite(E_min) or not np.isfinite(E_max) or E_max <= E_min:
        raise ValueError(f"Invalid energy window: E_min={E_min}, E_max={E_max}")
    if not np.isfinite(W_prime) or W_prime <= 0:
        raise ValueError(f"W_prime must be finite and positive, got {W_prime}")
    if safety_factor < 1.0:
        raise ValueError(f"safety_factor must be >= 1, got {safety_factor}")

    a = safety_factor * (E_max - E_min) / (2.0 * W_prime)
    shift = (E_min / a) + W_prime

    h_scaled = deepcopy(hamiltonian)
    h_scaled.tensors[h_scaled.root_id] = h_scaled.tensors[h_scaled.root_id] * (1.0 / a)
    h_scaled_shifted = subtract_identity_shift(h_scaled, shift)

    return h_scaled_shifted, float(a), float(shift)
