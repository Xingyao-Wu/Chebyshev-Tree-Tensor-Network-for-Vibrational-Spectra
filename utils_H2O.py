from typing import List
import numpy as np
from pytreenet.core.node import Node
from pytreenet.ttns import TTNS
from pytreenet.util.tensor_splitting import SplitMode

from Experiment.utils import get_harmonic_oscillator_orbitals, single_voxel_block_diag


def random_mps_h2o_harmonic_oscillator_0(
    physical_dim: List[int],
    omega: List[float],
    orb_state: np.ndarray,
    dtype: np.dtype = np.float64,
) -> TTNS:
    """
    H2O harmonic oscillator product-state TTNS.

    physical_dim: e.g. [13, 13, 13]
    omega: three harmonic frequencies
    orb_state: shape (1, 3), e.g. [[0, 0, 0]] or [[1, 0, 0]]
    """

    nsite = len(physical_dim)

    if nsite != 3:
        raise ValueError("H2O should have exactly 3 vibrational modes.")

    if orb_state.shape[1] != 3:
        raise ValueError("orb_state should have shape (m, 3).")

    # chain: site0 - site1 - site2
    # number of virtual neighbours for each physical tensor
    nneighbour = [1, 2, 1]

    ho_tensors = get_harmonic_oscillator_orbitals(
        physical_dim,
        omega,
        orb_state,
    )

    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1, 1)

    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(
            ho_tensors[i],
            nneighbour[i],
        )
        ho_tensors[i] = ho_tensors[i].astype(dtype)

    nodes = [
        (Node(tensor=ho_tensors[i], identifier=f"site{i}"), ho_tensors[i])
        for i in range(nsite)
    ]

    state = TTNS()

    # site0 -- site1 -- site2
    state.add_root(nodes[0][0], nodes[0][1])
    state.add_child_to_parent(nodes[1][0], nodes[1][1], 0, "site0", 0)
    state.add_child_to_parent(nodes[2][0], nodes[2][1], 0, "site1", 1)

    state.canonical_form(state.root_id, mode=SplitMode.KEEP)
    state.normalize()

    return state