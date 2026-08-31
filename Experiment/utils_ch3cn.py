from typing import List
import numpy as np
from pytreenet.core.node import Node
from pytreenet.random.random_node import random_tensor_node
from pytreenet.ttns import TTNS
from pytreenet.util.tensor_splitting import SplitMode
from pytreenet.special_ttn import ForkTreeProductState
from Experiment.utils import get_harmonic_oscillator_orbitals, single_voxel_block_diag

def create_node_mapping(node_order: list) -> dict:
    """
    Create a node mapping dictionary where keys are 0-11 and values are from the provided list.
    
    Args:
        node_order (list): List of integers specifying the new order of nodes
        
    Returns:
        dict: Mapping dictionary where keys are 0-11 and values are from node_order
    """
    if len(node_order) != 12:
        raise ValueError("node_order must contain exactly 12 values")
        
    if sorted(node_order) != list(range(12)):
        raise ValueError("node_order must contain all integers from 0 to 11")
        
    sort_idx = np.argsort(node_order)              # positions after sorting
    inv = np.empty_like(sort_idx)
    inv[sort_idx] = np.arange(12)                  # original index -> sorted position
    return {i: int(inv[i]) for i in range(12)}


def random_mps(physical_dim: List[int], virtual_dim: int, weights: List[float], node_order: List[int]) -> TTNS:
    """
    Generate a random MPS with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_mps: TTNS
            The random MPS.
    """
    nsite = len(physical_dim)
    shapes = [[virtual_dim]]+[[virtual_dim,virtual_dim]]*(nsite-2)+[[virtual_dim]]
    if nsite != len(shapes):
        raise ValueError("The length of physical_dim and shapes must be the same")
    sort_idx = np.argsort(node_order)
    shapes = [shapes[i] for i in sort_idx]
    new_shapes = []
    for i in range(nsite):
        new_shapes.append(shapes[i]+[physical_dim[i]])
        
    nodes = [random_tensor_node(shape, identifier="site"+str(i))
             for i, shape in enumerate(new_shapes)]
    
    node_mapping = create_node_mapping(node_order)
    random_mps = TTNS()
    random_mps.add_root(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1])
    random_mps.add_child_to_parent(nodes[node_mapping[1]][0],nodes[node_mapping[1]][1],0,f"site{node_mapping[0]}",0)
    for i in range(nsite-2):
        random_mps.add_child_to_parent(nodes[node_mapping[i+2]][0],nodes[node_mapping[i+2]][1],0,f"site{node_mapping[i+1]}",1)
    random_mps.canonical_form(f"site{node_mapping[nsite-1]}", mode=SplitMode.KEEP)
    random_mps.normalize()
    return random_mps

def random_mps_harmonic_oscillator_0(physical_dim: List[int], omega: List[float], orb_state: np.ndarray, node_order: List[int], dtype: np.dtype = np.float64) -> TTNS:
    """
    Generate a random MPS with given physical dimensions and virtual dimensions are set to be one.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        omega: List[float]
            The frequency of the harmonic oscillator.
        orb_state: np.ndarray
            The state of the harmonic oscillator.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_mps: TTNS
            The random MPS.
    """
    nsite = len(physical_dim)
    nneighbour = [1]+[2]*(nsite-2)+[1]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)
    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])      
        ho_tensors[i] = ho_tensors[i].astype(dtype)  

    sort_idx = np.argsort(node_order)
    ho_tensors = [ho_tensors[i] for i in sort_idx]
        
    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    node_mapping = create_node_mapping(node_order)
    random_mps = TTNS()
    random_mps.add_root(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1])
    random_mps.add_child_to_parent(nodes[node_mapping[1]][0],nodes[node_mapping[1]][1],0,f"site{node_mapping[0]}",0)
    for i in range(nsite-2):
        random_mps.add_child_to_parent(nodes[node_mapping[i+2]][0],nodes[node_mapping[i+2]][1],0,f"site{node_mapping[i+1]}",1)
    random_mps.canonical_form(f"site{node_mapping[nsite-1]}", mode=SplitMode.KEEP)
    random_mps.normalize()
    return random_mps


def random_leafonly(physical_dim: List[int], virtual_dim: int, weights: List[float], node_order: List[int]) -> TTNS:
    """
    Generate a random tree like the one in Larsson's paper with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: TTNS
            The random TTNS.
    """
    m = virtual_dim
    shapes = [[m],[m],[m],[m],[m],[m],[m],[m],[m],[m],[m],[m]]
    shapes_ancillary = [[m,m,1]]+[[m,m,m,1]]*10
    if len(physical_dim) != len(shapes):
        raise ValueError("The length of physical_dim and shapes must be the same")
    sort_idx = np.argsort(node_order)
    shapes = [shapes[i] for i in sort_idx]
    for s, d in zip(shapes, physical_dim):
        s.append(d)
    shapes.extend(shapes_ancillary)
    nodes = [random_tensor_node(shape, identifier="site"+str(i))
             for i, shape in enumerate(shapes)]
    node_mapping = create_node_mapping(node_order)
    random_ttns = TTNS()
    random_ttns.add_root(nodes[12][0], nodes[12][1])
    random_ttns.add_child_to_parent(nodes[13][0],nodes[13][1],0,"site12",0)
    random_ttns.add_child_to_parent(nodes[14][0],nodes[14][1],0,"site13",1)
    random_ttns.add_child_to_parent(nodes[15][0],nodes[15][1],0,"site14",1)
    random_ttns.add_child_to_parent(nodes[16][0],nodes[16][1],0,"site13",2)
    random_ttns.add_child_to_parent(nodes[17][0],nodes[17][1],0,"site16",1)
    random_ttns.add_child_to_parent(nodes[18][0],nodes[18][1],0,"site16",2)
    random_ttns.add_child_to_parent(nodes[19][0],nodes[19][1],0,"site12",1)
    random_ttns.add_child_to_parent(nodes[20][0],nodes[20][1],0,"site19",1)
    random_ttns.add_child_to_parent(nodes[21][0],nodes[21][1],0,"site20",1)
    random_ttns.add_child_to_parent(nodes[22][0],nodes[22][1],0,"site19",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[0]][0],nodes[node_mapping[0]][1],0,"site14",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[4]][0],nodes[node_mapping[4]][1],0,"site15",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[5]][0],nodes[node_mapping[5]][1],0,"site15",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[6]][0],nodes[node_mapping[6]][1],0,"site17",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[7]][0],nodes[node_mapping[7]][1],0,"site17",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[8]][0],nodes[node_mapping[8]][1],0,"site18",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[9]][0],nodes[node_mapping[9]][1],0,"site18",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[2]][0],nodes[node_mapping[2]][1],0,"site20",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[1]][0],nodes[node_mapping[1]][1],0,"site21",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[3]][0],nodes[node_mapping[3]][1],0,"site21",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[10]][0],nodes[node_mapping[10]][1],0,"site22",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[11]][0],nodes[node_mapping[11]][1],0,"site22",2)
    random_ttns.canonical_form(random_ttns.root_id, mode=SplitMode.KEEP)
    random_ttns.normalize()
    # random_ttns.init_multistate_center()
    return random_ttns

def random_leafonly_harmonic_oscillator_0(physical_dim: List[int], omega: float, orb_state: List[float], node_order: List[int], dtype: np.dtype = np.float64) -> TTNS:
    """
    Generate a random leafonly tree with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        omega: float
            The frequency of the harmonic oscillator.
        orb_state: List[float]
            The state of the harmonic oscillator.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: TTNS
            The random TTNS.
    """
    nsite = len(physical_dim)
    sort_idx = np.argsort(node_order)
    nneighbour = [1,1,1,1,1,1,1,1,1,1,1,1]
    nneighbour = [nneighbour[i] for i in sort_idx]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    m = orb_state.shape[0]

    if orb_state.shape[0] == 1:
        # m = 5
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)
            # random_col = np.random.randn(ho_tensors[i].shape[0],m-1)*1e-3
            # ho_tensors[i] = np.concatenate([ho_tensors[i], random_col], axis=1)
    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])     
        ho_tensors[i] = ho_tensors[i].astype(dtype) 
        # ho_tensors[i] += (np.random.randn(*ho_tensors[i].shape) + 1j * np.random.randn(*ho_tensors[i].shape)) * 1e-2

    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    shapes_ancillary = [[m,m,1]]+[[m,m,m,1]]*10
    nodes.extend([random_tensor_node(shape, identifier="site"+str(i+nsite), dtype=dtype)
             for i, shape in enumerate(shapes_ancillary)])
    
    node_mapping = create_node_mapping(node_order)
    
    random_ttns = TTNS()
    random_ttns.add_root(nodes[12][0], nodes[12][1])
    random_ttns.add_child_to_parent(nodes[13][0],nodes[13][1],0,"site12",0)
    random_ttns.add_child_to_parent(nodes[14][0],nodes[14][1],0,"site13",1)
    random_ttns.add_child_to_parent(nodes[15][0],nodes[15][1],0,"site14",1)
    random_ttns.add_child_to_parent(nodes[16][0],nodes[16][1],0,"site13",2)
    random_ttns.add_child_to_parent(nodes[17][0],nodes[17][1],0,"site16",1)
    random_ttns.add_child_to_parent(nodes[18][0],nodes[18][1],0,"site16",2)
    random_ttns.add_child_to_parent(nodes[19][0],nodes[19][1],0,"site12",1)
    random_ttns.add_child_to_parent(nodes[20][0],nodes[20][1],0,"site19",1)
    random_ttns.add_child_to_parent(nodes[21][0],nodes[21][1],0,"site20",1)
    random_ttns.add_child_to_parent(nodes[22][0],nodes[22][1],0,"site19",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[0]][0],nodes[node_mapping[0]][1],0,"site14",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[4]][0],nodes[node_mapping[4]][1],0,"site15",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[5]][0],nodes[node_mapping[5]][1],0,"site15",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[6]][0],nodes[node_mapping[6]][1],0,"site17",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[7]][0],nodes[node_mapping[7]][1],0,"site17",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[8]][0],nodes[node_mapping[8]][1],0,"site18",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[9]][0],nodes[node_mapping[9]][1],0,"site18",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[2]][0],nodes[node_mapping[2]][1],0,"site20",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[1]][0],nodes[node_mapping[1]][1],0,"site21",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[3]][0],nodes[node_mapping[3]][1],0,"site21",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[10]][0],nodes[node_mapping[10]][1],0,"site22",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[11]][0],nodes[node_mapping[11]][1],0,"site22",2)
    random_ttns.canonical_form(random_ttns.root_id, mode=SplitMode.KEEP)
    random_ttns.normalize()
    return random_ttns

def random_threetree(physical_dim: List[int], virtual_dim: int, weights: List[float], node_order: List[int]) -> TTNS:
    """
    Generate a random three-leg tree with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: MultiTTNS
            The random TTNS.
    """
    m = virtual_dim
    shapes = [[m],[m,m],[m,m],[m,m],
              [m,m],[m,m],[m,m],[m],    
              [m,m],[m,m],[m,m],[m]]
    
    shapes_ancillary = [m,m,m,1]
    if len(physical_dim) != len(shapes):
        raise ValueError("The length of physical_dim and shapes must be the same")
    sort_idx = np.argsort(node_order)
    shapes = [shapes[i] for i in sort_idx]
    for s, d in zip(shapes, physical_dim):
        s.append(d)
    shapes.append(shapes_ancillary)
       
    node_mapping = create_node_mapping(node_order)
        
    nodes = [random_tensor_node(shape, identifier="site"+str(i))
             for i, shape in enumerate(shapes)]
    random_ttns = TTNS()
    random_ttns.add_root(nodes[12][0], nodes[12][1])
    random_ttns.add_child_to_parent(nodes[node_mapping[3]][0], nodes[node_mapping[3]][1], 0, "site12", 0)
    random_ttns.add_child_to_parent(nodes[node_mapping[2]][0],nodes[node_mapping[2]][1],0,f"site{node_mapping[3]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[1]][0],nodes[node_mapping[1]][1],0,f"site{node_mapping[2]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[0]][0],nodes[node_mapping[0]][1],0,f"site{node_mapping[1]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[4]][0],nodes[node_mapping[4]][1],0,"site12",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[5]][0],nodes[node_mapping[5]][1],0,f"site{node_mapping[4]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[6]][0],nodes[node_mapping[6]][1],0,f"site{node_mapping[5]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[7]][0],nodes[node_mapping[7]][1],0,f"site{node_mapping[6]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[8]][0],nodes[node_mapping[8]][1],0,"site12",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[9]][0],nodes[node_mapping[9]][1],0,f"site{node_mapping[8]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[10]][0],nodes[node_mapping[10]][1],0,f"site{node_mapping[9]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[11]][0],nodes[node_mapping[11]][1],0,f"site{node_mapping[10]}",1)
    random_ttns.canonical_form(random_ttns.root_id, mode=SplitMode.KEEP)
    random_ttns.normalize()
    return random_ttns

def random_threetree_harmonic_oscillator_0(physical_dim: List[int], omega: float, orb_state: List[float], node_order: List[int], dtype: np.dtype = np.float64) -> TTNS:
    """
    Generate a random three-leg tree with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: MultiTTNS
            The random TTNS.
    """
    nsite = len(physical_dim)
    sort_idx = np.argsort(node_order)
    nneighbour = [1,2,2,2,2,2,2,1,2,2,2,1]
    nneighbour = [nneighbour[i] for i in sort_idx]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    m = orb_state.shape[0]

    if orb_state.shape[0] == 1:
        # m = 5
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)
            # random_col = np.random.randn(ho_tensors[i].shape[0],m-1)*1e-3
            # ho_tensors[i] = np.concatenate([ho_tensors[i], random_col], axis=1)
    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])     
        ho_tensors[i] = ho_tensors[i].astype(dtype) 
        # ho_tensors[i] += (np.random.randn(*ho_tensors[i].shape) + 1j * np.random.randn(*ho_tensors[i].shape)) * 1e-2
    
    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    shapes_ancillary = [[m,m,m,1]]
    nodes.extend([random_tensor_node(shape, identifier="site"+str(i+nsite), dtype=dtype)
             for i, shape in enumerate(shapes_ancillary)])
    
    node_mapping = create_node_mapping(node_order)
    
    random_ttns = TTNS()
    random_ttns.add_root(nodes[12][0], nodes[12][1])
    random_ttns.add_child_to_parent(nodes[node_mapping[3]][0], nodes[node_mapping[3]][1], 0, "site12", 0)
    random_ttns.add_child_to_parent(nodes[node_mapping[2]][0],nodes[node_mapping[2]][1],0,f"site{node_mapping[3]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[1]][0],nodes[node_mapping[1]][1],0,f"site{node_mapping[2]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[0]][0],nodes[node_mapping[0]][1],0,f"site{node_mapping[1]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[4]][0],nodes[node_mapping[4]][1],0,"site12",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[5]][0],nodes[node_mapping[5]][1],0,f"site{node_mapping[4]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[6]][0],nodes[node_mapping[6]][1],0,f"site{node_mapping[5]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[7]][0],nodes[node_mapping[7]][1],0,f"site{node_mapping[6]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[8]][0],nodes[node_mapping[8]][1],0,"site12",2)
    random_ttns.add_child_to_parent(nodes[node_mapping[9]][0],nodes[node_mapping[9]][1],0,f"site{node_mapping[8]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[10]][0],nodes[node_mapping[10]][1],0,f"site{node_mapping[9]}",1)
    random_ttns.add_child_to_parent(nodes[node_mapping[11]][0],nodes[node_mapping[11]][1],0,f"site{node_mapping[10]}",1)
    random_ttns.canonical_form(random_ttns.root_id, mode=SplitMode.KEEP)
    random_ttns.normalize()
    return random_ttns

def random_fork_tree(physical_dim: List[int], omega: float, orb_state: List[float], node_order: List[int], dtype: np.dtype = np.float64, nbranch: int=3) -> TTNS:
    """
    Generate a random fork tree with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        omega: float
            The frequency of the harmonic oscillator.
        orb_state: List[float]
            The state of the harmonic oscillator.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: MultiTTNS
            The random TTNS.
    """
    nsite = len(physical_dim)
    sort_idx = np.argsort(node_order)
    nnpb = nsite//nbranch # number of node per branch
    remainder = nsite % nbranch

    nneighbour = []
    for i in range(nbranch-1):
        nneighbour.extend([2]*(nnpb-1)+[1])
    nneighbour.extend([2]*(nnpb-1+remainder)+[1])
    nneighbour = [nneighbour[i] for i in sort_idx]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    m = orb_state.shape[0]

    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)
    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])     
        ho_tensors[i] = ho_tensors[i].astype(dtype) 
    sub_tensors=[]
    sub_id=[]
    for i in range(nbranch-1):
        sub_tensors.append(ho_tensors[i*nnpb:(i+1)*nnpb])
        sub_id.append([f"site{j}" for j in range(i*nnpb,(i+1)*nnpb)])
    sub_tensors.append(ho_tensors[(nbranch-1)*nnpb:])
    sub_id.append([f"site{j}" for j in range((nbranch-1)*nnpb,nsite)])
    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    shapes_ancillary = [[m,m,1]]+[[m,m,m,1]]*(nbranch-2)+[[m,m,1]]
    # nodes.extend([random_tensor_node(shape, identifier="site"+str(i+nsite), dtype=dtype)
            #  for i, shape in enumerate(shapes_ancillary)])
    main_tensors = [random_tensor_node(shape, identifier="site"+str(i+nsite), dtype=dtype)[1]
             for i, shape in enumerate(shapes_ancillary)]
    node_mapping = create_node_mapping(node_order)
    main_id = [f"site{nsite+i}" for i in range(nbranch)]
    random_ttns = ForkTreeProductState.from_tensors(main_tensors, sub_tensors, main_identifier_prefix=main_id, subchain_identifier_prefix=sub_id)
    # random_ttns.root_id = 'main_0'
    random_ttns.canonical_form(random_ttns.root_id, mode=SplitMode.KEEP)
    random_ttns.normalize()
    return random_ttns

def random_t3ns_harmonic_oscillator(physical_dim: List[int], omega: float, orb_state: List[float], node_order: List[int], dtype: np.dtype = np.float64, nbranch: int=3) -> TTNS:
    """
    Generate a random t3ns with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: MultiTTNS
            The random TTNS.
    """    
    nsite = len(physical_dim)
    sort_idx = np.argsort(node_order)
    nneighbour = [2,2,1,1,2,1,1,2,1,1,2,1]
    nneighbour = [nneighbour[i] for i in sort_idx]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    # ho_tensors = [ho_tensors[i] for i in sort_idx]
    m = orb_state.shape[0]

    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)
    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])     
        ho_tensors[i] = ho_tensors[i].astype(dtype) 

    # sort_idx = node_order #np.argsort(node_order)
    # ho_tensors = [ho_tensors[i] for i in sort_idx]
    
    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    shapes_ancillary = [[m,m,m,1],[m,m,m,1],[m,m,m,1],[m,m,m,1],[m,m,m,1],[m,m,1]]
    nodes.extend([random_tensor_node(shape, identifier="site"+str(i+nsite), dtype=dtype)
             for i, shape in enumerate(shapes_ancillary)])
    
    node_mapping = create_node_mapping(node_order)
    random_ttns = TTNS()
    random_ttns.add_root(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1])
    random_ttns.add_child_to_parent(nodes[12][0], nodes[12][1], 0, f"site{node_mapping[0]}", 0)
    random_ttns.add_child_to_parent(nodes[node_mapping[1]][0], nodes[node_mapping[1]][1], 0, "site12", 1)
    random_ttns.add_child_to_parent(nodes[14][0], nodes[14][1], 0, f"site{node_mapping[1]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[2]][0], nodes[node_mapping[2]][1], 0, "site14", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[3]][0], nodes[node_mapping[3]][1], 0, "site14", 2)
    random_ttns.add_child_to_parent(nodes[node_mapping[4]][0], nodes[node_mapping[4]][1], 0, "site12", 2)
    random_ttns.add_child_to_parent(nodes[15][0], nodes[15][1], 0, f"site{node_mapping[4]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[5]][0], nodes[node_mapping[5]][1], 0, "site15", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[6]][0], nodes[node_mapping[6]][1], 0, "site15", 2)
    random_ttns.add_child_to_parent(nodes[13][0], nodes[13][1], 0, f"site{node_mapping[0]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[7]][0], nodes[node_mapping[7]][1], 0, "site13", 1)
    random_ttns.add_child_to_parent(nodes[16][0], nodes[16][1], 0, f"site{node_mapping[7]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[8]][0], nodes[node_mapping[8]][1], 0, "site16", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[9]][0], nodes[node_mapping[9]][1], 0, "site16", 2)
    random_ttns.add_child_to_parent(nodes[node_mapping[10]][0], nodes[node_mapping[10]][1], 0, "site13", 2)
    random_ttns.add_child_to_parent(nodes[17][0], nodes[17][1], 0, f"site{node_mapping[10]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[11]][0], nodes[node_mapping[11]][1], 0, "site17", 1)
    random_ttns.canonical_form(random_ttns.root_id, mode=SplitMode.KEEP)
    random_ttns.normalize()
    return random_ttns
    
def random_x_harmonic_oscillator(physical_dim: List[int], omega: float, orb_state: List[float], node_order: List[int], dtype: np.dtype = np.float64, nbranch: int=3) -> TTNS:
    """
    Generate a random t3ns with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: MultiTTNS
            The random TTNS.
    """    
    nsite = len(physical_dim)
    sort_idx = np.argsort(node_order)
    nneighbour = [2,2,1,2,2,1,2,2,1,2,2,1]
    nneighbour = [nneighbour[i] for i in sort_idx]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    # ho_tensors = [ho_tensors[i] for i in sort_idx]
    m = orb_state.shape[0]

    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)
    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])     
        ho_tensors[i] = ho_tensors[i].astype(dtype) 

    # sort_idx = node_order #np.argsort(node_order)
    # ho_tensors = [ho_tensors[i] for i in sort_idx]
    
    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    shapes_ancillary = [[m,m,m,1],[m,m,m,1]]
    nodes.extend([random_tensor_node(shape, identifier="site"+str(i+nsite), dtype=dtype)
             for i, shape in enumerate(shapes_ancillary)])
    
    node_mapping = create_node_mapping(node_order)
    random_ttns = TTNS()
    random_ttns.add_root(nodes[12][0], nodes[12][1])
    random_ttns.add_child_to_parent(nodes[13][0], nodes[13][1], 0, "site12", 0)
    random_ttns.add_child_to_parent(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1],0, "site12", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[1]][0], nodes[node_mapping[1]][1], 0, f"site{node_mapping[0]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[2]][0], nodes[node_mapping[2]][1], 0, f"site{node_mapping[1]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[3]][0], nodes[node_mapping[3]][1],0, "site12", 2)
    random_ttns.add_child_to_parent(nodes[node_mapping[4]][0], nodes[node_mapping[4]][1], 0, f"site{node_mapping[3]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[5]][0], nodes[node_mapping[5]][1], 0, f"site{node_mapping[4]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[6]][0], nodes[node_mapping[6]][1],0, "site13", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[7]][0], nodes[node_mapping[7]][1], 0, f"site{node_mapping[6]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[8]][0], nodes[node_mapping[8]][1], 0, f"site{node_mapping[7]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[9]][0], nodes[node_mapping[9]][1],0, "site13", 2)
    random_ttns.add_child_to_parent(nodes[node_mapping[10]][0], nodes[node_mapping[10]][1], 0, f"site{node_mapping[9]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[11]][0], nodes[node_mapping[11]][1], 0, f"site{node_mapping[10]}", 1)
    random_ttns.canonical_form(random_ttns.root_id, mode=SplitMode.KEEP)
    random_ttns.normalize()
    return random_ttns

def random_twin_tree_harmonic_oscillator(physical_dim: List[int], omega: float, orb_state: List[float], node_order: List[int], dtype: np.dtype = np.float64, nbranch: int=3) -> TTNS:
    """
    Generate a random t3ns with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: MultiTTNS
            The random TTNS.
    """    
    nsite = len(physical_dim)
    sort_idx = np.argsort(node_order)
    nneighbour = [2,2,2,2,1,1,1,1,1,1,1,1]
    nneighbour = [nneighbour[i] for i in sort_idx]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    # ho_tensors = [ho_tensors[i] for i in sort_idx]
    m = orb_state.shape[0]

    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)
    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])     
        ho_tensors[i] = ho_tensors[i].astype(dtype) 

    # sort_idx = node_order #np.argsort(node_order)
    # ho_tensors = [ho_tensors[i] for i in sort_idx]
    
    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    shapes_ancillary = [[m,m,m,1],[m,m,m,1],[m,m,m,1],[m,m,m,1],[m,m,m,1],[m,m,m,1]]
    nodes.extend([random_tensor_node(shape, identifier="site"+str(i+nsite), dtype=dtype)
             for i, shape in enumerate(shapes_ancillary)])
    
    node_mapping = create_node_mapping(node_order)
    random_ttns = TTNS()
    # random_ttns.add_root(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1])
    random_ttns.add_root(nodes[12][0], nodes[12][1])
    random_ttns.add_child_to_parent(nodes[13][0], nodes[13][1], 0, "site12", 0)
    random_ttns.add_child_to_parent(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1],0, "site12", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[1]][0], nodes[node_mapping[1]][1], 0, "site12", 2)
    random_ttns.add_child_to_parent(nodes[node_mapping[2]][0], nodes[node_mapping[2]][1],0, "site13", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[3]][0], nodes[node_mapping[3]][1], 0, "site13", 2)
    random_ttns.add_child_to_parent(nodes[14][0], nodes[14][1], 0, f"site{node_mapping[0]}", 1)
    random_ttns.add_child_to_parent(nodes[15][0], nodes[15][1], 0, f"site{node_mapping[1]}", 1)
    random_ttns.add_child_to_parent(nodes[16][0], nodes[16][1], 0, f"site{node_mapping[2]}", 1)
    random_ttns.add_child_to_parent(nodes[17][0], nodes[17][1], 0, f"site{node_mapping[3]}", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[4]][0], nodes[node_mapping[4]][1], 0, "site14", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[5]][0], nodes[node_mapping[5]][1], 0, "site14", 2)
    random_ttns.add_child_to_parent(nodes[node_mapping[6]][0], nodes[node_mapping[6]][1], 0, "site15", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[7]][0], nodes[node_mapping[7]][1], 0, "site15", 2)
    random_ttns.add_child_to_parent(nodes[node_mapping[8]][0], nodes[node_mapping[8]][1], 0, "site16", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[9]][0], nodes[node_mapping[9]][1], 0, "site16", 2)
    random_ttns.add_child_to_parent(nodes[node_mapping[10]][0], nodes[node_mapping[10]][1], 0, "site17", 1)
    random_ttns.add_child_to_parent(nodes[node_mapping[11]][0], nodes[node_mapping[11]][1], 0, "site17", 2)
    random_ttns.canonical_form(random_ttns.root_id, mode=SplitMode.KEEP)
    random_ttns.normalize()
    return random_ttns
