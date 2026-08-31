from typing import List, Union, Tuple, Dict, Callable
from collections import Counter
import numpy as np
from fractions import Fraction
from scipy.special import roots_hermite
from pytreenet.ttno import TreeTensorNetworkOperator, TTNOFinder
from pytreenet.operators import Hamiltonian, TensorProduct
from pytreenet.ttns import TreeTensorNetworkState
from Experiment.potentials.potential_harmonic import get_potential_energy_harmonic

def get_laplacian(xs: np.ndarray) -> np.ndarray:
    """
    Get the Laplacian matrix for a given set of points.
    Input:
        xs: np.ndarray
            The points to compute the Laplacian matrix.
    Output:
        lp: np.ndarray
            The Laplacian matrix.
    """

    N = len(xs)
    lp = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                lp[i,j] = (-1)**(i - j)*(2*(xs[i] - xs[j])**(-2) - 0.5)
            else:
                lp[i,j] = 1.0/6 * (4*N - 1 - 2*xs[i]**2)

    return 0.5*lp

def get_name_list(N: List[int], potential_energy_func: Callable) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
    """
    Get the name list and conversion dictionary for the Hamiltonian.
    Input:
        N: List[int]
            The number of grid points for each basis function.
        potential_energy_func: Callable
            The function to get the potential energy.
    Output:
        name_list: List[Dict[str, Union[str, float]]]
            The name list for the Hamiltonian.
        conversion_dict: Dict[str, np.ndarray]
            The conversion dictionary for the Hamiltonian.
    """
    w, k3, k4 = potential_energy_func()
    unique_N = np.unique(N)
    conversion_dict = {}
    name_list = []
    
    # construct conversion dictionary
    for n in unique_N:
        x,_ = roots_hermite(n)
        t = get_laplacian(x)
        conversion_dict["I"+str(n)] = np.eye(n)
        conversion_dict["t"+str(n)] = t
        q = np.diag(x)
        conversion_dict["q"+str(n)] = q
        q2 = np.diag(x**2)
        conversion_dict["q"+str(n)+"^2"] = q2
        q3 = np.diag(x**3)
        conversion_dict["q"+str(n)+"^3"] = q3
        q4 = np.diag(x**4)
        conversion_dict["q"+str(n)+"^4"] = q4
    # kinetic part
    for i in range(len(w)):
        name_list.append({"site"+str(i): f"t{N[i]}", "coeff": w[i]})
        # conversion_dict["t"+str(N[i])] =conversion_dict["t"+str(N[i])]
        
    # harmonic part
    for i in range(len(w)):
        name_list.append({"site"+str(i): f"q{N[i]}^2", "coeff": w[i]*0.5})
        # conversion_dict["q"+str(N[i])+"^2"] = 0.5*w[i]*conversion_dict["q"+str(N[i])+"^2"]
        
        
    # three-body potential part
    k3_indices = np.nonzero(k3)
    k3_values = k3[k3_indices]
    for i in range(len(k3_values)):
        ind_q1 = k3_indices[0][i]
        ind_q2 = k3_indices[1][i]
        ind_q3 = k3_indices[2][i]
        indices = [ind_q1, ind_q2, ind_q3]
        count_dict = Counter(indices)
        name_dict = {}
        merge_coeff = False
        for idx, count in count_dict.items():
            if count > 1:
                if merge_coeff:
                    name_dict["site"+str(idx)] = f"q{N[idx]}^{count}"
                else:
                    name_dict["site"+str(idx)] = f"q{N[idx]}^{count}"
                    merge_coeff = True
                    conversion_dict["q"+str(N[idx])+"^"+str(count)] = conversion_dict["q"+str(N[idx])+"^"+str(count)]
            elif merge_coeff:
                name_dict["site"+str(idx)] = f"q{N[idx]}"
            else:
                name_dict["site"+str(idx)] = f"q{N[idx]}"
                # conversion_dict["q"+str(N[idx])] = conversion_dict["q"+str(N[idx])]
                merge_coeff = True
        name_dict["coeff"] = k3_values[i]
        name_list.append(name_dict)
        
    # four-body potential part
    k4_indices = np.nonzero(k4)
    k4_values = k4[k4_indices]
    for i in range(len(k4_values)):
        ind_q1 = k4_indices[0][i]
        ind_q2 = k4_indices[1][i]
        ind_q3 = k4_indices[2][i]
        ind_q4 = k4_indices[3][i]
        indices = [ind_q1, ind_q2, ind_q3, ind_q4]
        count_dict = Counter(indices)        
        name_dict = {}
        merge_coeff = False
        for idx, count in count_dict.items():
            if count > 1:
                if merge_coeff:
                    name_dict["site"+str(idx)] = f"q{N[idx]}^{count}"
                else:
                    name_dict["site"+str(idx)] = f"q{N[idx]}^{count}"
                    merge_coeff = True
                    # conversion_dict["q"+str(N[idx])+"^"+str(count)] = conversion_dict["q"+str(N[idx])+"^"+str(count)]
            elif merge_coeff:
                name_dict["site"+str(idx)] = f"q{N[idx]}"
            else:
                name_dict["site"+str(idx)] = f"q{N[idx]}"
                # conversion_dict["q"+str(N[idx])+"_"+str(k4_values[i])] = conversion_dict["q"+str(N[idx])]*k4_values[i]
                merge_coeff = True
            name_dict["coeff"] = k4_values[i]
                
        name_list.append(name_dict)
        
    return name_list, conversion_dict

def get_name_list_harmonic(num_harmonic_oscillator:int, physical_dim: int=6) -> Tuple[List[Dict[str, Union[str, float]]], Dict[str, np.ndarray]]:
    """
    Get the name list and conversion dictionary for the Hamiltonian.
    Input:
        num_harmonic_oscillator: int
            The number of coupled harmonic oscillators.
        physical_dim: int
            The physical dimension of the system.
    Output:
        name_list: List[Dict[str, Union[str, float]]]
            The name list for the Hamiltonian.
        conversion_dict: Dict[str, np.ndarray]
            The conversion dictionary for the Hamiltonian.
    """
    w_indices, Aij = get_potential_energy_harmonic(D=num_harmonic_oscillator)
    conversion_dict = {}
    name_list = []
    
    # construct conversion dictionary
    n = physical_dim
    x,_ = roots_hermite(n)
    t = get_laplacian(x)
    conversion_dict["I"+str(n)] = np.eye(n)
    conversion_dict["t"+str(n)] = t
    q = np.diag(x)
    conversion_dict["q"+str(n)] = q
    q2 = np.diag(x**2)
    conversion_dict["q"+str(n)+"^2"] = q2
    conversion_dict["I"+str(physical_dim)] = np.eye(physical_dim)

    for i in range(num_harmonic_oscillator):
        name_list.append({"site"+str(i): f"t{n}", "coeff": Aij[i,i]})
        
        # conversion_dict["t_"+str(i)] = conversion_dict["t"+str(n)]
        
    # harmonic part
    for i in range(num_harmonic_oscillator):
        name_list.append({"site"+str(i): "q"+str(n)+"^2", "coeff": Aij[i,i]*0.5})
        # conversion_dict["q"+str(i)+str(i)] = conversion_dict["q"+str(n)+"^2"]
        for j in range(i+1,num_harmonic_oscillator):
            name_list.append({"site"+str(i): "q"+str(n), "site"+str(j): "q"+str(n), "coeff": Aij[i,j]})
            # conversion_dict["q"+str(i)+str(j)] = conversion_dict["q"+str(n)]
   
    return name_list, conversion_dict

def get_simple_harmonic_oscillator_orbitals(num_orb: int, omega: float)->Tuple[np.ndarray, np.ndarray]:
    """
    Get the orbitals of a simple harmonic oscillator.
    Input:
        num_orb: int
            The number of basis functions.
        omega: float
            The frequency of the harmonic oscillator.
    Output:
        ev: np.ndarray
            The eigenvectors of the harmonic oscillator.
    """
    x,_ = roots_hermite(num_orb)
    t = get_laplacian(x)
    v = np.diag(x**2)*0.5 
    
    h = omega*(t + v)
    _, ev = np.linalg.eigh(h)
    return ev

def get_harmonic_oscillator_orbitals(num_orb: list[int], omega: list[float], orb_state: np.ndarray)->list[np.ndarray]:
    """
    Get the orbitals of multiple harmonic oscillators.
    Input:
        num_orb: list[int]
            The number of basis functions.
        omega: list[float]
            The frequency of the harmonic oscillator.
        orb_state: np.ndarray
            The state of the harmonic oscillator.
    Output:
        ev: list[np.ndarray]
            The eigenvectors of the harmonic oscillator corresponding to the given state.
    """
    if len(num_orb) != len(omega):
        raise ValueError("num_orb and omega must have the same length")
    if len(num_orb) != orb_state.shape[1]:
        raise ValueError("num_orb and orb_state must have the same length")
    evs = []
    for i in range(len(num_orb)):
        ev_i = get_simple_harmonic_oscillator_orbitals(num_orb[i], omega[i])
        ind = list(orb_state[:,i])#;print(ind, ev_i)
        evs.append(ev_i[:,ind])
    return evs
    
def get_orbitals_indices_first(nu: list[float], max_orb: int=100, num_orb: int=20)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds first N elemens of a sum:
        const + nu[0]*n_0 + nu[1]*n_1 + ... + nu[d]*n_d, where n_i = 0, 1, 2, ...
    Input:
        nu: list[float]
            The coefficients of the sum.
        max_orb: int
            The maximum number of orbitals.
        num_orb: int
            The number of orbitals to find.
    Output:
        orb_index: np.ndarray
            The indices of the orbitals.
        orb_state: np.ndarray
            The states of the orbitals.
        orb_Es: np.ndarray
            The energies of the orbitals.
    """
    nu = np.array(nu, dtype=np.float64)
    n = nu.size
    const = np.sum(nu)*0.5
    V = set([tuple(np.zeros(n))])
    ans = np.zeros((max_orb, n))
    values = np.zeros(max_orb)
    
    for iter_idx in range(1, max_orb):
        best_val = float("+inf")
        best_seq = None
        
        # Consider all possible states from previous iterations
        for i in range(iter_idx):
            for j in range(n):
                curr_seq = ans[i, :].copy()
                curr_seq[j] += 1
                if tuple(curr_seq) not in V:
                    curr_val = values[i] + nu[j]
                    if curr_val < best_val:
                        best_seq = curr_seq
                        best_val = curr_val
        
        if best_seq is None:
            break  # No more valid states to add
            
        ans[iter_idx, :] = best_seq
        values[iter_idx] = best_val
        V.add(tuple(best_seq))
        
    orb_Es = np.array(values + const, dtype=np.float64)
    orb_state = np.array(ans, dtype=np.int64)  
    
    sort_idx = orb_Es.argsort()
    orb_state, orb_Es = orb_state[sort_idx], orb_Es[sort_idx]
    orb_state, orb_Es = orb_state[:num_orb], orb_Es[:num_orb]
    orb_index = np.arange(num_orb)
    return orb_index, orb_state, orb_Es

def orbitals_array2str(arr, type="code"):
    ## type: code, latex
    if type == "code":
        result = ""
        for i, value in enumerate(arr):
            if value != 0:
                if result:
                    result += " + "
                if value == 1:
                    result += f"v{i+1}"
                if value != 1:
                    result += f"{value}v{i+1}"
        if result == "":
            result = "ZPE"
    
    elif type == "latex":
        result = ""
        for i, value in enumerate(arr):
            if value != 0:
                if result:
                    result += " + "
                if value == 1:
                    result += f"\\nu_{{{i+1}}}"
                if value != 1:
                    result += f"{value}\\nu_{{{i+1}}}"
        result = "$" + result + "$"
        if result == "$$":
            result = "ZPE"  
        
    return result

def get_energy_clusters(energy: Union[str, List[float]], energy_difference: float, max_cluster_size: int = None):
    """
    Get the energy clusters from a list of energies.
    If max_cluster_size is specified, split clusters larger than this size into two parts.
    The split clusters are inserted back into their original position.
    
    Args:
        energy: Either a file path or a list of energies
        energy_difference: Maximum energy difference for states to be in the same cluster
        max_cluster_size: Maximum allowed size for a cluster. If None, no splitting is performed.
    
    Returns:
        List of energy clusters, where each cluster is a list of energies
    """
    if isinstance(energy, str):
        with open(energy, "r") as file:
            energies = file.readlines()
        energies = [float(energy.strip()) for energy in energies]
    else:
        energies = energy

    # Sort the input values with original indices for stability
    sorted_vals = sorted(enumerate(energy), key=lambda x: x[1])

    clusters = []
    current_cluster = []
    current_base = None

    for idx, val in sorted_vals:
        if not current_cluster:
            # Start new cluster
            current_cluster = [(idx, val)]
            current_base = val
        else:
            last_val = current_cluster[-1][1]
            same_as_existing = any(abs(val - v) < 1e-12 for _, v in current_cluster)

            if (len(current_cluster) >= max_cluster_size and not same_as_existing) or (abs(val - current_base) > energy_difference and not same_as_existing):
                # Finalize current cluster
                clusters.append([i for i, _ in current_cluster])
                # Start new one
                current_cluster = [(idx, val)]
                current_base = val
            else:
                current_cluster.append((idx, val))

    if current_cluster:
        clusters.append([i for i, _ in current_cluster])
    return clusters

def single_voxel_block_diag(C, d=4):
    """
    C : (p, m)  - columns are the channel vectors of each tensor
    d : number of spatial axes (default = 4 → v1…v4)
    """
    p, m = C.shape
    out = np.zeros((m,)*d + (p,), dtype=C.dtype)

    for k in range(m):
        out[(k,)*d + (slice(None),)] = C[:, k]

    return out

def get_ttno(N: List[int], state: TreeTensorNetworkState, get_potential_energy: callable, hamiltonian: bool = False) -> TreeTensorNetworkOperator:
    """
    Get the TreeTensorNetworkOperator for the given potential energy function.
    """
    
    name_list, conversion_dict = get_name_list(N, get_potential_energy)
    conversion_dict["I1"] = np.eye(1)
    new_name_list = [{k: v for k, v in d.items() if k != 'coeff'} for d in name_list]
    coeffs = [(Fraction(str(np.around(d['coeff'], 8))), "1") for d in name_list]
    terms = [TensorProduct(new_name_list[i]) for i in range(len(new_name_list))]
    ham_terms = [(x,y,z) for (x,y),z in zip(coeffs, terms)]
    ham = Hamiltonian(ham_terms, conversion_dictionary=conversion_dict)
    ham_pad = ham.pad_with_identities(state, symbolic=True)
    ttno = TreeTensorNetworkOperator.from_hamiltonian(ham_pad, state, dtype=np.float64, method=TTNOFinder.SGE)
    if hamiltonian:
        return ttno, ham_pad
    else:
        return ttno
    
def get_ttno_harmonic_oscillator(N: List[int], state: TreeTensorNetworkState, omega: np.ndarray, hamiltonian: bool = False) -> TreeTensorNetworkOperator:
    """
    Get the TreeTensorNetworkOperator for the harmonic oscillator.
    """
    
    name_list, conversion_dict = get_name_list_harmonic(len(N), N[0])
    conversion_dict["I1"] = np.eye(1)
    new_name_list = [{k: v for k, v in d.items() if k != 'coeff'} for d in name_list]
    coeffs = [(Fraction(str(np.around(d['coeff'], 8))), "1") for d in name_list]
    terms = [TensorProduct(new_name_list[i]) for i in range(len(new_name_list))]
    ham_terms = [(x,y,z) for (x,y),z in zip(coeffs, terms)]
   
    ham = Hamiltonian(ham_terms, conversion_dictionary=conversion_dict)
    ham_pad = ham.pad_with_identities(state, symbolic=True)
    ttno = TreeTensorNetworkOperator.from_hamiltonian(ham_pad, state, dtype=np.float64, method=TTNOFinder.SGE)
    if hamiltonian:
        return ttno, ham_pad
    else:
        return ttno
    