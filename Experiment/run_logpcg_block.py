import numpy as np
import cProfile
import os
from fractions import Fraction
import matplotlib.pyplot as plt
import pandas as pd
import time

from pytreenet.operators import TensorProduct
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.util import SVDParameters
from pytreenet.dmrg.lobpcg import lobpcg_block, precond_lobpcg


from potentials import get_potential_energy_CH3CN, get_potential_energy_CH3CN_harmonic
from utils import get_orbitals_indices_first, get_energy_clusters, get_ttno
from utils_ch3cn import random_mps_harmonic_oscillator_0, random_threetree_harmonic_oscillator_0, \
    random_leafonly_harmonic_oscillator_0, random_fork_tree, random_t3ns_harmonic_oscillator,\
    random_twin_tree_harmonic_oscillator

np.random.seed=42

def run_calculation(state_type='mps', max_bond_dim=10, file_path="./Results/ch3cn"):
    start_time = time.time()
    N = [9,7,9,9,9,9,7,7,9,9,27,27] # physical dimensions of the model
    node_order = [0,1,2,3,4,5,6,7,8,9,10,11] # order of the modes
    # Create results directory for this state type
    results_dir = f"{file_path}/{state_type}/max_bond_dim_{max_bond_dim}"
    states_dir = f"{results_dir}/states"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    omega,_,_ = get_potential_energy_CH3CN_harmonic() # frequency of the harmonic oscillator
    
    _, orb_state, orb_Es = get_orbitals_indices_first(omega,num_orb=84) # get the orbitals and their harmonic energies
    clusters = get_energy_clusters(orb_Es, 0.01, 6) # cluster the energies into groups of states with energy difference less than 0.01 and maximum cluster size of 6
    print("Clusters", clusters)
    
    # generate the states
    states = []
    
    for i in range(len(orb_Es)):
        if state_type == 'mps':
            states.append(random_mps_harmonic_oscillator_0(N, omega, orb_state[i].reshape(1,-1),node_order))
            states[-1].canonical_form(states[-1].root_id)
        elif state_type == 'threetree':  # threetree
            states.append(random_threetree_harmonic_oscillator_0(N, omega, orb_state[i].reshape(1,-1),node_order))
            states[-1].canonical_form(states[-1].root_id)
        elif state_type == 'leafonly':  # leafonly
            states.append(random_leafonly_harmonic_oscillator_0(N, omega, orb_state[i].reshape(1,-1),node_order))
            states[-1].canonical_form(states[-1].root_id)
        elif state_type == 'fork3':  # fork tree
            states.append(random_fork_tree(N, omega, orb_state[i].reshape(1,-1), node_order, nbranch=3))
            states[-1].canonical_form(states[-1].root_id)
        elif state_type == 'fork4':  # fork tree with 4 branches
            states.append(random_fork_tree(N, omega, orb_state[i].reshape(1,-1), node_order, nbranch=4))
            states[-1].canonical_form(states[-1].root_id)
        elif state_type == 'fork6':  # fork tree with 6 branches
            states.append(random_fork_tree(N, omega, orb_state[i].reshape(1,-1), node_order, nbranch=6))
            states[-1].canonical_form(states[-1].root_id)
        elif state_type == 't3ns':  # t3ns
            node_order = [0,1,6,7,3,8,9,2,4,5,10,11] # t3ns new node order
            states.append(random_t3ns_harmonic_oscillator(N, omega, orb_state[i].reshape(1,-1), node_order))
            states[-1].canonical_form(states[-1].root_id)
        elif state_type == 'twin':
            states.append(random_twin_tree_harmonic_oscillator(N, omega, orb_state[i].reshape(1,-1), node_order))
            states[-1].canonical_form(states[-1].root_id)
        else:
            raise ValueError(f"State type {state_type} not supported")
    ttno, ham_pad = get_ttno(N, states[0], get_potential_energy_CH3CN, True) # get the TTNO and the Hamiltonian

    shift_term = (Fraction(-9), "1", TensorProduct({'site0':'I9','site1':'I7','site2':'I9','site3':'I9','site4':'I9','site5':'I9','site6':'I7','site7':'I7','site8':'I9','site9':'I9','site10':'I27','site11':'I27'}))
    # shift = 9
    # ham_pad.conversion_dictionary[f'{shift}I9'] = np.eye(9)*shift
    # shift_term = TensorProduct({'site0':f'{shift}I9','site1':'I7','site2':'I9','site3':'I9','site4':'I9','site5':'I9','site6':'I7','site7':'I7','site8':'I9','site9':'I9','site10':'I27','site11':'I27'})
    ham_pad.add_term(shift_term)
    ttno_shift = TreeTensorNetworkOperator.from_hamiltonian(ham_pad, states[0], dtype=np.float64)
    precond_func = lambda state, svd_params: precond_lobpcg(ttno_shift, state, svd_params)
          
    references = pd.read_csv("./Experiment/ch3cn_ref.csv")
    ref_energy = references["Energy_Ref"].to_numpy(copy=True)
    ref_energy[1:] = ref_energy[1:] + ref_energy[0]

    # file_path = [f"{states_dir}/lobpcg_state_{i}" for i in range(6)]
    # energies = pd.read_csv(f"{results_dir}/lobpcg_energies.csv")
    file_path = []
    energies = []
    print(f"ttno dims for {state_type}:", ttno.bond_dims().values())
    state_counter = 0 
    for cl in clusters:
        print("cluster", cl, "out of", len(clusters))
        time_start = time.time()
        states_list = [states[c] for c in cl]
        states_opt_list, energies_i = lobpcg_block(ttno, states_list, precond_func, SVDParameters(max_bond_dim=max_bond_dim, renorm=True, rel_tol=1e-8, total_tol=1e-8), 5, file_path)
        time_end = time.time()
        print(f"Time taken for state {cl} lobpcg_block: {time_end - time_start} seconds")

        energies+=energies_i 
        # save the states, energies and plot the energies
        for ix, s in enumerate(states_opt_list):
            s.save(f"{states_dir}/lobpcg_state_{state_counter + ix}") 
            file_path.append(f"{states_dir}/lobpcg_state_{state_counter + ix}")
            plt.plot(abs(np.array(energies_i[ix])*1000-ref_energy[state_counter + ix]), label="lobpcg")
            plt.yscale("log")
            plt.plot(range(len(energies_i[ix])), np.ones(len(energies_i[ix])), label="reference")
            plt.ylabel("Energy difference (cm-1)")
            plt.xlabel("Iteration")
            plt.legend()
            plt.title(f"lobpcg_energies_{state_type}_{state_counter + ix}")
            plt.savefig(f"{results_dir}/lobpcg_energies_{state_counter + ix}.png")
            plt.close()
        
        state_counter += len(cl)
    
        csv_file = f"{results_dir}/lobpcg_energies.csv"
        np.savetxt(csv_file, energies, delimiter=",")

    print(f"Time taken for {state_type} with max bond dim {max_bond_dim}: ", time.time() - start_time)
    return time.time() - start_time

def main():
    # Run for both MPS and threetree with separate profiling
    file_path = "./Results_test_ttno/ch3cn"
    for max_bond_dim in [8,12]:
        for state_type in ['mps', 'threetree', 'leafonly','fork3','fork4', 'fork6','t3ns']:
            try:
                print(f"\nRunning calculation for {state_type} with max bond dim {max_bond_dim}...")
                profiler = cProfile.Profile()
                profiler.enable()
                run_calculation(state_type, max_bond_dim=max_bond_dim, file_path=file_path)
            except Exception as exc:
                print(f"[ERROR] {state_type=} {max_bond_dim=}: {exc}")
            else:
                profiler.disable()
                profiler.dump_stats(
                    f"{file_path}/{state_type}/max_bond_dim_{max_bond_dim}/profile_output.prof"
                )
            finally:
                profiler.disable()
                print(f"Profiling complete for {state_type} with max bond dim {max_bond_dim}. Stats saved to {file_path}/{state_type}/max_bond_dim_{max_bond_dim}/profile_output.prof")

if __name__ == "__main__":
    main()
