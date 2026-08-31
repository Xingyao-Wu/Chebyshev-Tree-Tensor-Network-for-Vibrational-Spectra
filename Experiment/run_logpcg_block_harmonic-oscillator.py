import numpy as np
import cProfile
import os
import matplotlib.pyplot as plt
import pandas as pd
import time

from fractions import Fraction
from pytreenet.operators import TensorProduct
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.util import SVDParameters
from pytreenet.dmrg.lobpcg import lobpcg_block, precond_lobpcg

from potentials import get_potential_energy_harmonic
from utils import get_orbitals_indices_first, get_energy_clusters, get_ttno_harmonic_oscillator
from utils_harmonic import random_mps_harmonic_oscillator_0,random_threetree_harmonic_oscillator_0


np.random.seed=42

def run_calculation(state_type='mps', max_bond_dim=10, file_path="./Results/harmonic"):
    start_time = time.time()
    N = [8,8]*32
    node_order = np.arange(len(N)).tolist()

    # Create results directory for this state type
    results_dir = f"{file_path}/{state_type}/max_bond_dim_{max_bond_dim}"
    states_dir = f"{results_dir}/states"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)
    
    w_ref,Aij = get_potential_energy_harmonic(D=len(N))
    omega = np.diag(Aij)
    
    orb_index, orb_state, orb_Es = get_orbitals_indices_first(omega,num_orb=30)
    _,_,orb_Es_ref = get_orbitals_indices_first(w_ref,num_orb=30)
    clusters = get_energy_clusters(orb_Es, 0.1, 10)
    # print("Orbital energies (harmonic oscillator)", orb_Es)
    # print("Orbital energies (reference)", orb_Es_ref)    
    # print("Clusters", clusters)
    np.savetxt(f"{file_path}/orb_Es_harmonic_oscillator.csv", orb_Es, delimiter=",")
    np.savetxt(f"{file_path}/orb_Es_ref.csv", orb_Es_ref, delimiter=",")
    
    states = []
    
    for i in range(len(orb_Es)):
        if state_type == 'mps':
            states.append(random_mps_harmonic_oscillator_0(N, omega, orb_state[i].reshape(1,-1),node_order))
            states[-1].canonical_form(states[-1].root_id)
        elif state_type == 'threetree':  # threetree
            states.append(random_threetree_harmonic_oscillator_0(N, omega, orb_state[i].reshape(1,-1),node_order))
            states[-1].canonical_form(states[-1].root_id)
        else:
            raise ValueError(f"State type {state_type} not supported")
    # The TTNO can be built using the get_ttno_harmonic_oscillator function but it might take a while    
    # ttno, ham_pad = get_ttno_harmonic_oscillator(N, states[0], omega, True)
    # ttno.save(f"{results_dir}/ttno")
    # print(f"ttno dims for {state_type}:", ttno.bond_dims().values())
    # shift_term = (Fraction(-121), "1", TensorProduct({'site0':f'I{N[0]}'}))
    # ham_pad.add_term(shift_term)
    # ttno_shift = TreeTensorNetworkOperator.from_hamiltonian(ham_pad, states[0], dtype=np.float64)
    # print(f"ttno dims for {state_type} shifted:", ttno_shift.bond_dims().values())
    # ttno_shift.save(f"{results_dir}/ttno_shift")

    #TTNO can also be saved and loaded from the file
    ttno = TreeTensorNetworkOperator.load(f"{results_dir}/ttno")
    ttno_shift = TreeTensorNetworkOperator.load(f"{results_dir}/ttno_shift")
    print(f"ttno dims for {state_type}:", ttno.bond_dims().values())
    print(f"ttno dims for {state_type} shifted:", ttno_shift.bond_dims().values())

    precond_func = lambda state, svd_params: precond_lobpcg(ttno_shift, state, svd_params)
    
    for i in range(len(states)):
        print("hamonic energy",states[i].operator_expectation_value(ttno))
    file_path = []
    energies = []
    
    state_counter = 0
    for cl in clusters:
        # print("cluster", i, "out of", len(clusters))
        time_start = time.time()
        states_list = [states[c] for c in cl]
        states_opt_list, energies_i = lobpcg_block(ttno, states_list, precond_func, SVDParameters(max_bond_dim=max_bond_dim, renorm=True, rel_tol=1e-8, total_tol=1e-8), 6, file_path)
        time_end = time.time()
        print(f"Time taken for state {cl} lobpcg_block: {time_end - time_start} seconds")

        energies+=energies_i
        for ix, s in enumerate(states_opt_list):
            s.save(f"{states_dir}/lobpcg_state_{state_counter + ix}") 
            file_path.append(f"{states_dir}/lobpcg_state_{state_counter + ix}")
        
            plt.plot(abs(energies_i[ix]-orb_Es_ref[state_counter + ix]), label="lobpcg")
            plt.yscale("log")
            plt.plot(range(len(energies_i[ix])), np.ones(len(energies_i[ix])), label="reference")
            plt.ylabel("Energy difference (cm-1)")
            plt.xlabel("Iteration")
            plt.legend()
            plt.title(f"lobpcg_energies_{state_type}_{state_counter + ix}")
            plt.savefig(f"{results_dir}/lobpcg_energies_{state_counter + ix}.png")
            plt.close()
        print("energies", energies_i[-1], orb_Es_ref[state_counter + ix])
        state_counter += len(cl)
        
        csv_file = f"{results_dir}/lobpcg_energies.csv"
        np.savetxt(csv_file, energies, delimiter=",")

    print(f"Time taken for {state_type} with max bond dim {max_bond_dim}: ", time.time() - start_time)
    return time.time() - start_time

def main():
    # Run for both MPS and threetree with separate profiling
    for max_bond_dim in [5]:
        for state_type in ['mps']:
            print(f"\nRunning calculation for {state_type} with max bond dim {max_bond_dim}...")
            profiler = cProfile.Profile()
            profiler.enable()
            run_calculation(state_type, max_bond_dim=max_bond_dim,file_path="./Results_final/test_harmonic")
            profiler.disable()
            profiler.dump_stats(f"./Results_final/test_harmonic/{state_type}/max_bond_dim_{max_bond_dim}/profile_output.prof")

if __name__ == "__main__":
    main()
