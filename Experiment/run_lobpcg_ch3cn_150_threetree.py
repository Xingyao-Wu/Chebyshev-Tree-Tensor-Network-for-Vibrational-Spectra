import os
import time
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytreenet.operators import TensorProduct
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.util import SVDParameters
from pytreenet.dmrg.lobpcg import lobpcg_block, precond_lobpcg

from potentials import get_potential_energy_CH3CN, get_potential_energy_CH3CN_harmonic
from utils import get_orbitals_indices_first, get_energy_clusters, get_ttno
from utils_ch3cn import random_threetree_harmonic_oscillator_0


np.random.seed(42)


def load_reference_energies(reference_path: str) -> np.ndarray:
    references = pd.read_csv(reference_path)
    ref_energy = references["Energy_Ref"].to_numpy(copy=True)
    if ref_energy.size > 1:
        ref_energy[1:] = ref_energy[1:] + ref_energy[0]
    return ref_energy


def read_progress(results_dir: str) -> tuple[int, int]:
    progress_path = f"{results_dir}/progress.txt"
    if not os.path.exists(progress_path):
        return 0, 0

    progress = {}
    with open(progress_path, "r", encoding="utf-8") as progress_file:
        for line in progress_file:
            if "," not in line:
                continue
            key, value = line.strip().split(",", 1)
            progress[key] = value
    return int(progress.get("completed_states", 0)), int(progress.get("completed_clusters", 0))


def load_existing_energies(csv_file: str, completed_states: int) -> list[np.ndarray]:
    if completed_states == 0 or not os.path.exists(csv_file):
        return []
    loaded = np.loadtxt(csv_file, delimiter=",")
    loaded = np.atleast_2d(loaded)
    if loaded.shape[0] < completed_states:
        raise RuntimeError(
            f"Energy file has {loaded.shape[0]} rows, but progress says {completed_states} states."
        )
    return [row for row in loaded[:completed_states]]


def write_progress(
    results_dir: str,
    completed_states: int,
    total_states: int,
    completed_clusters: int,
    total_clusters: int,
    elapsed_seconds: float,
    running_cluster: int | None = None,
) -> None:
    with open(f"{results_dir}/progress.txt", "w", encoding="utf-8") as progress_file:
        progress_file.write(f"completed_states,{completed_states}\n")
        progress_file.write(f"total_states,{total_states}\n")
        progress_file.write(f"completed_clusters,{completed_clusters}\n")
        progress_file.write(f"total_clusters,{total_clusters}\n")
        if running_cluster is not None:
            progress_file.write(f"running_cluster,{running_cluster}\n")
        progress_file.write(f"elapsed_seconds,{elapsed_seconds}\n")


def save_energy_plot(
    energies_i,
    state_index: int,
    results_dir: str,
    ref_energy: np.ndarray | None,
) -> None:
    energy_history = np.asarray(energies_i, dtype=np.float64)
    if ref_energy is not None and state_index < ref_energy.size:
        plt.plot(abs(energy_history * 1000 - ref_energy[state_index]), label="lobpcg")
        plt.yscale("log")
        plt.plot(range(len(energy_history)), np.ones(len(energy_history)), label="reference")
        plt.ylabel("Energy difference (cm-1)")
    else:
        plt.plot(energy_history, label="lobpcg")
        plt.ylabel("Energy")
    plt.xlabel("Iteration")
    plt.legend()
    plt.title(f"lobpcg_energies_threetree_{state_index}")
    plt.savefig(f"{results_dir}/lobpcg_energies_{state_index}.png")
    plt.close()


def run_calculation(
    num_states: int = 150,
    max_bond_dim: int = 12,
    file_path: str = "/Users/link/Desktop/A_Python_code/Chebyshev_upload/Chebyshev/master_thesis_reference",
) -> float:
    start_time = time.time()
    state_type = "threetree"
    N = [9, 7, 9, 9, 9, 9, 7, 7, 9, 9, 27, 27]
    node_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    results_dir = f"{file_path}/{state_type}/max_bond_dim_{max_bond_dim}"
    states_dir = f"{results_dir}/states"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    omega, _, _ = get_potential_energy_CH3CN_harmonic()
    _, orb_state, orb_Es = get_orbitals_indices_first(
        omega,
        max_orb=num_states,
        num_orb=num_states,
    )
    clusters = get_energy_clusters(orb_Es, 0.01, 6)
    print("Number of target states:", len(orb_Es), flush=True)
    print("Number of clusters:", len(clusters), flush=True)
    print("Clusters:", clusters, flush=True)

    np.savetxt(f"{results_dir}/harmonic_initial_energies.csv", orb_Es, delimiter=",")
    np.savetxt(f"{results_dir}/harmonic_initial_states.csv", orb_state, delimiter=",", fmt="%d")

    states = []
    for i in range(len(orb_Es)):
        state = random_threetree_harmonic_oscillator_0(
            N,
            omega,
            orb_state[i].reshape(1, -1),
            node_order,
        )
        state.canonical_form(state.root_id)
        states.append(state)

    ttno, ham_pad = get_ttno(N, states[0], get_potential_energy_CH3CN, True)
    shift_term = (
        Fraction(-9),
        "1",
        TensorProduct(
            {
                "site0": "I9",
                "site1": "I7",
                "site2": "I9",
                "site3": "I9",
                "site4": "I9",
                "site5": "I9",
                "site6": "I7",
                "site7": "I7",
                "site8": "I9",
                "site9": "I9",
                "site10": "I27",
                "site11": "I27",
            }
        ),
    )
    ham_pad.add_term(shift_term)
    ttno_shift = TreeTensorNetworkOperator.from_hamiltonian(
        ham_pad,
        states[0],
        dtype=np.float64,
    )
    precond_func = lambda state, svd_params: precond_lobpcg(ttno_shift, state, svd_params)

    reference_path = "./Experiment/ch3cn_ref.csv"
    ref_energy = load_reference_energies(reference_path) if os.path.exists(reference_path) else None

    completed_states, completed_clusters = read_progress(results_dir)
    if completed_states or completed_clusters:
        print(
            f"Resuming from {completed_states} completed states and "
            f"{completed_clusters} completed clusters.",
            flush=True,
        )

    previous_state_paths = [
        f"{states_dir}/lobpcg_state_{state_index}"
        for state_index in range(completed_states)
    ]
    for state_path in previous_state_paths:
        if not os.path.exists(f"{state_path}.json") or not os.path.exists(f"{state_path}.npz"):
            raise FileNotFoundError(f"Missing saved state for resume: {state_path}")

    csv_file = f"{results_dir}/lobpcg_energies.csv"
    energies = load_existing_energies(csv_file, completed_states)
    print(f"ttno dims for {state_type}:", ttno.bond_dims().values(), flush=True)
    state_counter = completed_states
    for cluster_index, cl in enumerate(clusters[completed_clusters:], start=completed_clusters):
        print("cluster", cluster_index + 1, "of", len(clusters), cl, flush=True)
        write_progress(
            results_dir,
            state_counter,
            num_states,
            cluster_index,
            len(clusters),
            time.time() - start_time,
            running_cluster=cluster_index + 1,
        )
        time_start = time.time()
        states_list = [states[c] for c in cl]
        states_opt_list, energies_i = lobpcg_block(
            ttno,
            states_list,
            precond_func,
            SVDParameters(
                max_bond_dim=max_bond_dim,
                renorm=True,
                rel_tol=1e-8,
                total_tol=1e-8,
            ),
            5,
            previous_state_paths,
        )
        time_end = time.time()
        print(f"Time taken for cluster {cl} lobpcg_block: {time_end - time_start} seconds", flush=True)

        energies += energies_i
        for ix, state in enumerate(states_opt_list):
            state_index = state_counter + ix
            state_path = f"{states_dir}/lobpcg_state_{state_index}"
            state.save(state_path)
            previous_state_paths.append(state_path)
            save_energy_plot(energies_i[ix], state_index, results_dir, ref_energy)

        state_counter += len(cl)
        np.savetxt(csv_file, energies, delimiter=",")
        write_progress(
            results_dir,
            state_counter,
            num_states,
            cluster_index + 1,
            len(clusters),
            time.time() - start_time,
        )

    elapsed = time.time() - start_time
    print(f"Time taken for threetree with max bond dim {max_bond_dim}: {elapsed}", flush=True)
    return elapsed


def main():
    output_dir = "/Users/link/Desktop/A_Python_code/Chebyshev_upload/Chebyshev/master_thesis_reference"
    run_calculation(num_states=150, max_bond_dim=12, file_path=output_dir)


if __name__ == "__main__":
    main()
