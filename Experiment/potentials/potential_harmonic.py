import numpy as np

def get_interaction_matrix(D):
    """
    For N-D bilinearly coupled oscillator, get the interaction matrix.
    Ref: [1] Calculating vibrational spectra of molecules using tensor train decomposition.
                https://doi.org/10.1063/1.4962420
         [2] Using Nested Contractions and a Hierarchical Tensor Format To Compute 
                Vibrational Spectra of Molecules with Seven Atoms
    """
    wsquare_indices = 0.5 * np.arange(1, D+1)
    w_indices = np.sqrt(wsquare_indices)

    a = 0.1
    Aij = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            if i == j:
                Aij[i,j] = w_indices[i]
            else:
                Aij[i,j] = a 
    Aij = np.array(Aij)

    return Aij
def calculate_frequency_exact(D):
    """
    For N-D bilinearly coupled oscillator, calculate the exact eigenfrequencies.
    Here we need to multiply the omegas to the interaction matrix because we use a different convention to build the basis.
    """
    wsquare_indices = 0.5 * np.arange(1, D+1)
    w_indices = np.sqrt(wsquare_indices)

    a = 0.1
    Aij = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            if i == j:
                Aij[i,j] = w_indices[i]**2
            else:
                Aij[i,j] = a * np.sqrt(w_indices[i] * w_indices[j])
                
    nu, _ = np.linalg.eigh(Aij)
    w_indices = np.sqrt(nu)

    return w_indices

def get_potential_energy_harmonic(D=6):
    """
    For N-D bilinearly coupled oscillator, get the eigenfrequencies for the exact solution and the interaction matrix for solving it with TTN methods.
    """
    Aij = get_interaction_matrix(D)
    w_indices = calculate_frequency_exact(D)
    return w_indices, Aij

if __name__ == "__main__":
    w_indices, Aij = get_potential_energy_harmonic(D=6)


