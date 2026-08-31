import numpy as np  
import scipy
from pathlib import Path
from functools import reduce
from scipy.special import roots_hermite
from .orca_parser import get_anharmonic_constants, get_dipole_moment_1

current_path = Path(__file__).parent

def generate_grid(h, N ):
    x = np.linspace(-h, h, 2*N+1)
    return np.diag(x)

def generate_kinetic(N=3):
    T = (-2)*np.eye(2*N+1) + np.eye(2*N+1, k=1) + np.eye(2*N+1, k=-1)
    return T

def get_laplacian(Np: int) -> np.ndarray:
    """
    Get the Laplacian matrix for a given set of points.
    Input:
        xs: np.ndarray
            The points to compute the Laplacian matrix.
    Output:
        lp: np.ndarray
            The Laplacian matrix.
    """
    xs,_ = roots_hermite(Np)
    N = len(xs)
    lp = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                lp[i,j] = (-1)**(i - j)*(2*(xs[i] - xs[j])**(-2) - 0.5)
            else:
                lp[i,j] = 1.0/6 * (4*N - 1 - 2*xs[i]**2)

    return lp

def build_terms(text: str, order: int, n_modes: int, q: np.ndarray, I: np.ndarray):
    """
    For each line, returns ( [op1, op2, ..., op_n_modes], coefficient )
    where op_k = q**count_k if mode k appears count_k times in the row, else I.
    """
    terms = []
    for raw in text.strip().splitlines():
        parts = raw.split()
        if not parts:
            continue
        idxs = list(map(int, parts[:order]))        # e.g. [1,1,1] for cubic
        coeff = float(parts[order]) / 1000.0              # the trailing number
        ops = []
        for mode in range(1, n_modes + 1):
            power = idxs.count(mode)
            if power == 0:
                ops.append(I)
            elif power == 1:
                ops.append(q)
            else:
                ops.append(np.linalg.matrix_power(q, power))
        terms.append((ops, coeff/scipy.special.factorial(order)))
    return terms


def construct_hamiltonian_harmonic(N, filename: str):
    w, cubic, quartic = get_anharmonic_constants(
        filename,
        map_modes=[0, 1, 2],
        signs=[1, 1, 1],
    )

    t = get_laplacian(2 * N + 1)
    q = np.diag(roots_hermite(2 * N + 1)[0])
    I = np.eye(2 * N + 1)

    w = np.array(w, dtype=np.float64) / 1000.0

    H = []

    for i in range(3):
        t_identity = [I.copy() for _ in range(3)]
        t_identity[i] = t

        q_identity = [I.copy() for _ in range(3)]
        q_identity[i] = q @ q

        H.append((t_identity, 0.5 * w[i]))
        H.append((q_identity, 0.5 * w[i]))

    return H, w


def construct_hamiltonian(N, filename: str):
    w, cubic, quartic = get_anharmonic_constants(filename, map_modes=[0,1,2], signs=[1,1,1])
    t = get_laplacian(2*N+1)
    q = np.diag(roots_hermite(2*N+1)[0])
    w = np.array(w, dtype=np.float64) / 1000.0
    
    H = []
  
    for i in range(3):
        t_identity = [np.eye(2*N+1) for _ in range(3)]
        t_identity[i] = t
        q_identity = [np.eye(2*N+1) for _ in range(3)]  #
        q_identity[i] = q@q
        H.append((t_identity,  0.5*w[i]))
        H.append((q_identity, w[i] *0.5))
    cubic_terms = build_terms(cubic, 3, 3, q, np.eye(2*N+1))
    quartic_terms = build_terms(quartic, 4, 3, q, np.eye(2*N+1))
    H.extend(cubic_terms)
    H.extend(quartic_terms)
    return H, w, cubic, quartic

def construct_dipole_moment1(N, filename: str):
    """
    Build μ_x, μ_y, μ_z as lists of Kronecker terms for `to_matrix`.
    Each ORCA row is (Tx, Ty, Tz) for one mode; linear dipole is sum_m d_{α,m} q_m.
    """
    dipole_moment = np.asarray(get_dipole_moment_1(filename), dtype=float)
    eye = np.eye(2 * N + 1)
    q = np.diag(roots_hermite(2 * N + 1)[0])
    n_modes = dipole_moment.shape[0]
    dx, dy, dz = [], [], []
    for m in range(n_modes):
        ops = [eye.copy() for _ in range(n_modes)]
        ops[m] = q
        tx, ty, tz = dipole_moment[m]
        dx.append((ops, float(tx)))
        dy.append((ops, float(ty)))
        dz.append((ops, float(tz)))
    return dx, dy, dz
    

def kron_all(arrays):
    return reduce(np.kron, arrays)
    
def to_matrix(H):
    matrices = []
    for term in H:
        ops, coeff = term
        matrix = kron_all(ops) * coeff
        # print(matrix,"coeff", coeff)
        matrices.append(matrix)
    return sum(matrices)

if __name__ == "__main__":
    f = current_path / "../orca" / "h2o" / "vib"
    h = construct_hamiltonian(4,6, f)
    h = to_matrix(h)
    e,v = np.linalg.eigh(h)
    v5 = v[:,:5]
    # print("e", e[:5]-e[0])
    dx, dy, dz = construct_dipole_moment1(6, f)
    d_xx = v5.T @ to_matrix(dx) @ v5 
    print("d_xx", d_xx)
    d_yy = v5.T @ to_matrix(dy) @ v5 
    print("d_yy", d_yy)
    d_zz = v5.T @ to_matrix(dz) @ v5 
    print("d_zz", d_zz)