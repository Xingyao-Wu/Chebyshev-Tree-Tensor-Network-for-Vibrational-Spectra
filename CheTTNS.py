from __future__ import annotations

import copy
import time
import traceback

import numpy as np

from pytreenet.contractions.state_operator_contraction import get_matrix_element
from pytreenet.core.truncation import TruncationMethod, truncate_ttns
from pytreenet.dmrg import DMRGAlgorithm, VariationalFitting
from pytreenet.operators import Hamiltonian
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.ttns.ttns_ttno.application import ApplicationMethod, apply_ttno_to_ttns
from pytreenet.ttns.ttns_ttno.src import build_full_subtree_cache, find_new_tensors
from pytreenet.util import gpu_backend
from pytreenet.util import SVDParameters
from pytreenet.util.misc_functions import add, linear_combination
from pytreenet.util.std_utils import identity_mapping

from ttno_shift_utils import scale_ttno_with_energy_window_shifted


__all__ = [
    "APPLICATION_METHOD_LABELS",
    "TTNOApplier",
    "active_method_items",
    "apply_ttno_by_method",
    "cheb_ttns_CBC",
    "cheb_ttns_Density",
    "cheb_ttns_Direct_Truncate",
    "cheb_ttns_paper",
    "cheb_ttns_SRC",
    "cheb_ttns_variational_paper",
    "cheb_vector",
    "cheb_vector_plus_heff_runtime_s",
    "cheb_vector_runtime_s",
    "chebyshev_moments_ttns",
    "chebyshev_moments_vector",
    "chebyshev_kernel",
    "dirichlet_kernel",
    "dmrg_min_max_limited_ch3cn",
    "compress_ttno_via_ttns_svd",
    "extend_cheb_ttns_Direct_Truncate",
    "fejer_kernel",
    "format_table",
    "gpu_backend_name",
    "heff_runtime_s",
    "heff_ttns_no_shift_compressed_ttno",
    "heff_ttns_apply_then_overlap",
    "heff_ttns_apply_then_stochastic_overlap",
    "heff_ttns_hybrid_stochastic",
    "heff_ttns_chebyshev_overlap",
    "heff_ttns_no_shift",
    "jackson_kernel",
    "lorentz_kernel",
    "merge_cheb_step_timings",
    "phase_runtime_s",
    "psi_eff_ttns",
    "record_method_error",
    "reconstruct_delta_spectrum",
    "reconstruct_delta_spectrum_grid",
    "rescale_ttno_limited_ch3cn",
    "scale_ttno_with_energy_window",
    "src_ttns_ttno_application_real",
    "using_gpu_backend",
]


APPLICATION_METHOD_LABELS = {
    ApplicationMethod.DIRECT: "Direct",
    ApplicationMethod.DIRECT_TRUNCATE: "Direct+Truncate",
    ApplicationMethod.DENSITY_MATRIX: "Density Matrix",
    ApplicationMethod.HALF_DENSITY_MATRIX: "Half Density Matrix",
    ApplicationMethod.SRC: "SRC",
}


def _network_is_real(network) -> bool:
    """Return whether every tensor uses a real NumPy dtype."""
    return all(np.isrealobj(tensor) for tensor in network.tensors.values())


def _infer_real_src_dtype(ttns, ttno) -> np.dtype:
    dtypes = [tensor.dtype for tensor in ttns.tensors.values()]
    dtypes.extend(tensor.dtype for tensor in ttno.tensors.values())
    dtype = np.result_type(*dtypes)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(np.float64)
    return np.dtype(dtype)


def _real_copy_tensor(degree: int, dimension: int, dtype) -> np.ndarray:
    """Create the real-valued copy tensor used by the SRC random TTNS."""
    tensor = np.zeros((dimension,) * degree, dtype=dtype)
    diagonal = np.arange(dimension)
    tensor[(diagonal,) * degree] = 1
    return tensor


def _generate_real_random_matrices(
    ttns: TreeTensorNetworkState,
    desired_dimension: int,
    dtype,
    seed: int | None = None,
) -> TreeTensorNetworkState:
    """Generate the real random sketch TTNS for real-valued SRC."""
    tensors = {}
    node_seed = seed
    for node_id, node in ttns.nodes.items():
        desired_shape = (desired_dimension, *node.open_dimensions())
        rng = np.random.default_rng(node_seed)
        random_tensor = rng.normal(size=desired_shape).astype(dtype, copy=False)
        copy_tensor = _real_copy_tensor(node.nlegs(), desired_dimension, dtype)
        tensors[node_id] = np.tensordot(copy_tensor, random_tensor, axes=(0, 0))
        node_seed = None if node_seed is None else node_seed + 1
    return TreeTensorNetworkState.from_tensors(ttns, tensors)


def src_ttns_ttno_application_real(
    ttns: TreeTensorNetworkState,
    ttno: TreeTensorNetworkOperator,
    desired_dimension: int,
    id_trafo=None,
    seed: int | None = None,
    dtype=None,
) -> TreeTensorNetworkState:
    """Apply a real TTNO to a real TTNS using real randomized SRC sketches.

    This follows PyTreeNet's SRC contraction and local-QR algorithm, but both
    the random sketch and its copy tensors are real. Complex inputs are rejected
    rather than silently discarding their imaginary parts.
    """
    if not _network_is_real(ttns) or not _network_is_real(ttno):
        raise ValueError("Real SRC requires real-valued TTNS and TTNO tensors.")
    if desired_dimension < 1:
        raise ValueError("desired_dimension must be at least 1.")
    if id_trafo is None:
        id_trafo = identity_mapping
    if dtype is None:
        dtype = _infer_real_src_dtype(ttns, ttno)
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"Real SRC requires a floating dtype, got {dtype}.")

    random_ttns = _generate_real_random_matrices(
        ttns,
        int(desired_dimension),
        dtype=dtype,
        seed=seed,
    )
    subtree_cache = build_full_subtree_cache(ttns, ttno, random_ttns, id_trafo)
    new_tensors = find_new_tensors(ttns, ttno, subtree_cache, id_trafo)
    result = TreeTensorNetworkState.from_tensors(ttns, new_tensors)
    if not _network_is_real(result):
        raise RuntimeError("Real SRC unexpectedly produced complex tensors.")
    return result


def format_table(rows, columns, title=None):
    if title is not None:
        print("\n" + title)
    if not rows:
        print("(none)")
        return
    formatted_rows = []
    widths = {col: len(col) for col in columns}
    for row in rows:
        formatted = {}
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            formatted[col] = str(value)
            widths[col] = max(widths[col], len(formatted[col]))
        formatted_rows.append(formatted)
    fmt = "  ".join("{:<" + str(widths[col]) + "}" for col in columns)
    print(fmt.format(*columns))
    print(fmt.format(*["-" * widths[col] for col in columns]))
    for row in formatted_rows:
        print(fmt.format(*[row[col] for col in columns]))


def apply_ttno_by_method(ttns, ttno, method, svd_params, seed=None):
    if method == ApplicationMethod.SRC:
        if _network_is_real(ttns) and _network_is_real(ttno):
            return src_ttns_ttno_application_real(
                ttns=ttns,
                ttno=ttno,
                desired_dimension=int(svd_params.max_bond_dim),
                seed=seed,
            )
        return apply_ttno_to_ttns(
            ttns=ttns,
            ttno=ttno,
            method=method,
            desired_dimension=int(svd_params.max_bond_dim),
            seed=seed,
        )
    if method == ApplicationMethod.DIRECT_TRUNCATE:
        return apply_ttno_to_ttns(
            ttns=ttns,
            ttno=ttno,
            method=method,
            params=svd_params,
        )
    return apply_ttno_to_ttns(
        ttns=ttns,
        ttno=ttno,
        method=method,
        svd_params=svd_params,
    )


class TTNOApplier:
    def __init__(self, method, svd_params, seed=20260613):
        self.method = method
        self.svd_params = svd_params
        self.seed = seed
        self.calls = 0
        self.time_s = 0.0

    @property
    def label(self):
        return APPLICATION_METHOD_LABELS.get(self.method, str(self.method))

    def apply(self, ttns, ttno):
        self.calls += 1
        call_seed = None if self.seed is None else self.seed + self.calls
        tic = time.perf_counter()
        try:
            return apply_ttno_by_method(
                ttns=ttns,
                ttno=ttno,
                method=self.method,
                svd_params=self.svd_params,
                seed=call_seed,
            )
        finally:
            self.time_s += time.perf_counter() - tic


def record_method_error(data, phase, exc):
    data["status"] = f"failed: {phase}"
    data["error"] = repr(exc)
    data["traceback"] = traceback.format_exc()
    print(f"[FAILED] {data['label']} during {phase}: {exc}")


def active_method_items(contraction_data):
    return [
        (method_key, data)
        for method_key, data in contraction_data.items()
        if data.get("status") == "ok"
    ]


def cheb_vector_runtime_s(data):
    return float(data.get("timings", {}).get("cheb_vector_build_s", 0.0))


def heff_runtime_s(data):
    return float(data.get("timings", {}).get("heff_s", 0.0))


def cheb_vector_plus_heff_runtime_s(data):
    return cheb_vector_runtime_s(data) + heff_runtime_s(data)


def phase_runtime_s(data):
    return cheb_vector_runtime_s(data)


def using_gpu_backend() -> bool:
    return gpu_backend.enabled()


def gpu_backend_name() -> str:
    return gpu_backend.backend_name()


def jackson_kernel(N, xp_module=np):
    n = xp_module.arange(N)
    theta = xp_module.pi / (N + 1)
    return (
        (N - n + 1) * xp_module.cos(n * theta)
        + xp_module.sin(n * theta) / xp_module.tan(theta)
    ) / (N + 1)


def dirichlet_kernel(N, xp_module=np):
    return xp_module.ones(N, dtype=xp_module.float64)


def fejer_kernel(N, xp_module=np):
    n = xp_module.arange(N, dtype=xp_module.float64)
    return 1.0 - n / N


def lorentz_kernel(N, lambda_lorentz: float = 4.0, xp_module=np):
    if lambda_lorentz <= 0:
        raise ValueError("lambda_lorentz must be positive.")
    n = xp_module.arange(N, dtype=xp_module.float64)
    return xp_module.sinh(lambda_lorentz * (1.0 - n / N)) / xp_module.sinh(
        lambda_lorentz
    )


def chebyshev_kernel(
    N,
    kernel: str | np.ndarray = "jackson",
    xp_module=np,
    lambda_lorentz: float = 4.0,
):
    if isinstance(kernel, str):
        key = kernel.strip().lower()
        if key in {"jackson", "j"}:
            return jackson_kernel(N, xp_module)
        if key in {"dirichlet", "none", "raw", "truncated"}:
            return dirichlet_kernel(N, xp_module)
        if key in {"fejer", "fejér", "f"}:
            return fejer_kernel(N, xp_module)
        if key in {"lorentz", "l"}:
            return lorentz_kernel(N, lambda_lorentz=lambda_lorentz, xp_module=xp_module)
        raise ValueError(f"Unknown Chebyshev/KPM kernel: {kernel!r}")

    weights = xp_module.asarray(kernel, dtype=xp_module.float64)
    if len(weights) != N:
        raise ValueError(f"Kernel length {len(weights)} does not match moment count {N}.")
    return weights


def reconstruct_delta_spectrum(
    x: float,
    mu: np.ndarray,
    a: float,
    eps: float = 1e-12,
    kernel: str | np.ndarray = "jackson",
    lambda_lorentz: float = 4.0,
):
    N = len(mu)
    g = chebyshev_kernel(N, kernel=kernel, lambda_lorentz=lambda_lorentz)

    x = float(np.clip(x, -1.0 + eps, 1.0 - eps))
    theta = np.arccos(x)

    series = g[0] * mu[0]
    for n in range(1, N):
        series += 2.0 * g[n] * mu[n] * np.cos(n * theta)

    pref = 1.0 / (a * np.pi * np.sqrt(1.0 - x * x))
    return (pref * series).real


def reconstruct_delta_spectrum_grid(
    x_grid: np.ndarray,
    mu: np.ndarray,
    a: float,
    eps: float = 1e-12,
    kernel: str | np.ndarray = "jackson",
    lambda_lorentz: float = 4.0,
):
    """Vectorized Chebyshev spectrum reconstruction on NumPy or CuPy."""
    xp = gpu_backend.xp()
    x = xp.clip(xp.asarray(x_grid, dtype=xp.float64), -1.0 + eps, 1.0 - eps)
    mu_backend = xp.asarray(mu)
    g = chebyshev_kernel(
        len(mu_backend),
        kernel=kernel,
        xp_module=xp,
        lambda_lorentz=lambda_lorentz,
    )
    cos_theta = xp.cos(xp.arccos(x))

    series = g[0] * mu_backend[0] * xp.ones_like(x, dtype=mu_backend.dtype)
    if len(mu_backend) > 1:
        t_nm1 = xp.ones_like(x, dtype=xp.float64)
        t_n = cos_theta
        series = series + 2.0 * g[1] * mu_backend[1] * t_n
        for n in range(2, len(mu_backend)):
            t_np1 = 2.0 * cos_theta * t_n - t_nm1
            series = series + 2.0 * g[n] * mu_backend[n] * t_np1
            t_nm1, t_n = t_n, t_np1

    pref = 1.0 / (a * xp.pi * xp.sqrt(1.0 - x * x))
    return gpu_backend.asnumpy((pref * series).real)


def chebyshev_moments_ttns(
    psi_left: TreeTensorNetworkState,
    ttns_list: list[TreeTensorNetworkState],
):
    mu = np.zeros(len(ttns_list), dtype=np.complex128)
    for n, ttns in enumerate(ttns_list):
        mu[n] = ttns.scalar_product(psi_left)
    return mu


def psi_eff_ttns(ttns_list, C, ttns0):
    N = len(ttns_list)
    C = np.asarray(C)
    assert C.shape[0] == N, f"C must have first dimension {N}, got {C.shape}"

    ttns_left = copy.deepcopy(ttns0)
    b = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        b[i] = ttns_left.scalar_product(ttns_list[i])

    xp = gpu_backend.xp()
    return xp.asarray(C).conj().T @ xp.asarray(b)


def cheb_vector(H, N, v_ket):
    xp = gpu_backend.xp()
    H_backend = xp.asarray(H)
    psi_list = [xp.asarray(v_ket)]

    if N > 1:
        psi_list.append(H_backend @ psi_list[0])

    for n in range(1, N - 1):
        psi_np1 = 2 * H_backend @ psi_list[n] - psi_list[n - 1]
        psi_list.append(psi_np1)

    return psi_list


def chebyshev_moments_vector(v_bra, psi_list):
    xp = gpu_backend.xp()
    v_bra_backend = xp.asarray(v_bra)
    mu = xp.zeros(len(psi_list), dtype=xp.complex128)
    for n, psi in enumerate(psi_list):
        mu[n] = xp.vdot(v_bra_backend, xp.asarray(psi))
    return gpu_backend.asnumpy(mu)


def _empty_cheb_step_timing(profile_start_ttns=None, profile_stop_ttns=None):
    return {
        "profile_start_ttns": profile_start_ttns,
        "profile_stop_ttns": profile_stop_ttns,
        "profiled_ttns_numbers": [],
        "Hpsin_count": 0,
        "postprocess_count": 0,
        "psi_np1_count": 0,
        "Hpsin_s": 0.0,
        "postprocess_s": 0.0,
        "psi_np1_s": 0.0,
    }


def _record_timed_ttns_number(profile, target_ttns_number):
    if target_ttns_number not in profile["profiled_ttns_numbers"]:
        profile["profiled_ttns_numbers"].append(target_ttns_number)


def merge_cheb_step_timings(profiles, profile_start_ttns=None, profile_stop_ttns=None):
    merged = _empty_cheb_step_timing(profile_start_ttns, profile_stop_ttns)
    numbers_by_component = []
    for profile in profiles:
        if not profile:
            continue
        numbers = list(profile.get("profiled_ttns_numbers", []))
        numbers_by_component.append(numbers)
        merged["Hpsin_count"] += int(profile.get("Hpsin_count", 0))
        merged["postprocess_count"] += int(profile.get("postprocess_count", 0))
        merged["psi_np1_count"] += int(profile.get("psi_np1_count", 0))
        merged["Hpsin_s"] += float(profile.get("Hpsin_s", 0.0))
        merged["postprocess_s"] += float(profile.get("postprocess_s", 0.0))
        merged["psi_np1_s"] += float(profile.get("psi_np1_s", 0.0))
    merged["profiled_ttns_numbers_by_component"] = numbers_by_component
    if numbers_by_component:
        merged["profiled_ttns_numbers"] = numbers_by_component[0]
    return merged


def cheb_ttns_CBC(
    v_ket: TreeTensorNetworkState,
    H_scaled: TreeTensorNetworkOperator,
    N: int,
    svd_params: SVDParameters,
    num_sweeps: int = 2,
    dtype=np.float64,
    profile_start_ttns: int = 1,
    profile_stop_ttns: int | None = None,
):
    if profile_stop_ttns is None:
        profile_stop_ttns = N

    psi_list = [copy.deepcopy(v_ket)]
    profile = _empty_cheb_step_timing(profile_start_ttns, profile_stop_ttns)

    if N > 1:
        target_ttns_number = 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsi0 = apply_ttno_to_ttns(
            ttns=psi_list[0],
            ttno=H_scaled,
            method=ApplicationMethod.HALF_DENSITY_MATRIX,
            svd_params=svd_params,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi1 = truncate_ttns(Hpsi0, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["postprocess_s"] += time.perf_counter() - tic
            profile["postprocess_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi1)

    for n in range(1, N - 1):
        target_ttns_number = n + 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsin = apply_ttno_to_ttns(
            ttns=psi_list[n],
            ttno=H_scaled,
            method=ApplicationMethod.HALF_DENSITY_MATRIX,
            svd_params=svd_params,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi_np1 = add(Hpsin, psi_list[n - 1], c1=2.0, c2=-1.0)
        psi_np1 = truncate_ttns(psi_np1, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["psi_np1_s"] += time.perf_counter() - tic
            profile["psi_np1_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi_np1)

    return psi_list, profile


def cheb_ttns_Density(
    v_ket: TreeTensorNetworkState,
    H_scaled: TreeTensorNetworkOperator,
    N: int,
    svd_params: SVDParameters,
    num_sweeps: int = 2,
    dtype=np.float64,
    profile_start_ttns: int = 1,
    profile_stop_ttns: int | None = None,
):
    if profile_stop_ttns is None:
        profile_stop_ttns = N

    psi_list = [copy.deepcopy(v_ket)]
    profile = _empty_cheb_step_timing(profile_start_ttns, profile_stop_ttns)

    if N > 1:
        target_ttns_number = 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsi0 = apply_ttno_to_ttns(
            ttns=psi_list[0],
            ttno=H_scaled,
            method=ApplicationMethod.DENSITY_MATRIX,
            svd_params=svd_params,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi1 = truncate_ttns(Hpsi0, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["postprocess_s"] += time.perf_counter() - tic
            profile["postprocess_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi1)

    for n in range(1, N - 1):
        target_ttns_number = n + 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsin = apply_ttno_to_ttns(
            ttns=psi_list[n],
            ttno=H_scaled,
            method=ApplicationMethod.DENSITY_MATRIX,
            svd_params=svd_params,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi_np1 = add(Hpsin, psi_list[n - 1], c1=2.0, c2=-1.0)
        psi_np1 = truncate_ttns(psi_np1, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["psi_np1_s"] += time.perf_counter() - tic
            profile["psi_np1_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi_np1)

    return psi_list, profile


def cheb_ttns_SRC(
    v_ket: TreeTensorNetworkState,
    H_scaled: TreeTensorNetworkOperator,
    N: int,
    svd_params: SVDParameters,
    num_sweeps: int = 2,
    dtype=np.float64,
    profile_start_ttns: int = 1,
    profile_stop_ttns: int | None = None,
    seed: int | None = 20260613,
):
    if profile_stop_ttns is None:
        profile_stop_ttns = N

    psi_list = [copy.deepcopy(v_ket)]
    profile = _empty_cheb_step_timing(profile_start_ttns, profile_stop_ttns)
    desired_dimension = int(svd_params.max_bond_dim)

    if N > 1:
        target_ttns_number = 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsi0 = src_ttns_ttno_application_real(
            ttns=psi_list[0],
            ttno=H_scaled,
            desired_dimension=desired_dimension,
            seed=None if seed is None else seed + target_ttns_number,
            dtype=dtype,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi1 = truncate_ttns(Hpsi0, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["postprocess_s"] += time.perf_counter() - tic
            profile["postprocess_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi1)

    for n in range(1, N - 1):
        target_ttns_number = n + 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsin = src_ttns_ttno_application_real(
            ttns=psi_list[n],
            ttno=H_scaled,
            desired_dimension=desired_dimension,
            seed=None if seed is None else seed + target_ttns_number,
            dtype=dtype,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi_np1 = add(Hpsin, psi_list[n - 1], c1=2.0, c2=-1.0)
        psi_np1 = truncate_ttns(psi_np1, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["psi_np1_s"] += time.perf_counter() - tic
            profile["psi_np1_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi_np1)

    return psi_list, profile


def cheb_ttns_Direct_Truncate(
    v_ket: TreeTensorNetworkState,
    H_scaled: TreeTensorNetworkOperator,
    N: int,
    svd_params: SVDParameters,
    num_sweeps: int = 2,
    dtype=np.float64,
    profile_start_ttns: int = 1,
    profile_stop_ttns: int | None = None,
    progress_every: int | None = None,
):
    if profile_stop_ttns is None:
        profile_stop_ttns = N

    psi_list = [copy.deepcopy(v_ket)]
    profile = _empty_cheb_step_timing(profile_start_ttns, profile_stop_ttns)

    if N > 1:
        target_ttns_number = 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsi0 = apply_ttno_to_ttns(
            ttns=psi_list[0],
            ttno=H_scaled,
            method=ApplicationMethod.DIRECT,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi1 = truncate_ttns(Hpsi0, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["postprocess_s"] += time.perf_counter() - tic
            profile["postprocess_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi1)
        if progress_every and len(psi_list) % int(progress_every) == 0:
            print(f"  built {len(psi_list)}/{N} Chebyshev TTNS")

    for n in range(1, N - 1):
        target_ttns_number = n + 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsin = apply_ttno_to_ttns(
            ttns=psi_list[n],
            ttno=H_scaled,
            method=ApplicationMethod.DIRECT,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi_np1 = add(Hpsin, psi_list[n - 1], c1=2.0, c2=-1.0)
        psi_np1 = truncate_ttns(psi_np1, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["psi_np1_s"] += time.perf_counter() - tic
            profile["psi_np1_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi_np1)
        if progress_every and len(psi_list) % int(progress_every) == 0:
            print(f"  built {len(psi_list)}/{N} Chebyshev TTNS")

    return psi_list, profile


def extend_cheb_ttns_Direct_Truncate(
    psi_list: list[TreeTensorNetworkState],
    H_scaled: TreeTensorNetworkOperator,
    target_N: int,
    svd_params: SVDParameters,
    num_sweeps: int = 2,
    dtype=np.float64,
    profile_start_ttns: int = 1,
    profile_stop_ttns: int | None = None,
    progress_every: int | None = None,
):
    if profile_stop_ttns is None:
        profile_stop_ttns = target_N

    target_N = int(target_N)
    if target_N < 1:
        raise ValueError("target_N must be at least 1.")
    if len(psi_list) == 0:
        raise ValueError("psi_list is empty. Build the first vector from psi_ket first.")

    profile = _empty_cheb_step_timing(profile_start_ttns, profile_stop_ttns)
    if target_N <= len(psi_list):
        return psi_list, profile

    if len(psi_list) == 1 and target_N > 1:
        target_ttns_number = 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsi0 = apply_ttno_to_ttns(
            ttns=psi_list[0],
            ttno=H_scaled,
            method=ApplicationMethod.DIRECT,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi1 = truncate_ttns(Hpsi0, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["postprocess_s"] += time.perf_counter() - tic
            profile["postprocess_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi1)
        if progress_every and len(psi_list) % int(progress_every) == 0:
            print(f"  built {len(psi_list)}/{target_N} Chebyshev TTNS")

    for n in range(len(psi_list) - 1, target_N - 1):
        target_ttns_number = n + 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsin = apply_ttno_to_ttns(
            ttns=psi_list[n],
            ttno=H_scaled,
            method=ApplicationMethod.DIRECT,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi_np1 = add(Hpsin, psi_list[n - 1], c1=2.0, c2=-1.0)
        psi_np1 = truncate_ttns(psi_np1, TruncationMethod.SVD, svd_params)
        if do_profile:
            profile["psi_np1_s"] += time.perf_counter() - tic
            profile["psi_np1_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi_np1)
        if progress_every and len(psi_list) % int(progress_every) == 0:
            print(f"  built {len(psi_list)}/{target_N} Chebyshev TTNS")

    return psi_list, profile


def _variational_fit_operator_sum_ttns(
    operators: list[TreeTensorNetworkOperator],
    states: list[TreeTensorNetworkState],
    coeffs,
    init_state: TreeTensorNetworkState,
    svd_params: SVDParameters,
    num_sweeps: int,
    max_iter: int,
    residual_rank: int,
    dtype,
) -> TreeTensorNetworkState:
    fitter = VariationalFitting(
        operators,
        [copy.deepcopy(state) for state in states],
        copy.deepcopy(init_state),
        int(num_sweeps),
        int(max_iter),
        svd_params,
        "one-site",
        list(coeffs),
        residual_rank=int(residual_rank),
        dtype=dtype,
    )
    fitter.run()
    return fitter.get_result_state()


def cheb_ttns_paper(
    v_ket: TreeTensorNetworkState,
    H_scaled: TreeTensorNetworkOperator,
    N: int,
    svd_params: SVDParameters,
    num_sweeps: int = 3,
    dtype=np.float64,
    profile_start_ttns: int = 1,
    profile_stop_ttns: int | None = None,
    max_iter: int = 100,
    residual_rank: int = 0,
):
    """
    Build Chebyshev TTNS by fitting each recurrence target directly.

    Unlike ``cheb_ttns_variational_paper``, this does not first construct
    ``H_scaled @ psi_n``.  Each new vector is obtained by ALS fitting of the
    operator sum target ``H_scaled psi_0`` or
    ``2 H_scaled psi_n - psi_{n-1}``, matching the paper-style sweep.
    """
    if profile_stop_ttns is None:
        profile_stop_ttns = N

    psi_list = [copy.deepcopy(v_ket)]
    profile = _empty_cheb_step_timing(profile_start_ttns, profile_stop_ttns)
    identity_ttno = TreeTensorNetworkOperator.from_hamiltonian(
        Hamiltonian.identity_like(v_ket, dtype=dtype),
        v_ket,
        dtype=dtype,
    )

    if N > 1:
        target_ttns_number = 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        psi1 = _variational_fit_operator_sum_ttns(
            [H_scaled],
            [psi_list[0]],
            [1.0],
            init_state=psi_list[0],
            svd_params=svd_params,
            num_sweeps=num_sweeps,
            max_iter=max_iter,
            residual_rank=residual_rank,
            dtype=dtype,
        )
        if do_profile:
            profile["postprocess_s"] += time.perf_counter() - tic
            profile["postprocess_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi1)

    for n in range(1, N - 1):
        target_ttns_number = n + 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        psi_np1 = _variational_fit_operator_sum_ttns(
            [H_scaled, identity_ttno],
            [psi_list[n], psi_list[n - 1]],
            [2.0, -1.0],
            init_state=psi_list[n],
            svd_params=svd_params,
            num_sweeps=num_sweeps,
            max_iter=max_iter,
            residual_rank=residual_rank,
            dtype=dtype,
        )
        if do_profile:
            profile["psi_np1_s"] += time.perf_counter() - tic
            profile["psi_np1_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi_np1)

    return psi_list, profile


def cheb_ttns_variational_paper(
    v_ket: TreeTensorNetworkState,
    H_scaled: TreeTensorNetworkOperator,
    N: int,
    svd_params: SVDParameters,
    num_sweeps: int = 2,
    dtype=np.float64,
    profile_start_ttns: int = 1,
    profile_stop_ttns: int | None = None,
):
    if profile_stop_ttns is None:
        profile_stop_ttns = N

    psi_list = [copy.deepcopy(v_ket)]
    profile = _empty_cheb_step_timing(profile_start_ttns, profile_stop_ttns)

    if N > 1:
        target_ttns_number = 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsi0 = apply_ttno_to_ttns(
            ttns=psi_list[0],
            ttno=H_scaled,
            method=ApplicationMethod.DIRECT,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi1 = linear_combination(
            [Hpsi0],
            [1.0],
            svd_params.max_bond_dim,
            num_sweeps,
            dtype=dtype,
        )
        if do_profile:
            profile["postprocess_s"] += time.perf_counter() - tic
            profile["postprocess_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi1)

    for n in range(1, N - 1):
        target_ttns_number = n + 2
        do_profile = profile_start_ttns <= target_ttns_number <= profile_stop_ttns

        tic = time.perf_counter()
        Hpsin = apply_ttno_to_ttns(
            ttns=psi_list[n],
            ttno=H_scaled,
            method=ApplicationMethod.DIRECT,
        )
        if do_profile:
            profile["Hpsin_s"] += time.perf_counter() - tic
            profile["Hpsin_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)

        tic = time.perf_counter()
        psi_np1 = linear_combination(
            [Hpsin, psi_list[n - 1]],
            [2.0, -1.0],
            svd_params.max_bond_dim,
            num_sweeps,
            dtype=dtype,
        )
        if do_profile:
            profile["psi_np1_s"] += time.perf_counter() - tic
            profile["psi_np1_count"] += 1
            _record_timed_ttns_number(profile, target_ttns_number)
        psi_list.append(psi_np1)

    return psi_list, profile


def dmrg_min_max_limited_ch3cn(
    initial_state: TreeTensorNetworkState,
    hamiltonian: TreeTensorNetworkOperator,
    max_bond_dim=16,
    num_sweeps=2,
    max_iter=80,
    site="two-site",
):
    svd_params_dmrg = SVDParameters(
        max_bond_dim=int(max_bond_dim),
        rel_tol=0.0,
        total_tol=0.0,
        sum_trunc=False,
    )

    psi0_min = copy.deepcopy(initial_state)
    dm_min = DMRGAlgorithm(
        initial_state=psi0_min,
        hamiltonian=hamiltonian,
        num_sweeps=int(num_sweeps),
        max_iter=int(max_iter),
        svd_params=svd_params_dmrg,
        site=site,
    )
    energies_H = dm_min.run()
    psi_min = dm_min.state
    E_min = float(np.real(psi_min.operator_expectation_value(hamiltonian)))

    minus_ttno = copy.deepcopy(hamiltonian)
    minus_ttno.tensors[minus_ttno.root_id] *= -1.0

    psi0_max = copy.deepcopy(initial_state)
    dm_max = DMRGAlgorithm(
        initial_state=psi0_max,
        hamiltonian=minus_ttno,
        num_sweeps=int(num_sweeps),
        max_iter=int(max_iter),
        svd_params=svd_params_dmrg,
        site=site,
    )
    energies_negH = dm_max.run()
    psi_max = dm_max.state
    E_max = float(np.real(psi_max.operator_expectation_value(hamiltonian)))

    return E_min, E_max, psi_min


def scale_ttno_with_energy_window(
    hamiltonian,
    E_min,
    E_max,
    W_prime=0.9875,
    safety_factor=1.10,
):
    return scale_ttno_with_energy_window_shifted(
        hamiltonian,
        E_min=E_min,
        E_max=E_max,
        W_prime=W_prime,
        safety_factor=safety_factor,
    )


def heff_ttns_no_shift(ttns_list, C, H_scaled):
    N = len(ttns_list)
    C = np.asarray(C)
    assert C.shape[0] == N, f"C must have first dimension {N}, got {C.shape}"

    H = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        H[i, i] = ttns_list[i].operator_expectation_value(H_scaled)
        for j in range(i + 1, N):
            hij = get_matrix_element(
                ttns_list[i].conjugate(),
                H_scaled,
                ttns_list[j],
            )
            H[i, j] = hij
            H[j, i] = hij.conjugate()

    xp = gpu_backend.xp()
    C_backend = xp.asarray(C)
    H_backend = xp.asarray(H)
    return gpu_backend.asnumpy(C_backend.conj().T @ H_backend @ C_backend)


def _ttno_to_fused_physical_ttns(ttno: TreeTensorNetworkOperator):
    """Represent a TTNO as a TTNS by fusing each pair of physical legs."""
    fused_tensors = {}
    physical_dims = {}

    for node_id, node in ttno.nodes.items():
        tensor = ttno.tensors[node_id]
        nvirt = node.nvirt_legs()
        open_shape = tuple(tensor.shape[nvirt:])
        if len(open_shape) != 2:
            raise ValueError(
                f"TTNO node {node_id!r} must have two physical legs, "
                f"got open shape {open_shape}."
            )

        physical_dims[node_id] = open_shape
        fused_tensors[node_id] = tensor.reshape(
            tensor.shape[:nvirt] + (int(np.prod(open_shape)),)
        )

    return TreeTensorNetworkState.from_tensors(ttno, fused_tensors), physical_dims


def _fused_physical_ttns_to_ttno(
    ttns: TreeTensorNetworkState,
    physical_dims: dict[str, tuple[int, int]],
):
    """Undo `_ttno_to_fused_physical_ttns` after TTNS truncation."""
    ttno_tensors = {}

    for node_id, node in ttns.nodes.items():
        if node_id not in physical_dims:
            raise KeyError(f"Missing saved physical dimensions for node {node_id!r}.")

        tensor = ttns.tensors[node_id]
        nvirt = node.nvirt_legs()
        open_shape = tuple(tensor.shape[nvirt:])
        if len(open_shape) != 1:
            raise ValueError(
                f"Fused TTNS node {node_id!r} must have one physical leg, "
                f"got open shape {open_shape}."
            )

        phys_shape = physical_dims[node_id]
        fused_dim = int(np.prod(phys_shape))
        if open_shape[0] != fused_dim:
            raise ValueError(
                f"Fused physical dimension mismatch at node {node_id!r}: "
                f"got {open_shape[0]}, expected {fused_dim} from {phys_shape}."
            )

        ttno_tensors[node_id] = tensor.reshape(tensor.shape[:nvirt] + phys_shape)

    compressed_ttno = TreeTensorNetworkOperator.from_tensors(ttns, ttno_tensors)
    compressed_ttno.orthogonality_center_id = ttns.orthogonality_center_id
    return compressed_ttno


def compress_ttno_via_ttns_svd(
    ttno: TreeTensorNetworkOperator,
    svd_params: SVDParameters,
    return_info: bool = False,
):
    """Compress a TTNO by fusing physical legs, SVD-truncating as a TTNS.

    Each local operator tensor is reshaped from
    ``(virtual..., d_out, d_in)`` to ``(virtual..., d_out * d_in)``.  The
    resulting TTNS is truncated with the usual TTNS SVD routine, then reshaped
    back into a TTNO with the same physical dimensions.
    """
    if svd_params is None:
        raise ValueError("Pass svd_params for TTNO compression.")

    info = {
        "original_size": int(ttno.size()),
        "original_bond_dims": dict(ttno.bond_dims()),
    }

    fused_ttns, physical_dims = _ttno_to_fused_physical_ttns(ttno)
    if len(fused_ttns.nodes) > 1:
        truncated_ttns = truncate_ttns(fused_ttns, TruncationMethod.SVD, svd_params)
    else:
        truncated_ttns = fused_ttns
    compressed_ttno = _fused_physical_ttns_to_ttno(truncated_ttns, physical_dims)

    info.update({
        "compressed_size": int(compressed_ttno.size()),
        "compressed_bond_dims": dict(compressed_ttno.bond_dims()),
        "max_bond_dim": svd_params.max_bond_dim,
        "rel_tol": float(svd_params.rel_tol),
        "total_tol": float(svd_params.total_tol),
    })
    if not return_info:
        return compressed_ttno
    return compressed_ttno, info


def heff_ttns_no_shift_compressed_ttno(
    ttns_list,
    C,
    H_scaled,
    svd_params: SVDParameters,
    return_info: bool = False,
):
    """Compress ``H_scaled`` as a TTNO before calling ``heff_ttns_no_shift``."""
    H_compressed, info = compress_ttno_via_ttns_svd(
        H_scaled,
        svd_params,
        return_info=True,
    )
    Heff = heff_ttns_no_shift(ttns_list, C, H_compressed)
    if not return_info:
        return Heff
    return Heff, info


def _project_heff_matrix(H, C):
    xp = gpu_backend.xp()
    C_backend = xp.asarray(C)
    H_backend = xp.asarray(H)
    return gpu_backend.asnumpy(C_backend.conj().T @ H_backend @ C_backend)


def _exact_heff_element(ttns_list, H_scaled, i, j):
    if i == j:
        return ttns_list[i].operator_expectation_value(H_scaled)
    return get_matrix_element(
        ttns_list[i].conjugate(),
        H_scaled,
        ttns_list[j],
    )


def heff_ttns_apply_then_overlap(
    ttns_list,
    C,
    H_scaled,
    truncate=False,
    svd_params=None,
    force_hermitian=True,
    progress_every=None,
    return_info=False,
):
    """Build Heff by applying H to each TTNS, then taking overlaps.

    With ``truncate=False`` this uses ``ApplicationMethod.DIRECT`` and is an
    exact alternative to repeated TTNS-TTNO-TTNS sandwich contractions, but the
    intermediate ``H|psi_j>`` can have large bond dimensions.  With
    ``truncate=True`` it uses ``ApplicationMethod.HALF_DENSITY_MATRIX`` and the
    supplied ``svd_params`` to keep the intermediate states small.
    """
    N = len(ttns_list)
    C = np.asarray(C)
    assert C.shape[0] == N, f"C must have first dimension {N}, got {C.shape}"

    if truncate and svd_params is None:
        raise ValueError("Pass svd_params when truncate=True.")

    H = np.zeros((N, N), dtype=np.complex128)
    apply_method = (
        ApplicationMethod.HALF_DENSITY_MATRIX
        if truncate
        else ApplicationMethod.DIRECT
    )

    for j, ttns in enumerate(ttns_list):
        if truncate:
            Hpsi_j = apply_ttno_to_ttns(
                ttns=ttns,
                ttno=H_scaled,
                method=apply_method,
                svd_params=svd_params,
            )
        else:
            Hpsi_j = apply_ttno_to_ttns(
                ttns=ttns,
                ttno=H_scaled,
                method=apply_method,
            )

        for i in range(j + 1):
            H[i, j] = Hpsi_j.scalar_product(ttns_list[i])

        if progress_every and (j + 1) % int(progress_every) == 0:
            print(f"  apply-then-overlap Heff columns: {j + 1}/{N}")

    H = H + np.triu(H, k=1).conj().T
    if force_hermitian:
        H = 0.5 * (H + H.conj().T)

    Heff = _project_heff_matrix(H, C)
    if not return_info:
        return Heff

    return Heff, {
        "raw_dim": N,
        "projected_dim": int(C.shape[1]),
        "truncate": bool(truncate),
        "application_method": APPLICATION_METHOD_LABELS.get(apply_method, str(apply_method)),
        "force_hermitian": bool(force_hermitian),
    }


def _random_product_ttns_like(ttns, rng, dtype=None):
    dtypes = [tensor.dtype for tensor in ttns.tensors.values()]
    if dtype is None:
        dtype = np.result_type(*dtypes)
    dtype = np.dtype(dtype)
    complex_random = np.issubdtype(dtype, np.complexfloating)

    tensors = {}
    for node_id, node in ttns.nodes.items():
        neighbour_shape = (1,) * node.nneighbours()
        open_shape = tuple(node.open_dimensions())
        if complex_random:
            local = np.exp(2j * np.pi * rng.random(open_shape)).astype(dtype, copy=False)
        else:
            local = rng.choice((-1.0, 1.0), size=open_shape).astype(dtype, copy=False)
        tensors[node_id] = local.reshape(neighbour_shape + open_shape)
    return TreeTensorNetworkState.from_tensors(ttns, tensors)


def heff_ttns_apply_then_stochastic_overlap(
    ttns_list,
    C,
    H_scaled,
    num_samples=64,
    seed=20260705,
    exact_diagonal=True,
    exact_prefix=0,
    exact_region="leading",
    force_hermitian=True,
    dtype=None,
    progress_every=None,
    return_info=False,
):
    """Apply H exactly, then estimate TTNS-TTNS overlaps by random sketches.

    The estimator draws random product TTNS ``|r_s>`` with
    ``E[|r_s><r_s|] = I`` and uses
    ``<psi_i|H|psi_j> ~= mean_s <psi_i|r_s><r_s|Hpsi_j>``.
    ``H|psi_j>`` is built with ``ApplicationMethod.DIRECT`` and is not
    truncated, so the approximation is only in the final overlap contraction.

    ``exact_prefix`` can keep a leading region exact.  With
    ``exact_region="leading"``, every matrix element touching one of the first
    ``exact_prefix`` basis vectors is exact; with ``"block"``, only the leading
    block is exact; with ``"diagonal_prefix"``, only the first prefix diagonal
    entries are exact.
    """
    N = len(ttns_list)
    C = np.asarray(C)
    assert C.shape[0] == N, f"C must have first dimension {N}, got {C.shape}"

    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    exact_prefix = max(0, min(int(exact_prefix), N))
    exact_region = str(exact_region).lower()
    if exact_region not in {"diagonal_prefix", "block", "leading"}:
        raise ValueError("exact_region must be 'diagonal_prefix', 'block', or 'leading'.")

    def use_exact(i, j):
        if exact_diagonal and i == j:
            return True
        if exact_region == "diagonal_prefix":
            return i == j and i < exact_prefix
        if exact_region == "block":
            return i < exact_prefix and j < exact_prefix
        return i < exact_prefix or j < exact_prefix

    rng = np.random.default_rng(seed)
    random_states = [
        _random_product_ttns_like(ttns_list[0], rng, dtype=dtype)
        for _ in range(num_samples)
    ]

    left_sketch = np.zeros((N, num_samples), dtype=np.complex128)
    for sample, random_state in enumerate(random_states):
        for i, ttns in enumerate(ttns_list):
            left_sketch[i, sample] = random_state.scalar_product(ttns)

    H = np.zeros((N, N), dtype=np.complex128)
    filled = np.zeros((N, N), dtype=bool)
    exact_elements = 0
    stochastic_elements = 0

    for i in range(N):
        for j in range(i, N):
            if use_exact(i, j):
                hij = _exact_heff_element(ttns_list, H_scaled, i, j)
                H[i, j] = hij
                H[j, i] = hij.conjugate()
                filled[i, j] = True
                filled[j, i] = True
                exact_elements += 1

    for j, ttns in enumerate(ttns_list):
        pending_rows = [i for i in range(j + 1) if not filled[i, j]]
        if not pending_rows:
            if progress_every and (j + 1) % int(progress_every) == 0:
                print(f"  stochastic-overlap Heff columns: {j + 1}/{N}")
            continue

        Hpsi_j = apply_ttno_to_ttns(
            ttns=ttns,
            ttno=H_scaled,
            method=ApplicationMethod.DIRECT,
        )
        right_sketch = np.zeros(num_samples, dtype=np.complex128)
        for sample, random_state in enumerate(random_states):
            right_sketch[sample] = Hpsi_j.scalar_product(random_state)
        column_estimate = left_sketch[:, :] @ right_sketch / num_samples
        for i in pending_rows:
            hij = column_estimate[i]
            H[i, j] = hij
            H[j, i] = hij.conjugate()
            filled[i, j] = True
            filled[j, i] = True
            stochastic_elements += 1

        if progress_every and (j + 1) % int(progress_every) == 0:
            print(f"  stochastic-overlap Heff columns: {j + 1}/{N}")

    if force_hermitian:
        H = 0.5 * (H + H.conj().T)

    Heff = _project_heff_matrix(H, C)
    if not return_info:
        return Heff

    return Heff, {
        "raw_dim": N,
        "projected_dim": int(C.shape[1]),
        "num_samples": num_samples,
        "seed": seed,
        "exact_diagonal": bool(exact_diagonal),
        "exact_prefix": exact_prefix,
        "exact_region": exact_region,
        "exact_elements_upper": exact_elements,
        "stochastic_elements_upper": stochastic_elements,
        "force_hermitian": bool(force_hermitian),
        "application_method": APPLICATION_METHOD_LABELS[ApplicationMethod.DIRECT],
    }


def _stochastic_hpsi_src(ttns, H_scaled, stochastic_bond_dim, seed, dtype):
    svd_params = SVDParameters(
        max_bond_dim=int(stochastic_bond_dim),
        rel_tol=0.0,
        total_tol=0.0,
        sum_trunc=False,
    )
    return apply_ttno_by_method(
        ttns=ttns,
        ttno=H_scaled,
        method=ApplicationMethod.SRC,
        svd_params=svd_params,
        seed=seed,
    )


def _cbc_truncated_hpsi(ttns, H_scaled, svd_params):
    return apply_ttno_to_ttns(
        ttns=ttns,
        ttno=H_scaled,
        method=ApplicationMethod.HALF_DENSITY_MATRIX,
        svd_params=svd_params,
    )


def heff_ttns_hybrid_stochastic(
    ttns_list,
    C,
    H_scaled,
    exact_prefix=50,
    stochastic_bond_dim=16,
    stochastic_samples=1,
    seed=20260613,
    exact_region="diagonal_prefix",
    exact_diagonal=True,
    cbc_max_bond_dim=None,
    svd_params=None,
    force_hermitian=True,
    dtype=np.float64,
    progress_every=None,
    return_info=False,
):
    """Build Heff with an exact leading region and CBC-truncated tail.

    Tail entries first approximate ``H|psi_j>`` with the CBC
    (``HALF_DENSITY_MATRIX``) TTNO application, then compute
    ``<psi_i|Hpsi_j>`` as a TTNS-TTNS overlap.

    ``exact_region="diagonal_prefix"`` makes only the first ``exact_prefix``
    diagonal entries exact via ``operator_expectation_value``.  ``"block"``
    makes the leading ``exact_prefix`` by ``exact_prefix`` block exact.
    ``"leading"`` makes every entry touching one of the first ``exact_prefix``
    vectors exact.  For ``"block"`` and ``"leading"``, ``exact_diagonal=True``
    also makes all diagonal entries exact.
    """
    N = len(ttns_list)
    C = np.asarray(C)
    assert C.shape[0] == N, f"C must have first dimension {N}, got {C.shape}"

    exact_prefix = max(0, min(int(exact_prefix), N))
    if cbc_max_bond_dim is None:
        cbc_max_bond_dim = stochastic_bond_dim
    cbc_max_bond_dim = int(cbc_max_bond_dim)
    if cbc_max_bond_dim < 1:
        raise ValueError("cbc_max_bond_dim must be at least 1.")
    if svd_params is None:
        svd_params = SVDParameters(
            max_bond_dim=cbc_max_bond_dim,
            rel_tol=0.0,
            total_tol=0.0,
            sum_trunc=False,
        )

    exact_region = str(exact_region).lower()
    if exact_region not in {"diagonal_prefix", "block", "leading"}:
        raise ValueError("exact_region must be 'diagonal_prefix', 'block', or 'leading'.")

    def use_exact(i, j):
        if exact_region == "diagonal_prefix":
            return i == j and i < exact_prefix
        if exact_diagonal and i == j:
            return True
        if exact_region == "block":
            return i < exact_prefix and j < exact_prefix
        return i < exact_prefix or j < exact_prefix

    H = np.zeros((N, N), dtype=np.complex128)
    filled = np.zeros((N, N), dtype=bool)
    exact_elements = 0
    cbc_elements = 0
    cbc_columns = 0

    for i in range(N):
        for j in range(i, N):
            if use_exact(i, j):
                hij = _exact_heff_element(ttns_list, H_scaled, i, j)
                H[i, j] = hij
                H[j, i] = hij.conjugate()
                filled[i, j] = True
                filled[j, i] = True
                exact_elements += 1

    for j in range(N):
        pending_rows = [i for i in range(j + 1) if not filled[i, j]]
        if not pending_rows:
            continue

        cbc_columns += 1
        Hpsi_j = _cbc_truncated_hpsi(ttns_list[j], H_scaled, svd_params)
        for row_index, i in enumerate(pending_rows):
            hij = Hpsi_j.scalar_product(ttns_list[i])
            H[i, j] = hij
            H[j, i] = hij.conjugate()
            filled[i, j] = True
            filled[j, i] = True
            cbc_elements += 1

        if progress_every and cbc_columns % int(progress_every) == 0:
            print(
                "  CBC-truncated Heff columns:",
                cbc_columns,
                "last column:",
                j,
                "pending rows:",
                len(pending_rows),
            )

    if force_hermitian:
        H = 0.5 * (H + H.conj().T)

    Heff = _project_heff_matrix(H, C)
    if not return_info:
        return Heff

    return Heff, {
        "raw_dim": N,
        "projected_dim": int(C.shape[1]),
        "exact_prefix": exact_prefix,
        "exact_region": exact_region,
        "exact_diagonal": bool(exact_diagonal),
        "exact_elements_upper": exact_elements,
        "cbc_elements_upper": cbc_elements,
        "cbc_columns": cbc_columns,
        "cbc_max_bond_dim": cbc_max_bond_dim,
        "cbc_application_method": APPLICATION_METHOD_LABELS[ApplicationMethod.HALF_DENSITY_MATRIX],
        "legacy_stochastic_bond_dim_arg": stochastic_bond_dim,
    }


def _ttns_overlap_matrix(ttns_list):
    overlap = np.zeros((len(ttns_list), len(ttns_list)), dtype=np.complex128)
    for i, ttns in enumerate(ttns_list):
        overlap[i, i] = ttns.scalar_product()
        for j in range(i + 1, len(ttns_list)):
            overlap[i, j] = ttns_list[j].scalar_product(ttns_list[i])
            overlap[j, i] = overlap[i, j].conjugate()
    return 0.5 * (overlap + overlap.conj().T)


def heff_ttns_chebyshev_overlap(
    ttns_list,
    C,
    ovp=None,
    H_scaled=None,
    force_hermitian=True,
):
    """Build an effective Hamiltonian from Chebyshev TTNS overlaps.

    For a Chebyshev basis ``psi_n = T_n(H_scaled) psi_0``, the recurrence gives
    ``H psi_0 = psi_1`` and
    ``H psi_j = 0.5 * (psi_{j + 1} + psi_{j - 1})`` for ``j >= 1``.  This
    avoids the full set of ``<psi_i|H|psi_j>`` TTNS-TTNO-TTNS contractions.
    If the Chebyshev TTNS were truncated or variationally fitted, this is an
    approximation because the recurrence residual is ignored.

    ``C`` is the orthogonalization matrix for the first ``C.shape[0]`` TTNS
    vectors.  ``ttns_list`` may contain exactly those vectors, or one extra
    Chebyshev vector ``psi_N``.  The extra vector supplies the last diagonal
    element without an exact TTNO expectation value.  If it is not present,
    pass ``H_scaled`` so that only ``<psi_{N-1}|H|psi_{N-1}>`` is computed
    exactly.
    """
    C = np.asarray(C)
    basis_size = int(C.shape[0])
    if basis_size < 1:
        raise ValueError("C must have at least one row.")
    if len(ttns_list) < basis_size:
        raise ValueError(
            f"ttns_list has {len(ttns_list)} vectors but C needs {basis_size}."
        )

    if ovp is None:
        overlap = _ttns_overlap_matrix(ttns_list[: min(len(ttns_list), basis_size + 1)])
    else:
        overlap = np.asarray(ovp, dtype=np.complex128)
        if overlap.shape[0] < basis_size or overlap.shape[1] < basis_size:
            raise ValueError(
                f"ovp shape {overlap.shape} is too small for basis size {basis_size}."
            )
        overlap = overlap[:basis_size, :basis_size]
        overlap = 0.5 * (overlap + overlap.conj().T)

    H = np.zeros((basis_size, basis_size), dtype=np.complex128)

    if basis_size == 1:
        if len(ttns_list) >= 2:
            if ovp is None:
                H[0, 0] = overlap[0, 1]
            else:
                H[0, 0] = ttns_list[1].scalar_product(ttns_list[0])
        elif H_scaled is not None:
            H[0, 0] = ttns_list[0].operator_expectation_value(H_scaled)
        else:
            raise ValueError("Need psi_1 or H_scaled to build a one-vector Heff.")
    else:
        H[:, 0] = overlap[:basis_size, 1]
        for j in range(1, basis_size - 1):
            H[:, j] = 0.5 * (
                overlap[:basis_size, j + 1] + overlap[:basis_size, j - 1]
            )

        if len(ttns_list) > basis_size:
            if ovp is None and overlap.shape[0] > basis_size:
                last_next_overlap = overlap[basis_size - 1, basis_size]
            else:
                last_next_overlap = ttns_list[basis_size].scalar_product(
                    ttns_list[basis_size - 1]
                )
            H[basis_size - 1, basis_size - 1] = 0.5 * (
                last_next_overlap + overlap[basis_size - 1, basis_size - 2]
            )
        elif H_scaled is not None:
            H[basis_size - 1, basis_size - 1] = ttns_list[
                basis_size - 1
            ].operator_expectation_value(H_scaled)
        else:
            raise ValueError(
                "Need an extra Chebyshev vector psi_N or H_scaled for the last "
                "diagonal Heff element."
            )

        H[: basis_size - 1, basis_size - 1] = H[
            basis_size - 1, : basis_size - 1
        ].conjugate()

    if force_hermitian:
        H = 0.5 * (H + H.conj().T)

    xp = gpu_backend.xp()
    C_backend = xp.asarray(C)
    H_backend = xp.asarray(H)
    return gpu_backend.asnumpy(C_backend.conj().T @ H_backend @ C_backend)


def rescale_ttno_limited_ch3cn(
    initial_state,
    hamiltonian,
    max_bond_dim=16,
    num_sweeps=2,
    max_iter=80,
    W_prime=0.9875,
    safety_factor=1.10,
    site="two-site",
):
    E_min, E_max, psi_min = dmrg_min_max_limited_ch3cn(
        initial_state=initial_state,
        hamiltonian=hamiltonian,
        max_bond_dim=max_bond_dim,
        num_sweeps=num_sweeps,
        max_iter=max_iter,
        site=site,
    )
    H_scaled, a, shift = scale_ttno_with_energy_window(
        hamiltonian,
        E_min=E_min,
        E_max=E_max,
        W_prime=W_prime,
        safety_factor=safety_factor,
    )
    return H_scaled, a, shift, E_min, E_max, psi_min
