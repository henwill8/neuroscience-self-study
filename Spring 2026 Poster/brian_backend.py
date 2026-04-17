"""Brian2 device selection: prefer CUDA, then C++ standalone, then CPU runtime."""
from __future__ import annotations

import logging
import subprocess
import sys

_log = logging.getLogger(__name__)


def _nvidia_driver_reports_gpu():
    try:
        r = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and bool((r.stdout or '').strip())


def _prefer_cython_codegen():
    """Best-effort faster CPU codegen (not used for cuda_standalone)."""
    from brian2 import prefs

    try:
        from brian2.codegen.runtime.cython_rt import extension_manager as _em

        _em.create_extensions()
        prefs.codegen.target = 'cython'
    except Exception:
        prefs.codegen.target = 'numpy'


def _log_backend_resolved(params, used, *, note=None):
    """Log requested brian_device vs actual backend (always call before returning from configure)."""
    from brian2 import prefs

    requested = params.get('brian_device', 'auto')
    suffix = f"; {note}" if note else ""
    if used == 'cuda_standalone':
        msg = f"Brian2 backend: brian_device={requested!r} -> cuda_standalone (brian2cuda){suffix}"
    elif used == 'cpp_standalone':
        msg = f"Brian2 backend: brian_device={requested!r} -> cpp_standalone{suffix}"
    else:
        msg = (
            f"Brian2 backend: brian_device={requested!r} -> runtime "
            f"(codegen={prefs.codegen.target}){suffix}"
        )
    print(msg, flush=True)
    _log.info(msg)


def configure_brian_backend(params):
    """
    Call before any NeuronGroup, Synapses, or SpikeGeneratorGroup are created.

    params:
        brian_device : 'auto' | 'cuda' | 'cpu'
            auto — use cuda_standalone if brian2cuda imports and nvidia-smi sees a GPU.
            cuda — require GPU (raises if setup fails).
            cpu — force CPU runtime (cython or numpy codegen).

    Writes params['_brian_backend_used'] as 'cuda_standalone', 'cpp_standalone', or 'runtime'.

    Notes
    -----
    Brian2CUDA is officially supported on Linux with CUDA Toolkit / nvcc. Windows support is
    limited; see https://brian2cuda.readthedocs.io/en/latest/introduction/install.html
    """
    from brian2 import restore_initial_state, set_device

    restore_initial_state()

    mode = str(params.get('brian_device', 'auto')).strip().lower()
    if mode not in ('auto', 'cuda', 'cpu'):
        _log.warning("Unknown brian_device=%r; using 'auto'.", mode)
        mode = 'auto'

    if mode == 'cpu':
        set_device('runtime')
        _prefer_cython_codegen()
        params['_brian_backend_used'] = 'runtime'
        _log_backend_resolved(params, 'runtime', note='brian_device=cpu')
        return 'runtime'

    if mode == 'auto' and not _nvidia_driver_reports_gpu():
        try:
            set_device('cpp_standalone', build_on_run=True)
            params['_brian_backend_used'] = 'cpp_standalone'
            _log_backend_resolved(params, 'cpp_standalone', note='no GPU listed by nvidia-smi')
            return 'cpp_standalone'
        except Exception as err:
            _log.warning('cpp_standalone failed (%s); falling back to CPU runtime.', err)
            set_device('runtime')
            _prefer_cython_codegen()
            params['_brian_backend_used'] = 'runtime'
            _log_backend_resolved(params, 'runtime', note=f'no GPU; cpp_standalone init failed: {err}')
            return 'runtime'

    try:
        import brian2cuda  # noqa: F401 — registers cuda_standalone device
    except ImportError as err:
        if mode == 'cuda':
            raise RuntimeError(
                "brian_device='cuda' but brian2cuda is not installed or failed to import. "
                "On Linux: pip install brian2cuda (requires CUDA toolkit). "
                "See https://brian2cuda.readthedocs.io/en/latest/introduction/install.html"
            ) from err
        try:
            set_device('cpp_standalone', build_on_run=True)
            params['_brian_backend_used'] = 'cpp_standalone'
            _log_backend_resolved(params, 'cpp_standalone', note='brian2cuda not importable')
            return 'cpp_standalone'
        except Exception as cpp_err:
            _log.warning('cpp_standalone failed (%s); falling back to CPU runtime.', cpp_err)
            set_device('runtime')
            _prefer_cython_codegen()
            params['_brian_backend_used'] = 'runtime'
            _log_backend_resolved(params, 'runtime', note=f'brian2cuda not importable; cpp_standalone init failed: {cpp_err}')
            return 'runtime'

    if sys.platform == 'win32':
        _log.warning(
            'Brian2CUDA targets Linux; on Windows, GPU runs may fail. '
            'Use WSL2/Linux + CUDA or set params["brian_device"] = "cpu".'
        )

    try:
        set_device('cuda_standalone', build_on_run=True)
    except Exception as err:
        if mode == 'cuda':
            raise RuntimeError(
                'brian_device="cuda" but cuda_standalone could not be initialized. '
                'Check CUDA toolkit, nvcc, and GPU drivers.'
            ) from err
        _log.warning('cuda_standalone failed (%s); trying cpp_standalone.', err)
        try:
            set_device('cpp_standalone', build_on_run=True)
            params['_brian_backend_used'] = 'cpp_standalone'
            _log_backend_resolved(params, 'cpp_standalone', note=f'cuda_standalone init failed: {err}')
            return 'cpp_standalone'
        except Exception as cpp_err:
            _log.warning('cpp_standalone failed (%s); falling back to CPU runtime.', cpp_err)
            set_device('runtime')
            _prefer_cython_codegen()
            params['_brian_backend_used'] = 'runtime'
            _log_backend_resolved(params, 'runtime', note=f'cuda_standalone init failed: {err}; cpp_standalone init failed: {cpp_err}')
            return 'runtime'

    params['_brian_backend_used'] = 'cuda_standalone'
    _log_backend_resolved(params, 'cuda_standalone')
    return 'cuda_standalone'


def effective_profile(params):
    """Profiling is not supported on standalone backends; disable to avoid build/runtime errors."""
    if params.get('_brian_backend_used') in ('cuda_standalone', 'cpp_standalone'):
        return False
    return bool(params.get('doProfile', False))
