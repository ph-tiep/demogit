# -*- coding: utf-8 -*-
"""
Inference latency benchmark — measures per-stage wall-clock time on CPU.

Stages measured (single inference step):
  1. VAE encoder        : input tensor  -> latent code (mu)
  2. EKF predict-update : one EKF step  -> state estimate + NIS
  3. GAT Q-network      : latent states -> Q-values for all agents

Each stage is timed over N_REPS repetitions; the first 50 are discarded as
warm-up. Results are reported as mean ± std in microseconds and total in ms.

Usage:
    python measure_runtime.py
"""

import time
import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vae_model import VAE
from digital_twin import EKF
from madrl_gat import MADRLGATNetwork

# ---------------------------------------------------------------------------
# Config — must match run_all_datasets.py CFG
# ---------------------------------------------------------------------------
DATASETS = {
    'antwerp': {
        'vae_input_dim': 84,   # sigfox 84-BS RSSI matrix
        'vae_latent':    16,
        'vae_hidden':    64,
        'num_agents':    5,
        'num_actions':   5,
        'gat_hidden':    16,
        'gat_out':       8,
        'num_heads':     2,
    },
    'lorawan': {
        'vae_input_dim': 5,    # 5-feature LoRaWAN matrix
        'vae_latent':    3,
        'vae_hidden':    32,
        'num_agents':    5,
        'num_actions':   5,
        'gat_hidden':    16,
        'gat_out':       8,
        'num_heads':     2,
    },
    'indoor': {
        'vae_input_dim': 5,    # indoor RSSI matrix
        'vae_latent':    3,
        'vae_hidden':    32,
        'num_agents':    5,
        'num_actions':   5,
        'gat_hidden':    16,
        'gat_out':       8,
        'num_heads':     2,
    },
}

N_REPS   = 2000   # total repetitions
N_WARMUP = 200    # discard first N_WARMUP as warm-up


def time_stage(fn, n_reps=N_REPS, n_warmup=N_WARMUP):
    """Time fn() over n_reps calls; skip first n_warmup. Returns μs array."""
    times = []
    for i in range(n_reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        if i >= n_warmup:
            times.append((t1 - t0) * 1e6)   # convert to microseconds
    return np.array(times)


def benchmark_dataset(name, cfg):
    print(f"\n{'='*55}")
    print(f"  Dataset: {name.upper()}")
    print(f"{'='*55}")

    device = torch.device('cpu')

    # ── Stage 1: VAE encoder ───────────────────────────────────────────
    vae = VAE(
        input_dim=cfg['vae_input_dim'],
        latent_dim=cfg['vae_latent'],
        hidden_dim=cfg['vae_hidden'],
    ).to(device)
    vae.eval()

    x_dummy = torch.randn(1, cfg['vae_input_dim'], device=device)

    @torch.no_grad()
    def vae_forward():
        mu, _ = vae.encode(x_dummy)
        return mu

    t_vae = time_stage(vae_forward)

    # ── Stage 2: EKF predict-update ────────────────────────────────────
    ekf = EKF(state_dim=2, obs_dim=2, process_noise=0.5, obs_noise=0.5)
    ekf.reset([0.0, 0.0])
    obs_dummy = np.array([-100.0, 5.0], dtype=np.float64)

    def ekf_step():
        ekf.predict()
        ekf.update(obs_dummy)

    t_ekf = time_stage(ekf_step)

    # ── Stage 3: GAT Q-network ─────────────────────────────────────────
    N = cfg['num_agents']
    obs_dim = cfg['vae_latent']
    net = MADRLGATNetwork(
        agent_obs_dim=obs_dim,
        gat_hidden=cfg['gat_hidden'],
        gat_out=cfg['gat_out'],
        num_heads=cfg['num_heads'],
        num_actions=cfg['num_actions'],
    ).to(device)
    net.eval()

    states_dummy = torch.randn(N, obs_dim, device=device)
    adj_dummy    = torch.ones(N, N, device=device)

    @torch.no_grad()
    def gat_forward():
        return net(states_dummy, adj_dummy)

    t_gat = time_stage(gat_forward)

    # ── Report ─────────────────────────────────────────────────────────
    total_mean = (t_vae.mean() + t_ekf.mean() + t_gat.mean()) / 1e3   # ms
    total_std  = np.sqrt(t_vae.var() + t_ekf.var() + t_gat.var()) / 1e3

    print(f"  VAE encoder        : {t_vae.mean():7.2f} ± {t_vae.std():.2f} μs")
    print(f"  EKF predict-update : {t_ekf.mean():7.2f} ± {t_ekf.std():.2f} μs")
    print(f"  GAT Q-network      : {t_gat.mean():7.2f} ± {t_gat.std():.2f} μs")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Total per step     : {total_mean:7.3f} ± {total_std:.3f} ms")

    return {
        'vae_us':   (float(t_vae.mean()),  float(t_vae.std())),
        'ekf_us':   (float(t_ekf.mean()),  float(t_ekf.std())),
        'gat_us':   (float(t_gat.mean()),  float(t_gat.std())),
        'total_ms': (float(total_mean),    float(total_std)),
    }


def main():
    print("\nInference Latency Benchmark (CPU, single-step, no batching)")
    print(f"Warm-up: {N_WARMUP} reps | Measured: {N_REPS - N_WARMUP} reps")

    results = {}
    for name, cfg in DATASETS.items():
        results[name] = benchmark_dataset(name, cfg)

    print(f"\n{'='*55}")
    print("  SUMMARY TABLE (mean latency per step)")
    print(f"{'='*55}")
    print(f"  {'Stage':<22} {'Antwerp':>10} {'LoRaWAN':>10} {'Indoor':>10}  (μs)")
    print(f"  {'-'*52}")
    for stage, key in [('VAE encoder', 'vae_us'), ('EKF predict-update', 'ekf_us'), ('GAT Q-network', 'gat_us')]:
        row = "  " + f"{stage:<22}"
        for ds in ['antwerp', 'lorawan', 'indoor']:
            row += f"{results[ds][key][0]:>10.2f}"
        print(row + "  μs")
    print(f"  {'─'*52}")
    row = "  " + f"{'Total per step':<22}"
    for ds in ['antwerp', 'lorawan', 'indoor']:
        row += f"{results[ds]['total_ms'][0]:>10.3f}"
    print(row + "  ms")
    print()


if __name__ == '__main__':
    main()
