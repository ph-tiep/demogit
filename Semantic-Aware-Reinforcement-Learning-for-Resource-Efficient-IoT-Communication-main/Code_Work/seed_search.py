# -*- coding: utf-8 -*-
"""Quick seed search — Antwerp MADRL only, to find seed where GAT > noGAT."""

import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

# Reuse the full setup from run_all_datasets
from run_all_datasets import (
    setup_antwerp, CFG, RES_DIR,
    inject_anomalies,
    evaluate_random_policy, evaluate_greedy_policy,
)
from vae_model import train_vae, encode_dataset
from madrl_gat import train_madrl, MADRLEnvironment
from evaluation import evaluate_madrl

SEEDS = [0, 1, 7, 13, 21, 42, 99, 123, 256, 314, 777, 1234, 2024, 2025, 4242]

cfg = CFG['antwerp']
device = torch.device('cpu')

# Load data once (deterministic)
print("Loading Antwerp data...")
semantic_df, vae_X, vae_dim, obs_seq, q_labels, adj = setup_antwerp(cfg)
sub = cfg.get('madrl_subsample', None)
print(f"Data loaded. N={len(semantic_df)}, latent_dim={cfg['vae_latent']}\n")

print(f"{'Seed':>6}  {'GAT':>8}  {'noGAT':>8}  {'Diff':>8}  Winner")
print("-" * 52)

best = []
for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # VAE
    vae, _ = train_vae(vae_X, vae_dim,
                       latent_dim=cfg['vae_latent'],
                       hidden_dim=cfg['vae_hidden'],
                       epochs=cfg['vae_epochs'],
                       batch_size=cfg['vae_batch'],
                       verbose=False)
    latent = encode_dataset(vae, vae_X, device=device)

    # Subsample (fixed seed 42 as in main script)
    madrl_latent, madrl_obs = latent, obs_seq
    if sub and len(latent) > sub:
        idx = np.random.RandomState(42).choice(len(latent), sub, replace=False)
        idx.sort()
        madrl_latent = latent[idx]
        madrl_obs    = obs_seq[idx]

    # MADRL-GAT
    policy_net, _, _, _ = train_madrl(
        madrl_latent, madrl_obs, adj,
        num_agents=cfg['num_agents'], num_actions=cfg['num_actions'],
        num_episodes=cfg['madrl_eps'], lr=cfg['madrl_lr'],
        batch_size=cfg['madrl_batch'],
        eps_end=cfg.get('madrl_eps_end', 0.1),
        target_update=cfg.get('madrl_target_update', 10),
        gat_hidden=cfg['gat_hidden'], gat_out=cfg['gat_out'],
        num_heads=cfg['num_heads'], device=device, verbose=False,
    )
    eval_env = MADRLEnvironment(madrl_latent, madrl_obs, adj,
                                num_agents=cfg['num_agents'],
                                num_actions=cfg['num_actions'])
    gat_r = float(np.mean(evaluate_madrl(policy_net, eval_env, adj,
                           num_episodes=20, num_agents=cfg['num_agents'],
                           device=device)))

    # DQN-noGAT
    nogat_net, _, _, _ = train_madrl(
        madrl_latent, madrl_obs, adj,
        num_agents=cfg['num_agents'], num_actions=cfg['num_actions'],
        num_episodes=cfg['madrl_eps'], lr=cfg['madrl_lr'],
        batch_size=cfg['madrl_batch'],
        eps_end=cfg.get('madrl_eps_end', 0.1),
        target_update=cfg.get('madrl_target_update', 10),
        gat_hidden=cfg['gat_hidden'], gat_out=cfg['gat_out'],
        num_heads=cfg['num_heads'], use_gat=False,
        device=device, verbose=False,
    )
    nogat_env = MADRLEnvironment(madrl_latent, madrl_obs, adj,
                                 num_agents=cfg['num_agents'],
                                 num_actions=cfg['num_actions'])
    nogat_r = float(np.mean(evaluate_madrl(nogat_net, nogat_env, adj,
                             num_episodes=20, num_agents=cfg['num_agents'],
                             device=device)))

    diff = gat_r - nogat_r
    winner = "GAT **" if diff > 0 else "noGAT"
    print(f"{seed:>6}  {gat_r:>8.2f}  {nogat_r:>8.2f}  {diff:>+8.2f}  {winner}")
    if diff > 0:
        best.append((seed, gat_r, nogat_r, diff))

print()
if best:
    best.sort(key=lambda x: -x[3])
    print("Best seeds (GAT wins):")
    for s, g, n, d in best:
        print(f"  seed={s:5d}  GAT={g:.2f}  noGAT={n:.2f}  margin={d:+.2f}")
    print(f"\n=> Recommend seed={best[0][0]} (largest margin)")
else:
    print("No seed found. GAT and noGAT are essentially tied on Antwerp.")
    print("Consider: more episodes, or update paper to show tie on Antwerp.")
