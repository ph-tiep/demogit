# -*- coding: utf-8 -*-
"""
Quick reproduction test: original config (no obs_noise_std, global seed=42).
Does NOT modify any existing code.
"""
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import random
from collections import deque
import torch.nn.functional as F

from run_all_datasets import setup_antwerp, CFG
from vae_model import train_vae, encode_dataset
from evaluation import evaluate_madrl

# ── Import network classes only (not MADRLEnvironment) ──────────────────────
from madrl_gat import MADRLGATNetwork, DQNNoGAT, ReplayBuffer

# ── Original MADRLEnvironment (no obs_noise_std) ────────────────────────────
class OriginalEnv:
    def __init__(self, latent_states, original_obs, adj_matrix,
                 num_agents=10, num_actions=5):
        self.latent      = latent_states.astype(np.float32)
        self.orig        = original_obs.astype(np.float32)
        self.adj         = adj_matrix
        self.num_agents  = num_agents
        self.num_actions = num_actions
        self.T = len(latent_states)
        self.t = 0

    def _broadcast(self, g):
        return np.tile(g, (self.num_agents, 1))   # all agents see SAME state

    def reset(self):
        self.t = 0
        return self._broadcast(self.latent[0])

    @staticmethod
    def _reward(orig_row, actions):
        mean_rssi = float(orig_row[0])
        num_bs    = float(orig_row[1])
        RSSI_MIN, RSSI_MAX = -145.0, -50.0
        norm_rssi  = float(np.clip((mean_rssi - RSSI_MIN) / (RSSI_MAX - RSSI_MIN), 0., 1.))
        action_norm = float(np.mean(actions)) / 4.0
        effective_s = norm_rssi + action_norm * (1.0 - norm_rssi)
        norm_bs     = min(num_bs / 20.0, 1.0)
        coop_bonus  = max(0.0, 1.0 - float(np.std(actions)) / 2.0)
        return 5.0 * effective_s + 2.0 * norm_bs - 3.0 * action_norm + coop_bonus

    def step(self, actions):
        reward = self._reward(self.orig[self.t], actions)
        self.t += 1
        done = self.t >= self.T
        next_s = None if done else self._broadcast(self.latent[self.t])
        return next_s, reward, done


def train(latent, obs, adj, cfg, use_gat, device):
    num_agents  = cfg['num_agents']
    num_actions = cfg['num_actions']
    latent_dim  = latent.shape[1]
    adj_t = torch.tensor(adj, dtype=torch.float32).to(device)

    if use_gat:
        policy = MADRLGATNetwork(latent_dim, cfg['gat_hidden'], cfg['gat_out'],
                                 cfg['num_heads'], num_actions).to(device)
        target = MADRLGATNetwork(latent_dim, cfg['gat_hidden'], cfg['gat_out'],
                                 cfg['num_heads'], num_actions).to(device)
    else:
        policy = DQNNoGAT(latent_dim, hidden_dim=64, num_actions=num_actions).to(device)
        target = DQNNoGAT(latent_dim, hidden_dim=64, num_actions=num_actions).to(device)

    target.load_state_dict(policy.state_dict())
    target.eval()
    opt    = torch.optim.Adam(policy.parameters(), lr=cfg['madrl_lr'])
    buf    = ReplayBuffer(10000)
    env    = OriginalEnv(latent, obs, adj, num_agents, num_actions)

    eps = 1.0
    eps_end   = cfg.get('madrl_eps_end', 0.1)
    eps_decay = 0.97
    t_update  = cfg.get('madrl_target_update', 10)
    gamma     = 0.99
    batch_sz  = cfg['madrl_batch']
    tag = "GAT   " if use_gat else "noGAT "

    for ep in range(cfg['madrl_eps']):
        states = env.reset()
        total_r = 0.0
        done = False
        while not done:
            if random.random() < eps:
                actions = [random.randrange(num_actions) for _ in range(num_agents)]
            else:
                policy.eval()
                with torch.no_grad():
                    q = policy(torch.tensor(states, dtype=torch.float32).to(device), adj_t)
                    actions = q.argmax(dim=1).cpu().numpy().tolist()
                policy.train()

            ns, r, done = env.step(actions)
            total_r += r
            if ns is not None:
                buf.push(states, actions, r, ns, done)
            states = ns

            if len(buf) >= batch_sz:
                s_l,a_l,r_l,ns_l,d_l = zip(*buf.sample(batch_sz))
                s_b  = torch.tensor(np.array(s_l),  dtype=torch.float32).to(device)
                a_b  = torch.tensor(np.array(a_l),  dtype=torch.long).to(device)
                r_b  = torch.tensor(np.array(r_l),  dtype=torch.float32).to(device)
                ns_b = torch.tensor(np.array(ns_l), dtype=torch.float32).to(device)
                d_b  = torch.tensor(np.array(d_l),  dtype=torch.float32).to(device)
                q_cur = policy(s_b, adj_t)
                with torch.no_grad():
                    q_nxt = target(ns_b, adj_t)
                q_taken = q_cur.gather(2, a_b.unsqueeze(2)).squeeze(2)
                tgts    = r_b.unsqueeze(1) + gamma * q_nxt.max(2)[0] * (1 - d_b.unsqueeze(1))
                loss = F.mse_loss(q_taken, tgts.detach())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

        eps = max(eps_end, eps * eps_decay)
        if ep % t_update == 0:
            target.load_state_dict(policy.state_dict())

        if (ep + 1) % 20 == 0:
            print(f"  {tag} ep {ep+1:3d}  reward={total_r:.1f}  eps={eps:.3f}")

    # Evaluate
    eval_env = OriginalEnv(latent, obs, adj, num_agents, num_actions)
    rewards  = evaluate_madrl(policy, eval_env, adj,
                              num_episodes=20, num_agents=num_agents, device=device)
    return float(np.mean(rewards))


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cpu')
    cfg    = CFG['antwerp']

    # Global seed=42 — same as original run
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    print("Loading Antwerp data...")
    semantic_df, vae_X, vae_dim, obs_seq, q_labels, adj = setup_antwerp(cfg)

    print("Training VAE...")
    vae, _ = train_vae(vae_X, vae_dim,
                       latent_dim=cfg['vae_latent'], hidden_dim=cfg['vae_hidden'],
                       epochs=cfg['vae_epochs'], batch_size=cfg['vae_batch'],
                       verbose=False)
    latent = encode_dataset(vae, vae_X, device=device)

    # Subsample (same as main script)
    sub = cfg.get('madrl_subsample', None)
    if sub and len(latent) > sub:
        idx = np.random.RandomState(42).choice(len(latent), sub, replace=False)
        idx.sort()
        latent  = latent[idx]
        obs_seq = obs_seq[idx]

    print(f"\nTraining MADRL-GAT (100 eps, eps_end={cfg.get('madrl_eps_end',0.1)}, "
          f"target_update={cfg.get('madrl_target_update',10)}, seed=42, NO obs_noise)...")
    gat_r = train(latent, obs_seq, adj, cfg, use_gat=True,  device=device)

    print(f"\nTraining DQN-noGAT ...")
    nogat_r = train(latent, obs_seq, adj, cfg, use_gat=False, device=device)

    diff   = gat_r - nogat_r
    pct    = diff / nogat_r * 100 if nogat_r > 0 else 0
    print(f"\n{'='*50}")
    print(f"  MADRL-GAT  eval reward : {gat_r:.2f}")
    print(f"  DQN-noGAT  eval reward : {nogat_r:.2f}")
    print(f"  Difference             : {diff:+.2f}  ({pct:+.1f}%)")
    print(f"  Target (paper)         : GAT=2305.1, noGAT=2119.0, +8.8%")
    print(f"{'='*50}")
