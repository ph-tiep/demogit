# -*- coding: utf-8 -*-
"""
Multi-Agent Deep Reinforcement Learning with Graph Attention Network (MADRL+GAT).
Vectorized batch processing -- all B samples processed in one forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


# ---------------------------------------------------------------------------
# Graph Attention Network (batched)
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """Graph attention layer supporting both [N,D] and [B,N,D] inputs."""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(0.2)
        self.drop  = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        h   : [N, D]  or  [B, N, D]
        adj : [N, N]
        Returns same leading dims with out_dim as last.
        """
        batched = h.dim() == 3
        if not batched:
            h = h.unsqueeze(0)          # [1, N, D]
        B, N, _ = h.shape

        Wh = self.W(h)                                          # [B, N, out]
        Wh = Wh.view(B, N, self.num_heads, self.head_dim)       # [B, N, H, D]

        hi = Wh.unsqueeze(2).expand(-1, -1, N, -1, -1)         # [B, N, N, H, D]
        hj = Wh.unsqueeze(1).expand(-1, N, -1, -1, -1)         # [B, N, N, H, D]
        pair = torch.cat([hi, hj], dim=-1)                      # [B, N, N, H, 2D]

        e = (pair * self.a).sum(-1)                             # [B, N, N, H]
        e = self.leaky(e)

        mask = (adj == 0).unsqueeze(0).unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask, float('-inf'))
        alpha = F.softmax(e, dim=2)                             # [B, N, N, H]
        alpha = self.drop(alpha)

        out = (alpha.unsqueeze(-1) * hj).sum(2)                # [B, N, H, D]
        out = out.reshape(B, N, -1)                             # [B, N, out]

        return out.squeeze(0) if not batched else out


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.gat1  = GATLayer(in_dim, hidden_dim, num_heads, dropout)
        self.gat2  = GATLayer(hidden_dim, out_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        # x: [N,D] or [B,N,D]
        h = F.elu(self.norm1(self.gat1(x, adj)))
        return self.norm2(self.gat2(h, adj))


# ---------------------------------------------------------------------------
# MADRL Policy / Q-network
# ---------------------------------------------------------------------------

class MADRLGATNetwork(nn.Module):
    """Shared Q-network. Accepts [N,D] or [B,N,D]."""

    def __init__(self, agent_obs_dim, gat_hidden=32, gat_out=16,
                 num_heads=4, num_actions=5, dropout=0.1):
        super().__init__()
        self.gat = GATEncoder(agent_obs_dim, gat_hidden, gat_out,
                              num_heads, dropout)
        self.policy = nn.Sequential(
            nn.Linear(agent_obs_dim + gat_out, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, states, adj):
        """states: [N,D] or [B,N,D] -> Q-values: same leading dims + num_actions"""
        gat_feat = self.gat(states, adj)
        combined = torch.cat([states, gat_feat], dim=-1)
        return self.policy(combined)


class DQNNoGAT(nn.Module):
    """MLP Q-network without graph attention — ablation baseline."""

    def __init__(self, agent_obs_dim, hidden_dim=64, num_actions=5):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(agent_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, states, adj=None):
        return self.policy(states)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MADRLEnvironment:
    def __init__(self, latent_states, original_obs, adj_matrix,
                 num_agents=10, num_actions=5, obs_noise_std=0.15, seed=42):
        self.latent  = latent_states.astype(np.float32)
        self.orig    = original_obs.astype(np.float32)
        self.adj     = adj_matrix
        self.num_agents  = num_agents
        self.num_actions = num_actions
        self.T = len(latent_states)
        self.t = 0

        # Fixed per-agent observation offsets so agents have heterogeneous
        # local views of the channel — required for GAT attention to be meaningful
        rng = np.random.RandomState(seed)
        self.agent_biases = rng.normal(
            0, obs_noise_std, (num_agents, latent_states.shape[1])
        ).astype(np.float32)

    def _broadcast(self, g):
        return np.tile(g, (self.num_agents, 1)) + self.agent_biases   # [N_agents, latent_dim]

    def reset(self):
        self.t = 0
        return self._broadcast(self.latent[0])

    @staticmethod
    def _reward(orig_row, actions):
        """
        Per-agent reward matching Eq. (2) in paper:
          r = 5*tilde_s + 2*n_bar - 3*a_bar + max(0, 1 - sigma_a/2)
        tilde_s = s_bar + a_bar*(1-s_bar): action boosts effective signal
        only when channel is weak (low s_bar), creating state-dependent
        optimal actions that require inter-agent coordination via GAT.
        Crossover: boosting beneficial when s_bar < 0.4 (5*(1-s) > 3).
        """
        mean_rssi = float(orig_row[0])
        num_bs    = float(orig_row[1])

        RSSI_MIN, RSSI_MAX = -145.0, -50.0
        norm_rssi = float(np.clip((mean_rssi - RSSI_MIN) / (RSSI_MAX - RSSI_MIN),
                                  0.0, 1.0))

        action_norm = float(np.mean(actions)) / 4.0                     # a_bar in [0,1]
        effective_s = norm_rssi + action_norm * (1.0 - norm_rssi)       # tilde_s
        norm_bs     = min(num_bs / 20.0, 1.0)                           # n_bar
        coop_bonus  = max(0.0, 1.0 - float(np.std(actions)) / 2.0)     # [0,1]

        return 5.0 * effective_s + 2.0 * norm_bs - 3.0 * action_norm + coop_bonus

    def step(self, actions):
        reward = self._reward(self.orig[self.t], actions)
        self.t += 1
        done = self.t >= self.T
        next_s = None if done else self._broadcast(self.latent[self.t])
        return next_s, reward, done


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


# ---------------------------------------------------------------------------
# Training (fully vectorized)
# ---------------------------------------------------------------------------

def train_madrl(latent_states, original_obs, adj_matrix,
                num_agents=10, num_actions=5,
                num_episodes=50, lr=1e-3,
                gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.97,
                batch_size=32, buffer_capacity=10000, target_update=5,
                gat_hidden=32, gat_out=16, num_heads=4,
                use_gat=True,
                device=None, verbose=True):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_dim = latent_states.shape[1]
    adj_t = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

    if use_gat:
        policy_net = MADRLGATNetwork(latent_dim, gat_hidden, gat_out,
                                     num_heads, num_actions).to(device)
        target_net = MADRLGATNetwork(latent_dim, gat_hidden, gat_out,
                                     num_heads, num_actions).to(device)
    else:
        policy_net = DQNNoGAT(latent_dim, hidden_dim=64, num_actions=num_actions).to(device)
        target_net = DQNNoGAT(latent_dim, hidden_dim=64, num_actions=num_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    env = MADRLEnvironment(latent_states, original_obs, adj_matrix,
                           num_agents, num_actions)

    eps = eps_start
    ep_rewards, ep_losses = [], []

    tag = "MADRL+GAT" if use_gat else "DQN-noGAT"
    print(f"  {tag}: latent={latent_dim}, agents={num_agents}, "
          f"actions={num_actions}, episodes={num_episodes}, device={device}")

    for episode in range(num_episodes):
        states = env.reset()
        total_reward = 0.0
        step_losses  = []
        done = False

        while not done:
            # Action selection
            if random.random() < eps:
                actions = [random.randrange(num_actions) for _ in range(num_agents)]
            else:
                policy_net.eval()
                with torch.no_grad():
                    s_t = torch.tensor(states, dtype=torch.float32).to(device)
                    q   = policy_net(s_t, adj_t)         # [N, A]
                    actions = q.argmax(dim=1).cpu().numpy().tolist()
                policy_net.train()

            next_states, reward, done = env.step(actions)
            total_reward += reward
            if next_states is not None:
                buffer.push(states, actions, reward, next_states, done)
            states = next_states

            # ── Vectorized optimization step ─────────────────────────────
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                s_l, a_l, r_l, ns_l, d_l = zip(*batch)

                # [B, N, D]
                s_b  = torch.tensor(np.array(s_l),  dtype=torch.float32).to(device)
                a_b  = torch.tensor(np.array(a_l),  dtype=torch.long).to(device)   # [B, N]
                r_b  = torch.tensor(np.array(r_l),  dtype=torch.float32).to(device) # [B]
                ns_b = torch.tensor(np.array(ns_l), dtype=torch.float32).to(device) # [B, N, D]
                d_b  = torch.tensor(np.array(d_l),  dtype=torch.float32).to(device) # [B]

                # Single forward pass for entire batch
                q_cur = policy_net(s_b, adj_t)           # [B, N, A]
                with torch.no_grad():
                    q_nxt = target_net(ns_b, adj_t)      # [B, N, A]

                q_taken  = q_cur.gather(2, a_b.unsqueeze(2)).squeeze(2)   # [B, N]
                q_nxt_mx = q_nxt.max(2)[0]                                 # [B, N]
                targets  = (r_b.unsqueeze(1)
                            + gamma * q_nxt_mx * (1 - d_b.unsqueeze(1)))   # [B, N]

                loss = F.mse_loss(q_taken, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                step_losses.append(loss.item())

        eps = max(eps_end, eps * eps_decay)
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        avg_loss = float(np.mean(step_losses)) if step_losses else 0.0
        ep_rewards.append(total_reward)
        ep_losses.append(avg_loss)

        if verbose and (episode + 1) % 10 == 0:
            print(f"  Ep [{episode+1}/{num_episodes}]  "
                  f"Reward={total_reward:.1f}  Loss={avg_loss:.4f}  eps={eps:.3f}")

    print(f"  {tag} done.")
    return policy_net, target_net, ep_rewards, ep_losses


# ---------------------------------------------------------------------------
# Adjacency builders
# ---------------------------------------------------------------------------

def adj_from_correlation(rssi_matrix, top_k=10, threshold=0.3):
    activity = (~np.isnan(rssi_matrix)).sum(axis=0)
    top_idx  = np.argsort(activity)[::-1][:top_k]
    sub      = rssi_matrix[:, top_idx]
    sub      = np.where(np.isnan(sub), np.nanmean(sub, axis=0), sub)
    corr     = np.corrcoef(sub.T)
    np.fill_diagonal(corr, 0)
    adj = (corr > threshold).astype(float) + np.eye(top_k)
    return adj.astype(np.float32), top_idx


def adj_from_distances(coords, sigma=150.0, threshold=0.1):
    n = len(coords)
    coords = np.array(coords, dtype=float)
    adj = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.linalg.norm(coords[i] - coords[j])
                w = np.exp(-(d**2) / (2 * sigma**2))
                adj[i, j] = float(w > threshold)
    return adj


def adj_fully_connected(n):
    return np.ones((n, n), dtype=np.float32)
