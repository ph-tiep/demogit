import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from models import DQN, ReplayMemory, SimpleEnv, create_random_forest_classifier, create_isolation_forest
from sklearn.model_selection import train_test_split
from digital_twin import DigitalTwinEKF, EKFConfig, train_digital_twin

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000
LR = 1e-3

def reward_function(state, action):
    """Simple reward function"""
    mean_rssi = state[0]
    num_active = state[1]
    return mean_rssi * 10 + num_active * 5 - action

def select_action(state, policy_net, num_actions, eps, device):
    if random.random() < eps:
        return random.randrange(num_actions)
    else:
        with torch.no_grad():
            state_v = torch.tensor(state).unsqueeze(0).to(device)
            return policy_net(state_v).argmax().item()

def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < BATCH_SIZE:
        return 0.0
    
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.tensor(np.array(batch[0])).to(device)
    action_batch = torch.tensor(batch[1]).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch[2]).to(device)
    next_state_batch = torch.tensor(np.array([s for s in batch[3] if s is not None])).to(device)

    non_final_mask = torch.tensor(tuple(s is not None for s in batch[3]), device=device, dtype=torch.bool)
    non_final_next_states = next_state_batch

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = reward_batch + (GAMMA * next_state_values)
    loss = nn.functional.mse_loss(state_action_values.squeeze(), expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train_dqn(states, num_episodes=2):
    """Train DQN agent"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = 5
    
    # Auto-detect input dimension from states
    input_dim = states.shape[1] if len(states.shape) > 1 else len(states[0])
    print(f"DQN input dimension: {input_dim}")
    
    # Initialize networks
    policy_net = DQN(input_dim=input_dim, output_dim=num_actions).to(device)
    target_net = DQN(input_dim=input_dim, output_dim=num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_CAPACITY)
    env = SimpleEnv(states, reward_function)
    
    eps = EPS_START
    losses = []
    total_rewards = []
    
    print("Starting DQN training...")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(state, policy_net, num_actions, eps, device)
            next_state, reward, done = env.step(action)
            memory.push((state, action, reward, next_state))
            state = next_state
            total_reward += reward
            
            loss = optimize_model(policy_net, target_net, optimizer, memory, device)
            if loss > 0:
                losses.append(loss)
        
        eps = max(EPS_END, eps * EPS_DECAY)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        total_rewards.append(total_reward)
        print(f"Episode {episode+1}/{num_episodes} - Total reward: {total_reward:.2f} - Epsilon: {eps:.2f}")
    
    print("DQN training completed.")
    return policy_net, target_net, losses, total_rewards

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier"""
    print("Training Random Forest classifier...")
    clf = create_random_forest_classifier()
    clf.fit(X_train, y_train)
    print("Random Forest training completed.")
    return clf

def train_isolation_forest(X_scaled):
    """Train Isolation Forest for anomaly detection"""
    print("Training Isolation Forest for anomaly detection...")
    iso_forest = create_isolation_forest()
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    print("Isolation Forest training completed.")
    return iso_forest, anomaly_labels

def train_digital_twin_on_data(X_train, feature_names, epochs=1):
    """
    Train Digital Twin (EKF) on training data.
    """
    from digital_twin import DigitalTwinEKF, EKFConfig

    print("Initializing Digital Twin (EKF)...")

    config = EKFConfig(
        state_dim=3,
        obs_dim=X_train.shape[1],
        process_noise=0.01,
        measurement_noise=0.05,
        innovation_threshold=3.5,
        min_innovation_history=50
    )

    dt = DigitalTwinEKF(config)

    burn_in = min(100, len(X_train) // 10)
    print(f"Digital Twin burn-in on {burn_in} samples...")

    for i in range(burn_in):
        obs_params = X_train[i]
        dt.step(obs_params)

    print("Digital Twin initialization completed.")
    print(f"  Initial state quality: {np.clip(dt.x[0], 0.0, 1.0):.4f}")
    print(f"  RSSI component: {np.clip(dt.x[1], 0.0, 1.0):.4f}")
    print(f"  BS component: {np.clip(dt.x[2], 0.0, 1.0):.4f}")

    return dt, config

def predict_with_digital_twin(dt, X_test, feature_names):
    """
    Use trained Digital Twin to predict connection quality and detect anomalies.
    """
    print("Running Digital Twin predictions on test set...")

    qualities = np.zeros(len(X_test))
    anomaly_scores = np.zeros(len(X_test))
    is_anomalies = np.zeros(len(X_test), dtype=bool)

    for i in range(len(X_test)):
        obs = X_test[i]
        _, quality, score, is_anom = dt.step(obs)
        qualities[i] = quality
        anomaly_scores[i] = score
        is_anomalies[i] = is_anom

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(X_test)} samples...")

    print(f"Digital Twin prediction completed.")
    print(f"  Mean quality: {np.mean(qualities):.4f}")
    print(f"  Anomaly rate: {np.mean(is_anomalies):.4f}")

    return qualities, anomaly_scores, is_anomalies
