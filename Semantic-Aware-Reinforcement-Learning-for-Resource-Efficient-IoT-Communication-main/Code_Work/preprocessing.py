# -*- coding: utf-8 -*-
"""
Preprocessing module — handles all three IoT datasets and prepares
inputs for the VAE, Digital Twin, and MADRL+GAT pipeline.

Datasets
--------
data/       : Sigfox Antwerp   (outdoor LoRa-like, 84 base-stations)
Data_2/     : LoRaWAN Italy    (agricultural sensors, 8 nodes)
Data_3/     : Indoor WiFi      (indoor positioning, 10 APs)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# ==========================================================================
# ── ORIGINAL BASELINE FUNCTIONS (kept for backward-compatibility) ──────────
# ==========================================================================

def preprocess_sigfox_data(sigfox_path, bs_mapping_path, output_path):
    """Original Sigfox preprocessing used by main.py."""
    sigfox_df = pd.read_csv(sigfox_path)
    sigfox_df.columns = [c.strip().replace("'", "") for c in sigfox_df.columns]
    sigfox_df["RX Time"] = pd.to_datetime(
        sigfox_df["RX Time"].str.replace("'", ""), errors='coerce')
    sigfox_df["hour"] = sigfox_df["RX Time"].dt.hour

    rssi_cols = [c for c in sigfox_df.columns if c.startswith("BS ")]
    rssi_data = sigfox_df[rssi_cols].replace(-200, np.nan)
    sigfox_df["mean_rssi"] = rssi_data.mean(axis=1)
    sigfox_df["num_active_bs"] = rssi_data.notna().sum(axis=1)

    bs_mapping = pd.read_csv(bs_mapping_path)
    bs_mapping.columns = [c.strip() for c in bs_mapping.columns]

    semantic_df = sigfox_df[["mean_rssi", "num_active_bs",
                              "Latitude", "Longitude", "hour"]]
    semantic_df.to_csv(output_path, index=False)
    print(f"Semantic features saved to '{output_path}'")
    print(semantic_df.head())
    return semantic_df


def prepare_classification_data(data):
    features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
    data = data.copy()
    data['label'] = ((data['mean_rssi'] > -110) &
                     (data['num_active_bs'] >= 3)).astype(int)
    print(f"Label distribution:\n{data['label'].value_counts()}")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data[features])
    return X_scaled, data['label'], features, scaler


def prepare_anomaly_data(data):
    features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data[features])
    return X_scaled, features, scaler


# ==========================================================================
# ── NEW: PER-DATASET RAW LOADERS ──────────────────────────────────────────
# ==========================================================================

def load_sigfox_raw(data_dir):
    """
    Load Sigfox Antwerp raw data.
    Returns
    -------
    rssi_matrix : np.ndarray [N, 84]  — NaN where station inactive
    semantic_df : pd.DataFrame        — 5-feature semantic table
    top_bs_cols : list[str]           — names of the 84 BS columns
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "sigfox_dataset_antwerp.csv")
    df.columns = [c.strip().replace("'", "") for c in df.columns]
    df["RX Time"] = pd.to_datetime(
        df["RX Time"].str.replace("'", ""), errors='coerce')
    df["hour"] = df["RX Time"].dt.hour

    bs_cols = [c for c in df.columns if c.startswith("BS ")]
    rssi_raw = df[bs_cols].replace(-200, np.nan).values.astype(np.float32)

    mean_rssi = np.nanmean(rssi_raw, axis=1)
    num_active = (~np.isnan(rssi_raw)).sum(axis=1).astype(np.float32)

    semantic_df = pd.DataFrame({
        'mean_rssi':    mean_rssi,
        'num_active_bs': num_active,
        'Latitude':     df['Latitude'].values,
        'Longitude':    df['Longitude'].values,
        'hour':         df['hour'].values,
    })
    semantic_df = semantic_df.dropna().reset_index(drop=True)

    # Keep rssi_matrix aligned with semantic_df (drop same NaN rows)
    valid_mask = ~np.isnan(mean_rssi)
    rssi_matrix = rssi_raw[valid_mask]

    return rssi_matrix, semantic_df, bs_cols


def load_lorawan_raw(data_dir):
    """
    Load LoRaWAN Italy sensor data.
    Returns
    -------
    feature_matrix : np.ndarray [N, 5]  — [rssi, snr, soil_temp, soil_hum, battery]
    semantic_df    : pd.DataFrame       — 5+3 feature table (incl. soil & battery)
    node_coords    : dict {nodeid: (lat, lon)}
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "sensors_data.csv", sep=';')

    # Parse node coordinates
    node_coords = {}
    coord_path = data_dir / "nodes_coordinates.txt"
    if coord_path.exists():
        with open(coord_path) as f:
            lines = f.readlines()
        for line in lines[2:]:   # skip header lines
            parts = line.split()
            if len(parts) >= 3:
                try:
                    nid = parts[0]
                    lat = float(parts[2])
                    lon = float(parts[3])
                    node_coords[nid] = (lat, lon)
                except (ValueError, IndexError):
                    pass

    features = ['gtw_rssi', 'gtw_snr', 'soil_temp', 'soil_hum', 'battery']
    df_clean = df[features].dropna().reset_index(drop=True)
    feature_matrix = df_clean.values.astype(np.float32)

    df_full = df.copy()
    df_full['mean_rssi'] = df_full['gtw_rssi']
    df_full['num_active_bs'] = 1
    df_full['hour'] = pd.to_datetime(df_full['timestamp'], unit='s').dt.hour

    semantic_df = df_full[['mean_rssi', 'num_active_bs', 'gtw_rssi', 'gtw_snr',
                            'soil_temp', 'soil_hum', 'battery', 'hour']].dropna()
    semantic_df = semantic_df.reset_index(drop=True)

    return feature_matrix, semantic_df, node_coords


def load_indoor_raw(data_dir):
    """
    Load Indoor WiFi RSSI data.
    Tries to read indoor_raw_rssi.ods (10-AP matrix);
    falls back to semantic_features_indoor_ods.csv.
    Returns
    -------
    rssi_matrix : np.ndarray [N, K]  — raw per-AP RSSI values
    semantic_df : pd.DataFrame       — 5-feature semantic table
    pos_coords  : np.ndarray [P, 2]  — floor-plan (x, y) of P positions
    """
    data_dir = Path(data_dir)

    # Load floor-plan coordinates
    pos_coords = []
    coord_path = data_dir / "indoor_pos_coords.txt"
    if coord_path.exists():
        with open(coord_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        pos_coords.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass
    pos_coords = np.array(pos_coords) if pos_coords else np.zeros((11, 2))

    # Try to read ODS raw RSSI
    ods_path = data_dir / "indoor_raw_rssi.ods"
    rssi_matrix = None
    if ods_path.exists():
        try:
            raw_df = pd.read_excel(ods_path, engine='odf', header=0)
            # Drop non-numeric columns (position labels etc.)
            numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
            rssi_matrix = raw_df[numeric_cols].values.astype(np.float32)
        except Exception as e:
            print(f"  [Indoor] Could not read ODS ({e}); using CSV fallback.")

    # Fallback: semantic CSV
    csv_path = data_dir / "semantic_features_indoor_ods.csv"
    semantic_df = pd.read_csv(csv_path)

    if rssi_matrix is None or rssi_matrix.shape[1] == 0:
        # Fallback: tile mean_rssi into 5 columns as proxy
        rssi_matrix = semantic_df[['mean_rssi']].values.astype(np.float32)
        rssi_matrix = np.tile(rssi_matrix, (1, 5))

    return rssi_matrix, semantic_df, pos_coords


# ==========================================================================
# ── VAE INPUT PREPARATION ─────────────────────────────────────────────────
# ==========================================================================

def prepare_vae_input_sigfox(rssi_matrix):
    """
    Normalize 84-column RSSI matrix to [0,1] for VAE.
    -200 (inactive) → 0, active RSSI range ~ [-145, -60] dBm.
    """
    RSSI_MIN, RSSI_MAX = -145.0, -60.0
    filled = np.where(np.isnan(rssi_matrix), RSSI_MIN, rssi_matrix)
    filled = np.clip(filled, RSSI_MIN, RSSI_MAX)
    normalized = (filled - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)
    scaler_params = {'min': RSSI_MIN, 'max': RSSI_MAX}
    return normalized.astype(np.float32), scaler_params


def prepare_vae_input_lorawan(feature_matrix):
    """MinMax-normalize LoRaWAN 5-feature matrix to [0,1]."""
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(feature_matrix).astype(np.float32)
    return normalized, scaler


def prepare_vae_input_indoor(rssi_matrix):
    """MinMax-normalize indoor RSSI matrix to [0,1]."""
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(rssi_matrix).astype(np.float32)
    return normalized, scaler


# ==========================================================================
# ── SEMANTIC OBSERVATION SEQUENCE (for Digital Twin + MADRL) ──────────────
# ==========================================================================

def get_obs_sequence(semantic_df):
    """
    Extract [mean_rssi, num_active_bs] sequence for EKF observations.
    Returns np.ndarray [N, 2].
    """
    return semantic_df[['mean_rssi', 'num_active_bs']].values.astype(np.float32)


def get_quality_labels(semantic_df,
                       rssi_col='mean_rssi', bs_col='num_active_bs',
                       rssi_thresh=-110.0, bs_thresh=3):
    """
    Binary quality label: 1 = good connection.
    Falls back to median-RSSI split if all labels end up same class
    (e.g. LoRaWAN where num_active_bs is always 1).
    """
    rssi = semantic_df[rssi_col].values
    bs   = semantic_df[bs_col].values
    labels = ((rssi > rssi_thresh) & (bs >= bs_thresh)).astype(int)

    if labels.sum() == 0 or labels.sum() == len(labels):
        median_rssi = float(np.median(rssi))
        labels = (rssi > median_rssi).astype(int)
        print(f"  [Label] Single-class detected -> median-RSSI split "
              f"(threshold={median_rssi:.1f} dBm)")
    return labels


def get_anomaly_labels_from_if(semantic_df, contamination=0.05):
    """Generate pseudo-anomaly labels using Isolation Forest (for evaluation)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import MinMaxScaler
    features = [c for c in ['mean_rssi', 'num_active_bs', 'Latitude', 'Longitude', 'hour']
                if c in semantic_df.columns]
    X = MinMaxScaler().fit_transform(semantic_df[features].values)
    iso = IsolationForest(n_estimators=100, contamination=contamination,
                          random_state=42)
    preds = iso.fit_predict(X)
    return (preds == -1).astype(int)  # 1 = anomaly


# ==========================================================================
# ── ORIGINAL UTILS (kept for main.py compatibility) ───────────────────────
# ==========================================================================

def _legacy_prepare_classification_data(data):
    return prepare_classification_data(data)


def _legacy_prepare_anomaly_data(data):
    return prepare_anomaly_data(data)
