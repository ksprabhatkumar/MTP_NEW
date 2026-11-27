import os
import json
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter

# --- OPPORTUNITY CONSTANTS ---
OPP_LABELS = {
    0: 0, 406516: 1, 406517: 2, 404516: 3, 404517: 4, 406520: 5, 404520: 6,
    406505: 7, 404505: 8, 406519: 9, 404519: 10, 406511: 11, 404511: 12,
    406508: 13, 404508: 14, 408512: 15, 407521: 16, 405506: 17
}
FEATURE_COLS_OPP = list(range(37, 46)) + list(range(50, 59)) + list(range(63, 72)) + list(range(76, 85)) + list(range(89, 98))
LABEL_COL_OPP = 249

# --- REALDISP CONSTANTS ---
# Acceleration and gyro only: six channels per IMU, in the documented nine-node order.
REALDISP_NODE_STARTS = (28, 41, 15, 54, 2, 93, 80, 106, 67)
NODE_COLS_REALDISP = tuple(column for start in REALDISP_NODE_STARTS for column in range(start, start + 6))
LABEL_COL_REALDISP = 119

def save_tensors(name, data, labels, persons):
    os.makedirs("dataset/processed", exist_ok=True)
    X_tensor = torch.tensor(np.array(data), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(labels), dtype=torch.long)
    P_tensor = torch.tensor(np.array(persons), dtype=torch.long)

    torch.save(X_tensor, f"dataset/processed/{name}_x.pt")
    torch.save(Y_tensor, f"dataset/processed/{name}_y.pt")
    torch.save(P_tensor, f"dataset/processed/{name}_p.pt")
    return
    print(f"✅ Saved {name.upper()}: X {X_tensor.shape}, Y {Y_tensor.shape}, P {P_tensor.shape}\n")

def process_dsads():
    print("Processing DSADS dataset (OPTIMIZED - Z-Score Normalization)...")
    base_path = os.path.join("har_data", "DSADS", "data")
    if not os.path.exists(base_path): return

    all_data, all_labels, all_persons = [], [], []
    for activity_id in tqdm(range(1, 20), desc="Processing DSADS"):
        for person_id in range(1, 9):
            for segment_id in range(1, 61):
                file_path = os.path.join(base_path, f"a{activity_id:02d}", f"p{person_id}", f"s{segment_id:02d}.txt")
                try:
                    data_matrix = np.loadtxt(file_path, delimiter=',')
                    all_data.append(data_matrix.reshape(125, 5, 9))
                    all_labels.append(activity_id - 1)
                    all_persons.append(person_id)
                except Exception:
                    pass

    if len(all_data) == 0: return
    X_np = np.array(all_data, dtype=np.float32)
    
    means = np.mean(X_np, axis=(0, 1), keepdims=True)
    stds = np.std(X_np, axis=(0, 1), keepdims=True)
    stds[stds == 0] = 1.0 
    X_normalized = (X_np - means) / stds
    print(f"Data Normalized! Global Mean: {np.mean(X_normalized):.4f}, Std: {np.std(X_normalized):.4f}")
    
    save_tensors("dsads", X_normalized, all_labels, all_persons)

def process_opportunity():
    print("Processing OPPORTUNITY dataset (OPTIMIZED - Z-Score Normalization)...")
    base_path = os.path.join("har_data", "OPPORTUNITY", "dataset")
    if not os.path.exists(base_path): return

    all_windows, all_labels, all_persons = [], [], []
    window_size, step_size = 24, 12

    for subject_id in range(1, 5):
        files = [f for f in os.listdir(base_path) if f.startswith(f"S{subject_id}-") and f.endswith(".dat")]
        for file_name in tqdm(files, desc=f"Subject {subject_id}"):
            file_path = os.path.join(base_path, file_name)
            
            df = pd.read_csv(file_path, sep=r'\s+', header=None)
            
            features = df[FEATURE_COLS_OPP].copy()
            labels = df[LABEL_COL_OPP].copy()
            
            features = features.interpolate(method='linear', limit_direction='both').fillna(0).values
            labels = labels.values
            mapped_labels = np.array([OPP_LABELS.get(lbl, 0) for lbl in labels])
            
            for start in range(0, len(features) - window_size, step_size):
                window_data = features[start : start + window_size]
                window_labels = mapped_labels[start : start + window_size]
                most_common_label = Counter(window_labels).most_common(1)[0][0]
                
                all_windows.append(window_data.reshape(window_size, 5, 9))
                all_labels.append(most_common_label)
                all_persons.append(subject_id)

    X_np = np.array(all_windows, dtype=np.float32)
    
    means = np.mean(X_np, axis=(0, 1), keepdims=True)
    stds = np.std(X_np, axis=(0, 1), keepdims=True)
    stds[stds == 0] = 1.0 
    X_normalized = (X_np - means) / stds
    print(f"Data Normalized! Global Mean: {np.mean(X_normalized):.4f}, Std: {np.std(X_normalized):.4f}")
    
    save_tensors("opportunity", X_normalized, all_labels, all_persons)

def _realdisp_file_metadata(filename):
    match = re.match(r"subject(?P<subject>\d+)_(?P<scenario>ideal|self|mutual|induced)(?P<recording>\d*)\.log$", filename, re.I)
    if not match:
        return None
    scenario = match.group("scenario").lower()
    # Historical project files call the recorded induced condition "mutual".
    if scenario == "mutual":
        scenario = "induced"
    return int(match.group("subject")), scenario, match.group("recording") or "0"


def process_realdisp(subjects=None):
    """Create unnormalised O1 windows and an auditable scenario manifest."""
    print("Processing REALDISP for Objective 1 (raw acc/gyro windows; no global normalisation)...")
    base_path = os.path.join("har_data", "REALDISP")
    if not os.path.exists(base_path):
        print(f"REALDISP directory not found: {base_path}")
        return
    all_windows, all_labels, all_persons, manifest = [], [], [], []
    for filename in sorted(os.listdir(base_path)):
        parsed = _realdisp_file_metadata(filename)
        if parsed is None:
            continue
        subject_id, scenario, recording_id = parsed
        if subjects is not None and subject_id not in subjects:
            continue
        file_path = os.path.join(base_path, filename)
        df = pd.read_csv(file_path, sep='\t|,|\s+', header=None, engine='python')
        if df.shape[1] <= LABEL_COL_REALDISP:
            raise ValueError(f"{filename} has {df.shape[1]} columns; expected at least 120")
        features = df.iloc[:, list(NODE_COLS_REALDISP)].interpolate(method='linear', limit_direction='both').fillna(0).to_numpy(dtype=np.float32)
        labels = df.iloc[:, LABEL_COL_REALDISP].to_numpy()
        for start in range(0, len(features) - 50 + 1, 25):
            label = Counter(labels[start:start + 50]).most_common(1)[0][0]
            if int(label) == 0:
                continue
            all_windows.append(features[start:start + 50].reshape(50, 9, 6))
            all_labels.append(int(label) - 1)
            all_persons.append(subject_id)
            manifest.append({"subject": subject_id, "scenario": scenario, "recording": recording_id, "source_file": filename, "start": start, "stop": start + 50, "label": int(label) - 1, "availability": [1] * 9})
    if not all_windows:
        raise RuntimeError("No REALDISP windows were created; check file names and label column.")
    save_tensors("realdisp_o1", all_windows, all_labels, all_persons)
    manifest_path = "dataset/processed/realdisp_o1_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    induced_subjects = sorted({row["subject"] for row in manifest if row["scenario"] == "induced"})
    unexpected = set(induced_subjects) - {2, 5, 15}
    if unexpected:
        print(f"Warning: induced recordings found for undocumented subjects: {sorted(unexpected)}")
    print(f"Induced-placement coverage: {induced_subjects or 'none'}")
    print(f"Saved {len(manifest)} raw O1 windows and manifest to {manifest_path}")

if __name__ == "__main__":
    print("🚀 Starting Data Processing (OPTIMIZED MODE)...\n")
    process_dsads()
    process_opportunity()
    process_realdisp()
    print("🎉 All datasets processed & Normalized!")
