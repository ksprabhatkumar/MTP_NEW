import os
import torch
from torch.utils.data import Dataset, DataLoader

class HARDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataset_config(dataset_name):
    """Returns dataset-specific configurations (Nodes, Features, Classes)."""
    configs = {
        "dsads": {"num_nodes": 5, "in_features": 9, "num_classes": 19, "total_subjects": 8},
        "opportunity": {"num_nodes": 5, "in_features": 9, "num_classes": 18, "total_subjects": 4},
        "realdisp": {"num_nodes": 9, "in_features": 9, "num_classes": 33, "total_subjects": 17}
    }
    return configs[dataset_name.lower()]

def get_loso_dataloaders(dataset_name, test_subject_id, batch_size=128):
    """Dynamically loads datasets and creates the n-2 (Train), 1 (Internal Val), and 1 (Test) splits."""
    
    base_path = f"dataset/processed/{dataset_name.lower()}"
    
    if not os.path.exists(f"{base_path}_x.pt"):
        raise FileNotFoundError(f"Could not find processed tensors for {dataset_name}! Ensure they are in dataset/processed/")

    x = torch.load(f"{base_path}_x.pt").float() # Ensure float32 for normalization
    y = torch.load(f"{base_path}_y.pt").long()
    p = torch.load(f"{base_path}_p.pt").long()

    config = get_dataset_config(dataset_name)
    total_subjects = config["total_subjects"]

    # 🌟 1. The Validation Subject (1 subject)
    test_mask = (p == test_subject_id)

    # 🌟 2. The Internal Training Val Subject (1 subject from the n-1 pool)
    internal_val_id = test_subject_id + 1 if test_subject_id < total_subjects else 1
    internal_val_mask = (p == internal_val_id)

    # 🌟 3. The Actual Training Pool (n-2 subjects)
    train_mask = ~(test_mask | internal_val_mask)

    # Split data
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[internal_val_mask], y[internal_val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    # ==========================================================
    # 🌟 FIX #2: Z-SCORE NORMALIZATION (Solves the MSE Explosion)
    # ==========================================================
    # We calculate the mean and standard deviation strictly on the TRAINING set
    # to prevent "data leakage", then apply it to all three sets.
    mean = x_train.mean(dim=(0, 1, 2), keepdim=True)
    std = x_train.std(dim=(0, 1, 2), keepdim=True) + 1e-8 # 1e-8 prevents division by zero
    
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    # Create datasets
    train_dataset = HARDataset(x_train, y_train)
    internal_val_dataset = HARDataset(x_val, y_val)
    test_dataset = HARDataset(x_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    internal_val_loader = DataLoader(internal_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, internal_val_loader, test_loader