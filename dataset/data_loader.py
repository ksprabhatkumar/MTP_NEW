import os
import json
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


class Objective1Dataset(Dataset):
    """REALDISP windows with their explicit sender-availability masks."""
    def __init__(self, x, y, availability):
        self.x, self.y, self.availability = x, y, availability

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.availability[idx]

def get_dataset_config(dataset_name):
    configs = {
        "dsads": {"num_nodes": 5, "in_features": 9, "num_classes": 19, "total_subjects": 8},
        "opportunity": {"num_nodes": 5, "in_features": 9, "num_classes": 18, "total_subjects": 4},
        "realdisp": {"num_nodes": 9, "in_features": 6, "num_classes": 33, "total_subjects": 17}
    }
    return configs[dataset_name.lower()]

def get_loso_dataloaders(dataset_name, test_subject_id, batch_size=128, test_on_mutual=False):
    base_path = f"dataset/processed/{dataset_name.lower()}"
    
    if not os.path.exists(f"{base_path}_x.pt"):
        raise FileNotFoundError(f"Processed tensors not found for {dataset_name}!")

    # 🌟 Added weights_only=True to silence warnings
    x = torch.load(f"{base_path}_x.pt", weights_only=True).float() 
    y = torch.load(f"{base_path}_y.pt", weights_only=True).long()
    p = torch.load(f"{base_path}_p.pt", weights_only=True).long()

    config = get_dataset_config(dataset_name)
    total_subjects = config["total_subjects"]

    test_mask = (p == test_subject_id)
    internal_val_id = test_subject_id + 1 if test_subject_id < total_subjects else 1
    internal_val_mask = (p == internal_val_id)
    train_mask = ~(test_mask | internal_val_mask)

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[internal_val_mask], y[internal_val_mask]
    
    if test_on_mutual and dataset_name.lower() == "realdisp":
        mut_path = "dataset/processed/realdisp_mutual"
        if not os.path.exists(f"{mut_path}_x.pt"):
            return None, None, None
            
        # 🌟 Added weights_only=True here as well
        x_mut = torch.load(f"{mut_path}_x.pt", weights_only=True).float()
        y_mut = torch.load(f"{mut_path}_y.pt", weights_only=True).long()
        p_mut = torch.load(f"{mut_path}_p.pt", weights_only=True).long()
        
        test_mask_mut = (p_mut == test_subject_id)
        x_test, y_test = x_mut[test_mask_mut], y_mut[test_mask_mut]
    else:
        x_test, y_test = x[test_mask], y[test_mask]

    mean = x_train.mean(dim=(0, 1, 2), keepdim=True)
    std = x_train.std(dim=(0, 1, 2), keepdim=True) + 1e-8 
    
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    if len(x_test) > 0:
        x_test = (x_test - mean) / std

    train_loader = DataLoader(HARDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    internal_val_loader = DataLoader(HARDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(HARDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader, internal_val_loader, test_loader


def get_objective1_loso_loaders(test_subject_id, batch_size=64, num_workers=0):
    """Return leakage-free nested-LOSO loaders plus recorded-placement test data.

    The validation subject rotates deterministically to the next available ideal subject.
    Normalisation is fitted only on the ideal-placement training partition.
    """
    base = "dataset/processed/realdisp_o1"
    manifest_path = f"{base}_manifest.json"
    required = [f"{base}_{suffix}.pt" for suffix in ("x", "y", "p")] + [manifest_path]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Objective 1 processed data missing: " + ", ".join(missing))
    x = torch.load(f"{base}_x.pt", weights_only=True).float()
    y = torch.load(f"{base}_y.pt", weights_only=True).long()
    subjects = torch.load(f"{base}_p.pt", weights_only=True).long()
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if len(manifest) != len(x):
        raise ValueError("REALDISP manifest and tensor row counts differ")
    scenarios = [row["scenario"] for row in manifest]
    available = torch.tensor([row.get("availability", [1] * 9) for row in manifest], dtype=torch.bool)
    ideal = torch.tensor([scenario == "ideal" for scenario in scenarios], dtype=torch.bool)
    test_mask = ideal & (subjects == test_subject_id)
    candidates = [subject for subject in range(1, 18) if subject != test_subject_id and bool((ideal & (subjects == subject)).any())]
    if not candidates:
        raise ValueError(f"No ideal validation subject is available for held-out subject {test_subject_id}")
    validation_subject = next((subject for subject in candidates if subject > test_subject_id), candidates[0])
    validation_mask = ideal & (subjects == validation_subject)
    train_mask = ideal & ~(test_mask | validation_mask)
    if not bool(train_mask.any()) or not bool(validation_mask.any()) or not bool(test_mask.any()):
        raise ValueError(f"Invalid O1 split for held-out subject {test_subject_id}")
    mean = x[train_mask].mean(dim=(0, 1, 2), keepdim=True)
    std = x[train_mask].std(dim=(0, 1, 2), keepdim=True).clamp_min(1e-8)

    def make_loader(mask, shuffle=False):
        normalised = (x[mask] - mean) / std
        return DataLoader(Objective1Dataset(normalised, y[mask], available[mask]), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    induced_mask = torch.tensor([scenario == "induced" for scenario in scenarios], dtype=torch.bool) & (subjects == test_subject_id)
    return {
        "train": make_loader(train_mask, shuffle=True),
        "validation": make_loader(validation_mask),
        "ideal_test": make_loader(test_mask),
        "induced_test": make_loader(induced_mask) if bool(induced_mask.any()) else None,
        "validation_subject": validation_subject,
        "normalization": {"mean": mean, "std": std},
        "manifest_summary": {"test_subject": test_subject_id, "validation_subject": validation_subject, "train_windows": int(train_mask.sum()), "ideal_test_windows": int(test_mask.sum()), "induced_test_windows": int(induced_mask.sum())},
    }
