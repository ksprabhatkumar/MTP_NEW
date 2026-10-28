"""Rotation-aware REALDISP features and leakage-free Objective 2 folds."""
import json
from pathlib import Path
import numpy as np
import torch

from workstation_experiments import Corpus

INVALID = {'subject6_self.log', 'subject13_self.log', 'subject15_mutual4.log'}
WIDTHS = {'raw': 6, 'raw_aug': 6, 'magnitude': 2, 'mag_dot': 5}


def features(raw, kind):
    """Convert [B,T,N,6] raw acc/gyro into local rotation invariants."""
    if kind in ('raw', 'raw_aug'):
        return raw
    acceleration = raw[..., :3]
    angular = raw[..., 3:]
    a_norm = torch.linalg.vector_norm(acceleration, dim=-1)
    g_norm = torch.linalg.vector_norm(angular, dim=-1)
    if kind == 'magnitude':
        return torch.stack((a_norm, g_norm), dim=-1)
    if kind != 'mag_dot':
        raise ValueError(kind)
    # The first lagged dot product uses the first vector itself.
    lag_a = torch.cat((acceleration[:, :1], acceleration[:, :-1]), dim=1)
    lag_g = torch.cat((angular[:, :1], angular[:, :-1]), dim=1)
    return torch.stack((a_norm, g_norm,
        (acceleration * angular).sum(-1),
        (acceleration * lag_a).sum(-1),
        (angular * lag_g).sum(-1)), dim=-1)


def random_rotations(batch, nodes, device, generator=None):
    """Independent proper SO(3) rotation per window and sensor."""
    q = torch.randn(batch, nodes, 4, device=device, generator=generator)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), -1
    ).reshape(batch, nodes, 3, 3)


def rotate_raw(raw, rotation):
    if raw.ndim != 4 or raw.shape[-1] != 6 or rotation.shape != (len(raw), raw.shape[2], 3, 3):
        raise ValueError('Expected raw [B,T,N,6], rotations [B,N,3,3]')
    a = torch.einsum('bnij,btnj->btni', rotation, raw[..., :3])
    g = torch.einsum('bnij,btnj->btni', rotation, raw[..., 3:])
    return torch.cat((a, g), -1)


class O2Corpus(Corpus):
    def __init__(self):
        super().__init__()
        invalid_rows = [row for row in self.audit['recordings'] if row['source'] in INVALID]
        if {row['source'] for row in invalid_rows} != INVALID:
            raise ValueError('Expected zero-signal files differ from preflight audit')
        for row in invalid_rows:
            indices = np.flatnonzero(self.sources == row['source'])
            if not bool((self.x[indices] == 0).all()):
                raise ValueError(f'Invalid-file assumption no longer true: {row["source"]}')
        self.valid = ~np.isin(self.sources, list(INVALID))
        self.valid_manifest = {'excluded': [row['source'] for row in invalid_rows],
            'retained_windows': int(self.valid.sum()),
            'retained_ideal_subjects': sorted(set(self.subjects[(self.scenarios == 'ideal') & self.valid].tolist()))}
        if self.valid_manifest['retained_ideal_subjects'] != list(range(1, 18)):
            raise ValueError('Ideal subject coverage changed')
        self.eligible_validation = []
        for subject in range(1, 18):
            masks = [(self.subjects == subject) & (self.scenarios == scenario) & self.valid
                     for scenario in ('ideal', 'self')]
            if all(mask.any() and len(set(self.y[mask].tolist())) >= 30 for mask in masks):
                self.eligible_validation.append(subject)

    def fold(self, subject, kind):
        if kind not in WIDTHS:
            raise ValueError(kind)
        eligible = [s for s in self.eligible_validation if s != subject]
        validation = next((s for s in eligible if s > subject), eligible[0])
        ideal = self.scenarios == 'ideal'
        train_mask = ideal & self.valid & ~np.isin(self.subjects, [subject, validation])
        masks = {'train': train_mask,
            'val_ideal': ideal & self.valid & (self.subjects == validation),
            'val_self': (self.scenarios == 'self') & self.valid & (self.subjects == validation),
            'ideal': ideal & self.valid & (self.subjects == subject),
            'self': (self.scenarios == 'self') & self.valid & (self.subjects == subject),
            'induced': (self.scenarios == 'induced') & self.valid & (self.subjects == subject)}
        indices = {key: np.flatnonzero(value) for key, value in masks.items()}
        train_raw = self.x[indices['train']].cuda()
        train_encoded = features(train_raw, kind)
        mean = train_encoded.mean((0, 1), keepdim=True)
        std = train_encoded.std((0, 1), keepdim=True).clamp_min(1e-6)
        data = {}
        for key, ix in indices.items():
            if not len(ix):
                continue
            raw = train_raw if key == 'train' else self.x[ix].cuda()
            available = self.availability[ix].cuda()
            encoded = ((features(raw, kind) - mean) / std) * available[:, None, :, None]
            data[key] = {'x': encoded, 'raw': raw, 'y': self.y[ix].cuda(), 'a': available}
        del train_encoded, train_raw
        # Class support for a validation rule that treats ideal and self fairly.
        val_ideal = set(self.y[indices['val_ideal']].tolist())
        val_self = set(self.y[indices['val_self']].tolist())
        common = sorted(val_ideal & val_self)
        if len(common) < 29:
            raise ValueError(f'Insufficient validation common classes: {validation}, {common}')
        summary = {'test_subject': subject, 'validation_subject': validation,
            'train_subjects': sorted(set(self.subjects[indices['train']].tolist())),
            'feature_mode': kind, 'feature_width': WIDTHS[kind],
            'excluded_recordings': sorted(INVALID),
            'common_validation_classes': common,
            'counts': {key: len(ix) for key, ix in indices.items()},
            'recordings': {key: sorted(set(self.sources[ix].tolist())) for key, ix in indices.items()},
            'normalization': 'feature transform before train-only mean/std by sensor/feature'}
        return {'data': data, 'indices': indices, 'mean': mean.cpu(), 'std': std.cpu(),
                'summary': summary}
