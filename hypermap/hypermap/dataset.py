"""
hypermap/dataset.py
PyTorch Dataset for HyperMAP training and evaluation.
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


class PerturbDataset(Dataset):
    """
    Pairs of (control_expression, perturbation_embedding, true_delta, condition_id).

    All tensors are stored as-is; device placement is handled by the trainer.
    """

    def __init__(self, control_exp, pert_emb, pert_exp, pert_conditions):
        self.control_exp     = control_exp
        self.pert_emb        = pert_emb
        self.pert_exp        = pert_exp
        self.pert_conditions = pert_conditions

    def __len__(self):
        return len(self.pert_exp)

    def __getitem__(self, idx):
        return (
            self.control_exp[idx],
            self.pert_emb[idx],
            self.pert_exp[idx],
            self.pert_conditions[idx],
        )


def collate_min2(batch):
    """
    Collate function that ensures batch size >= 2.
    Duplicates the single sample if a size-1 tail batch appears.
    This prevents degenerate gradients in CrossAttention's softmax over dim=0.
    """
    if len(batch) == 1:
        batch = batch + [batch[0]]
    return default_collate(batch)
