"""
Dataset classes for UA-ToM experiments.

Supports:
- ManiSkill manipulation tasks
- Multi-task loading
- Various data formats
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class PartnerDataset(Dataset):
    """
    Unified dataset for partner modeling experiments.
    
    Loads data from .npz files with the following expected keys:
    - images: [N, T, H, W, C] RGB images (optional)
    - observations: [N, T, obs_dim] state observations (optional)
    - partner_actions: [N, T] or [N, T, action_dim] partner actions
    - partner_types: [N, T] partner type labels
    
    At least one of (images, observations) must be present.
    """
    
    def __init__(
        self,
        data_path: str,
        max_episodes: Optional[int] = None,
        flatten_images: bool = False,
    ):
        """
        Args:
            data_path: Path to .npz data file
            max_episodes: Limit number of episodes (for debugging)
            flatten_images: If True, flatten images to observation vectors
        """
        self.data_path = data_path
        self.flatten_images = flatten_images
        
        print(f"Loading data from {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        
        # Load images
        self.images = None
        if 'images' in data:
            images = data['images']
            # Convert HWC -> CHW if needed
            if images.shape[-1] == 3 and len(images.shape) == 5:
                images = np.transpose(images, (0, 1, 4, 2, 3))
            self.images = torch.from_numpy(images).float() / 255.0
            print(f"  Images: {self.images.shape}")
        
        # Load observations
        self.observations = None
        if 'observations' in data:
            self.observations = torch.from_numpy(data['observations']).float()
            print(f"  Observations: {self.observations.shape}")
        elif 'states' in data:
            self.observations = torch.from_numpy(data['states']).float()
            print(f"  States: {self.observations.shape}")
        elif self.images is not None and flatten_images:
            # Flatten images as observations
            B, T, C, H, W = self.images.shape
            self.observations = self.images.view(B, T, -1)
            print(f"  Flattened images: {self.observations.shape}")
        
        # Load actions
        actions = data['partner_actions']
        if len(actions.shape) == 3:
            # Continuous -> discrete via argmax
            actions = np.argmax(actions, axis=-1)
        self.actions = torch.from_numpy(actions).long()
        self.action_dim = int(self.actions.max().item()) + 1
        print(f"  Actions: {self.actions.shape}, dim={self.action_dim}")
        
        # Load types
        self.types = torch.from_numpy(data['partner_types']).long()
        self.num_types = int(self.types.max().item()) + 1
        print(f"  Types: {self.types.shape}, num={self.num_types}")
        
        # Compute observation dimension
        if self.observations is not None:
            self.obs_dim = self.observations.shape[-1]
        elif self.images is not None:
            self.obs_dim = self.images.shape[2] * self.images.shape[3] * self.images.shape[4]
        else:
            raise ValueError("No observations or images found in data")
        
        # Limit episodes
        if max_episodes is not None and max_episodes < len(self.actions):
            self.images = self.images[:max_episodes] if self.images is not None else None
            self.observations = self.observations[:max_episodes] if self.observations is not None else None
            self.actions = self.actions[:max_episodes]
            self.types = self.types[:max_episodes]
        
        print(f"  Total: {len(self)} episodes")
    
    def __len__(self) -> int:
        return self.actions.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'actions': self.actions[idx],
            'types': self.types[idx],
        }
        
        if self.images is not None:
            item['images'] = self.images[idx]
        
        if self.observations is not None:
            item['observations'] = self.observations[idx]
        elif self.images is not None:
            # Flatten images on-the-fly
            item['observations'] = self.images[idx].view(self.images.shape[1], -1)
        
        return item
    
    @property
    def seq_len(self) -> int:
        return self.actions.shape[1]


class MultiTaskDataset(Dataset):
    """
    Dataset supporting multiple tasks for generalization experiments.
    """
    
    def __init__(self, data_paths: Dict[str, str], **kwargs):
        """
        Args:
            data_paths: Dict mapping task_name -> data_path
        """
        self.tasks = {}
        self.task_indices = []  # (task_name, episode_idx)
        
        for task_name, path in data_paths.items():
            if not Path(path).exists():
                print(f"  Warning: {task_name} not found at {path}, skipping")
                continue
            
            dataset = PartnerDataset(path, **kwargs)
            self.tasks[task_name] = dataset
            
            for i in range(len(dataset)):
                self.task_indices.append((task_name, i))
        
        if not self.tasks:
            raise ValueError("No valid tasks found!")
        
        # Use first task's dimensions
        first = next(iter(self.tasks.values()))
        self.action_dim = first.action_dim
        self.num_types = first.num_types
        self.obs_dim = first.obs_dim
        
        print(f"  Multi-task: {len(self)} episodes across {len(self.tasks)} tasks")
    
    def __len__(self) -> int:
        return len(self.task_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_name, episode_idx = self.task_indices[idx]
        item = self.tasks[task_name][episode_idx]
        item['task_name'] = task_name
        return item


def create_switch_labels(types: torch.Tensor) -> torch.Tensor:
    """
    Create binary switch labels from type sequence.
    
    Args:
        types: [B, T] or [T] partner types
    
    Returns:
        [B, T] or [T] binary labels (1 = switch occurred)
    """
    if types.dim() == 1:
        switches = torch.zeros_like(types)
        switches[1:] = (types[1:] != types[:-1]).long()
        return switches
    
    switches = torch.zeros_like(types)
    switches[:, 1:] = (types[:, 1:] != types[:, :-1]).long()
    return switches


def get_dataloaders(
    data_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, PartnerDataset]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_path: Path to data file
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_ratio: Fraction of data for training
        seed: Random seed for split
    
    Returns:
        (train_loader, val_loader, dataset)
    """
    dataset = PartnerDataset(data_path, **dataset_kwargs)
    
    # Split indices
    n = len(dataset)
    n_train = int(train_ratio * n)
    
    torch.manual_seed(seed)
    indices = torch.randperm(n).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, dataset
