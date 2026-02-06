"""
Evaluation metrics for UA-ToM experiments.

Provides:
- Standard metrics (accuracy, F1)
- Switch detection metrics (event-level)
- VLA-centric metrics (degradation, recovery)
- Unified evaluation function
"""

from typing import Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from ..models.base import BaseModel
from ..data.dataset import create_switch_labels


def compute_switch_f1_event(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    tolerance: int = 1,
) -> Dict[str, float]:
    """
    Compute event-level switch detection metrics.
    
    A prediction is a true positive if it falls within ±tolerance
    timesteps of a ground truth switch.
    
    Args:
        predictions: [B, T] predicted switch signals (0 or 1)
        labels: [B, T] ground truth switch labels
        tolerance: Window size for matching
    
    Returns:
        Dictionary with precision, recall, f1, fp_rate
    """
    B, T = labels.shape
    
    true_positives = 0
    false_positives = 0
    total_gt_switches = 0
    detected_switches = 0
    
    for b in range(B):
        # Get ground truth switch timesteps
        gt_switches = (labels[b] == 1).nonzero(as_tuple=True)[0].tolist()
        pred_switches = (predictions[b] == 1).nonzero(as_tuple=True)[0].tolist()
        
        total_gt_switches += len(gt_switches)
        
        # Check each prediction
        for pt in pred_switches:
            matched = False
            for gt in gt_switches:
                if abs(pt - gt) <= tolerance:
                    matched = True
                    break
            
            if matched:
                true_positives += 1
            else:
                false_positives += 1
        
        # Check each ground truth
        for gt in gt_switches:
            detected = False
            for pt in pred_switches:
                if abs(pt - gt) <= tolerance:
                    detected = True
                    break
            if detected:
                detected_switches += 1
    
    # Compute metrics
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = detected_switches / (total_gt_switches + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # False positive rate
    total_negatives = (labels == 0).sum().item()
    fp_rate = false_positives / (total_negatives + 1e-8)
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'fp_rate': fp_rate * 100,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'total_gt_switches': total_gt_switches,
    }


def compute_post_switch_metrics(
    action_correct: torch.Tensor,
    switch_labels: torch.Tensor,
    window_size: int = 5,
    recovery_threshold: float = 0.9,
) -> Dict[str, float]:
    """
    Compute post-switch adaptation metrics.
    
    Args:
        action_correct: [B, T-1] binary correct predictions
        switch_labels: [B, T] ground truth switch labels
        window_size: Window for pre/post comparison
        recovery_threshold: Fraction of pre-switch accuracy to recover
    
    Returns:
        Dictionary with degradation, recovery_time, pre/post accuracy
    """
    B, T = switch_labels.shape
    
    pre_accs = []
    post_accs = []
    recovery_times = []
    
    for b in range(B):
        switches = (switch_labels[b, 1:] == 1).nonzero(as_tuple=True)[0]
        
        for st in switches:
            st = st.item()
            
            # Skip if too close to boundaries
            if st < window_size or st + window_size >= T - 1:
                continue
            
            # Pre-switch accuracy
            pre_acc = action_correct[b, st-window_size:st].float().mean().item()
            pre_accs.append(pre_acc)
            
            # Post-switch accuracy
            post_acc = action_correct[b, st:st+window_size].float().mean().item()
            post_accs.append(post_acc)
            
            # Recovery time
            threshold = pre_acc * recovery_threshold
            recovered = False
            
            for t in range(st, min(T - 1 - 3, st + 20)):
                window_acc = action_correct[b, t:t+3].float().mean().item()
                if window_acc >= threshold:
                    recovery_times.append(t - st)
                    recovered = True
                    break
            
            if not recovered:
                recovery_times.append(20)  # Max
    
    return {
        'pre_switch_acc': np.mean(pre_accs) * 100 if pre_accs else 0,
        'post_switch_acc': np.mean(post_accs) * 100 if post_accs else 0,
        'degradation': (np.mean(pre_accs) - np.mean(post_accs)) * 100 if pre_accs else 0,
        'recovery_time': np.mean(recovery_times) if recovery_times else 0,
    }


def evaluate_model(
    model: BaseModel,
    dataloader: DataLoader,
    device: torch.device,
    switch_threshold: float = 0.3,
    tolerance: int = 1,
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        dataloader: Validation data
        device: Device
        switch_threshold: Threshold for switch detection
        tolerance: Window for event-level F1
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    # Collectors
    all_action_preds = []
    all_action_targets = []
    all_type_preds = []
    all_type_targets = []
    all_switch_preds = []
    all_switch_labels = []
    all_action_correct = []
    
    with torch.no_grad():
        for batch in dataloader:
            observations = batch['observations'].to(device)
            actions = batch['actions'].to(device)
            types = batch['types'].to(device)
            images = batch.get('images')
            if images is not None:
                images = images.to(device)
            
            # Forward
            outputs = model(observations, actions, types, images)
            
            # Get switch signal
            switch_signal = model.get_switch_signal(outputs)
            
            # Action predictions
            action_preds = outputs.action_logits[:, :-1].argmax(dim=-1)
            action_targets = actions[:, 1:]
            
            all_action_preds.append(action_preds.cpu())
            all_action_targets.append(action_targets.cpu())
            all_action_correct.append((action_preds == action_targets).cpu())
            
            # Type predictions
            type_preds = outputs.type_logits.argmax(dim=-1)
            all_type_preds.append(type_preds.cpu())
            all_type_targets.append(types.cpu())
            
            # Switch detection
            switch_preds = (switch_signal > switch_threshold).long()
            switch_labels = create_switch_labels(types)
            
            all_switch_preds.append(switch_preds.cpu())
            all_switch_labels.append(switch_labels.cpu())
    
    # Concatenate
    action_preds = torch.cat(all_action_preds)
    action_targets = torch.cat(all_action_targets)
    type_preds = torch.cat(all_type_preds)
    type_targets = torch.cat(all_type_targets)
    switch_preds = torch.cat(all_switch_preds)
    switch_labels = torch.cat(all_switch_labels)
    action_correct = torch.cat(all_action_correct)
    
    # Compute metrics
    metrics = {}
    
    # Basic accuracy
    metrics['action_acc'] = accuracy_score(
        action_targets.flatten().numpy(),
        action_preds.flatten().numpy(),
    ) * 100
    
    metrics['type_acc'] = accuracy_score(
        type_targets.flatten().numpy(),
        type_preds.flatten().numpy(),
    ) * 100
    
    # Timestep-level switch metrics
    switch_preds_flat = switch_preds.flatten().numpy()
    switch_labels_flat = switch_labels.flatten().numpy()
    
    metrics['switch_f1_timestep'] = f1_score(
        switch_labels_flat, switch_preds_flat, zero_division=0,
    ) * 100
    
    # Event-level switch metrics
    event_metrics = compute_switch_f1_event(switch_preds, switch_labels, tolerance)
    metrics['switch_f1'] = event_metrics['f1']
    metrics['switch_precision'] = event_metrics['precision']
    metrics['switch_recall'] = event_metrics['recall']
    metrics['fp_rate'] = event_metrics['fp_rate']
    
    # Post-switch metrics
    post_metrics = compute_post_switch_metrics(action_correct, switch_labels)
    metrics.update(post_metrics)
    
    return metrics


def evaluate_zeroshot(
    model: BaseModel,
    dataloader: DataLoader,
    device: torch.device,
    held_out_type: int,
    switch_threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Zero-shot evaluation for novel partner types.
    
    Measures detection of switches TO/FROM unseen partner types.
    """
    model.eval()
    
    to_novel_total = 0
    to_novel_detected = 0
    from_novel_total = 0
    from_novel_detected = 0
    known_total = 0
    known_detected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            observations = batch['observations'].to(device)
            actions = batch['actions'].to(device)
            types = batch['types'].to(device)
            images = batch.get('images')
            if images is not None:
                images = images.to(device)
            
            outputs = model(observations, actions, types, images)
            switch_signal = model.get_switch_signal(outputs)
            
            B, T = types.shape
            
            for b in range(B):
                # Find switch types
                for t in range(1, T):
                    if types[b, t] != types[b, t-1]:
                        # Check detection in window
                        w_s = max(0, t - 1)
                        w_e = min(t + 2, T)
                        detected = (switch_signal[b, w_s:w_e] > switch_threshold).any()
                        
                        if types[b, t] == held_out_type:
                            # Switch TO novel
                            to_novel_total += 1
                            if detected:
                                to_novel_detected += 1
                        elif types[b, t-1] == held_out_type:
                            # Switch FROM novel
                            from_novel_total += 1
                            if detected:
                                from_novel_detected += 1
                        else:
                            # Known-to-known
                            known_total += 1
                            if detected:
                                known_detected += 1
    
    return {
        'to_novel_detection': (to_novel_detected / max(to_novel_total, 1)) * 100,
        'from_novel_detection': (from_novel_detected / max(from_novel_total, 1)) * 100,
        'known_detection': (known_detected / max(known_total, 1)) * 100,
        'n_to_novel': to_novel_total,
        'n_from_novel': from_novel_total,
        'n_known': known_total,
    }
