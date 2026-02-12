"""
Training utilities for MRAT Forex Model
Includes loss functions, metrics, and training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader


class ForexDataset(Dataset):
    """PyTorch Dataset for MRAT model."""
    
    def __init__(self, samples: List, labels: np.ndarray = None):
        self.samples = samples
        self.labels = labels
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        batch = {
            'ohlc': torch.FloatTensor(sample.ohlc),
            'events': {
                k: torch.FloatTensor(v) if v.dtype == np.float64 
                else torch.LongTensor(v)
                for k, v in sample.events.items()
            },
            'macro_indicators': {
                k: torch.FloatTensor(v)
                for k, v in sample.macro_indicators.items()
            }
        }
        
        if self.labels is not None:
            batch['label'] = torch.LongTensor([self.labels[idx]])[0]
        
        return batch


def collate_fn(batch_list):
    """Custom collate function to handle variable-length sequences."""
    # Stack all tensors
    collated = {
        'ohlc': torch.stack([b['ohlc'] for b in batch_list]),
        'events': {},
        'macro_indicators': {}
    }
    
    # Handle events - pad to max length in batch
    for key in batch_list[0]['events'].keys():
        tensors = [b['events'][key] for b in batch_list]
        # Find max length
        max_len = max(t.shape[0] if len(t.shape) > 0 else 1 for t in tensors)
        
        # Pad all tensors to max length
        padded = []
        for t in tensors:
            if len(t.shape) == 0:  # Scalar
                t = t.unsqueeze(0)
            if t.shape[0] < max_len:
                pad_size = max_len - t.shape[0]
                t = torch.cat([t, torch.zeros(pad_size, dtype=t.dtype)])
            padded.append(t)
        
        collated['events'][key] = torch.stack(padded)
    
    # Handle macro indicators
    for key in batch_list[0]['macro_indicators'].keys():
        collated['macro_indicators'][key] = torch.stack([
            b['macro_indicators'][key] for b in batch_list
        ])
    
    # Handle labels if present
    if 'label' in batch_list[0]:
        collated['labels'] = torch.stack([b['label'] for b in batch_list])
    
    return collated


class RegimeStratifiedSampler:
    """
    Ensures all regimes are represented in each batch.
    Prevents model from ignoring rare regimes.
    """
    
    def __init__(self, regime_labels: np.ndarray, batch_size: int):
        self.regime_labels = regime_labels
        self.batch_size = batch_size
        self.unique_regimes = np.unique(regime_labels)
        
        # Group indices by regime
        self.regime_indices = {
            regime: np.where(regime_labels == regime)[0]
            for regime in self.unique_regimes
        }
    
    def __iter__(self):
        """Generate batches with regime stratification."""
        while True:
            batch_indices = []
            samples_per_regime = self.batch_size // len(self.unique_regimes)
            
            for regime in self.unique_regimes:
                # Sample from this regime
                regime_idx = self.regime_indices[regime]
                selected = np.random.choice(
                    regime_idx,
                    size=min(samples_per_regime, len(regime_idx)),
                    replace=False
                )
                batch_indices.extend(selected)
            
            # Shuffle batch
            np.random.shuffle(batch_indices)
            yield batch_indices[:self.batch_size]
    
    def __len__(self):
        return len(self.regime_labels) // self.batch_size


class CalibrationLoss(nn.Module):
    """
    Loss for confidence calibration.
    Ensures model confidence matches actual accuracy.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, confidence: torch.Tensor, predictions: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            confidence: [batch] predicted confidence scores
            predictions: [batch, n_classes] prediction logits
            labels: [batch] true labels
        Returns:
            calibration_loss: scalar
        """
        # Get predicted class
        pred_class = torch.argmax(predictions, dim=-1)
        
        # Correctness indicator
        correct = (pred_class == labels).float()
        
        # Calibration loss: MSE between confidence and correctness
        calibration_loss = F.mse_loss(confidence.squeeze(), correct)
        
        return calibration_loss


class AttentionRegularizationLoss(nn.Module):
    """
    Regularizes attention weights to prevent over-focusing.
    Encourages attention diversity across modalities.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, attention_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            attention_weights: Dictionary of attention weight tensors
        Returns:
            reg_loss: scalar
        """
        total_loss = 0.0
        
        for key, weights in attention_weights.items():
            # L2 norm of attention weights
            l2_loss = torch.mean(weights ** 2)
            total_loss += l2_loss
        
        return total_loss / len(attention_weights)


class FactorDiversityLoss(nn.Module):
    """
    Encourages using multiple factors (not just one dominant signal).
    Uses negative entropy as diversity measure.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, factor_attribution: torch.Tensor) -> torch.Tensor:
        """
        Args:
            factor_attribution: [batch, n_factors] attribution weights (sum to 1)
        Returns:
            diversity_loss: scalar (negative entropy)
        """
        # Compute entropy of attribution distribution
        entropy = -torch.sum(
            factor_attribution * torch.log(factor_attribution + 1e-8),
            dim=-1
        )
        
        # We want HIGH entropy (diverse attribution)
        # So loss is NEGATIVE entropy (minimize negative entropy = maximize entropy)
        diversity_loss = -torch.mean(entropy)
        
        return diversity_loss


class MRATLoss(nn.Module):
    """
    Combined loss function for MRAT model.
    
    Components:
    1. Prediction loss (cross-entropy)
    2. Confidence calibration loss
    3. Attention regularization
    4. Factor diversity loss
    5. Regime detection loss (if labels available)
    """
    
    def __init__(self, config):
        super().__init__()
        self.lambda_cal = config.get('lambda_cal', 0.1)
        self.lambda_att = config.get('lambda_att', 0.01)
        self.lambda_div = config.get('lambda_div', 0.001)
        self.lambda_regime = config.get('lambda_regime', 0.05)
        
        self.calibration_loss = CalibrationLoss()
        self.attention_reg = AttentionRegularizationLoss()
        self.diversity_loss = FactorDiversityLoss()
    
    def forward(self, outputs: Dict, labels: torch.Tensor,
                regime_labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and individual components.
        
        Args:
            outputs: Model outputs dictionary
            labels: Ground truth labels [batch]
            regime_labels: Optional regime labels [batch, n_regime_dims]
        Returns:
            losses: Dictionary of loss components
        """
        # 1. Prediction loss
        pred_loss = F.cross_entropy(outputs['logits'], labels)
        
        # 2. Calibration loss
        cal_loss = self.calibration_loss(
            outputs['confidence'],
            outputs['logits'],
            labels
        )
        
        # 3. Attention regularization
        att_loss = self.attention_reg(outputs['attention_weights'])
        
        # 4. Factor diversity loss
        div_loss = self.diversity_loss(outputs['factor_attribution'])
        
        # 5. Regime detection loss (optional)
        regime_loss = torch.tensor(0.0, device=pred_loss.device)
        if regime_labels is not None:
            # Assuming we have separate logits for each regime dimension
            # This is a placeholder - would need actual regime logits
            pass
        
        # Total loss
        total_loss = (
            pred_loss +
            self.lambda_cal * cal_loss +
            self.lambda_att * att_loss +
            self.lambda_div * div_loss +
            self.lambda_regime * regime_loss
        )
        
        losses = {
            'total': total_loss,
            'prediction': pred_loss,
            'calibration': cal_loss,
            'attention_reg': att_loss,
            'diversity': div_loss,
            'regime': regime_loss
        }
        
        return losses


def evaluate_by_regime(model: nn.Module, dataloader: DataLoader,
                      device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate model performance stratified by regime.
    
    Args:
        model: MRAT model
        dataloader: Validation dataloader
        device: Computation device
    Returns:
        metrics: Dictionary of metrics by regime
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_regimes = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Collect predictions
            preds = torch.argmax(outputs['direction_probs'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_confidences.extend(outputs['confidence'].cpu().numpy())
            
            # Collect regime classifications
            regime_probs = outputs['regime_probs']
            # Get dominant regime for each dimension
            risk_regime = torch.argmax(regime_probs['risk_sentiment'], dim=-1)
            vol_regime = torch.argmax(regime_probs['volatility'], dim=-1)
            all_regimes.append({
                'risk': risk_regime.cpu().numpy(),
                'volatility': vol_regime.cpu().numpy()
            })
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Overall metrics
    overall_acc = accuracy_score(all_labels, all_preds)
    
    # Regime-specific metrics (placeholder - would need full implementation)
    metrics = {
        'overall_accuracy': overall_acc,
        'mean_confidence': np.mean(all_confidences),
        # Would add regime-stratified metrics here
    }
    
    return metrics


def train_mrat_model(model: nn.Module, train_loader: DataLoader,
                    val_loader: DataLoader, config: Dict,
                    device: str = 'cpu') -> Dict[str, List[float]]:
    """
    Training loop for MRAT model.
    
    Args:
        model: MRAT model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        device: Computation device
    Returns:
        history: Training history
    """
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss function
    criterion = MRATLoss(config)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_confidence': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.get('epochs', 100)):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Compute loss
            losses = criterion(outputs, batch['labels'])
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_losses.append(losses['total'].item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        val_metrics = evaluate_by_regime(model, val_loader, device)
        val_acc = val_metrics['overall_accuracy']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_train_loss)  # Using train loss for now
        history['val_accuracy'].append(val_acc)
        history['val_confidence'].append(val_metrics['mean_confidence'])
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)
        
        # Early stopping based on validation accuracy
        if val_acc > best_val_loss:  # Using accuracy as metric
            best_val_loss = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_mrat_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= config.get('patience', 10):
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Acc = {val_metrics['overall_accuracy']:.4f}")
    
    return history


def generate_prediction_output(model: nn.Module, batch: Dict,
                              pair_name: str = "EUR/USD",
                              device: str = 'cpu') -> Dict:
    """
    Generate complete prediction output with explainability.
    
    Args:
        model: Trained MRAT model
        batch: Input batch
        pair_name: Currency pair name
        device: Computation device
    Returns:
        prediction_output: Complete output dictionary (JSON-ready)
    """
    model.eval()
    
    # Move batch to device
    batch = {k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(batch)
    
    # Extract predictions
    direction_probs = outputs['direction_probs'][0].cpu().numpy()
    confidence = outputs['confidence'][0].item()
    risk_scores = outputs['risk_scores'][0].cpu().numpy()
    factor_attr = outputs['factor_attribution'][0].cpu().numpy()
    regime_probs = {
        k: v[0].cpu().numpy() for k, v in outputs['regime_probs'].items()
        if k != 'gate_weights'
    }
    
    # Determine direction
    direction_labels = ['LONG', 'NEUTRAL', 'SHORT']
    direction_idx = np.argmax(direction_probs)
    direction = direction_labels[direction_idx]
    
    # Regime labels
    regime_labels_map = {
        'risk_sentiment': ['RISK_ON', 'RISK_OFF'],
        'volatility': ['LOW', 'MEDIUM', 'HIGH', 'EXTREME'],
        'trend': ['RANGE', 'WEAK', 'STRONG', 'EXTREME'],
        'policy': ['EASING', 'NEUTRAL', 'TIGHTENING'],
        'stress': ['ORDERLY', 'STRESSED', 'CRISIS']
    }
    
    regime_state = {}
    for regime_dim, probs in regime_probs.items():
        if regime_dim in regime_labels_map:
            labels = regime_labels_map[regime_dim]
            dominant_idx = np.argmax(probs)
            regime_state[regime_dim] = {
                'label': labels[dominant_idx],
                'probability': float(probs[dominant_idx])
            }
    
    # Factor names
    factor_names = [
        'interest_rate_differential',
        'yield_curve_dynamics',
        'commodity_correlation',
        'price_momentum',
        'economic_events',
        'sentiment'
    ]
    
    factor_attribution = {
        name: float(attr)
        for name, attr in zip(factor_names, factor_attr)
    }
    
    # Construct output
    prediction_output = {
        'pair': pair_name,
        'timestamp': '2024-01-15T14:30:00Z',  # Would use actual timestamp
        'prediction': {
            'direction': direction,
            'probability_distribution': {
                'LONG': float(direction_probs[0]),
                'NEUTRAL': float(direction_probs[1]),
                'SHORT': float(direction_probs[2])
            },
            'confidence': float(confidence),
            'expected_move': {
                'mean': float(np.random.randn() * 0.005),  # Placeholder
                'std': 0.0031,
                'percentiles': {
                    '5': -0.0042,
                    '50': 0.0035,
                    '95': 0.0115
                }
            },
            'horizon': '4H'
        },
        'risk_assessment': {
            'volatility_regime': 'MEDIUM',
            'tail_risk_score': float(risk_scores[0]),
            'liquidity_warning': bool(risk_scores[2] > 0.7),
            'optimal_position_size': float(confidence * (1 - risk_scores[0]))
        },
        'regime_state': regime_state,
        'explainability': {
            'factor_attribution': factor_attribution,
            'key_drivers': [
                {
                    'factor': 'interest_rate_differential',
                    'impact': factor_attribution['interest_rate_differential'],
                    'description': 'Interest rate dynamics driving currency valuation'
                }
            ]
        }
    }
    
    return prediction_output


if __name__ == '__main__':
    print("MRAT Training Utilities loaded successfully")
