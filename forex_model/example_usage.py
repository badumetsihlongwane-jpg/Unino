"""
Complete Example: MRAT Forex Prediction Model
Demonstrates end-to-end usage from data loading to prediction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta

# Import MRAT components
from models.mrat_model import MRATForexModel, MRATConfig, create_mrat_model
from transformers.data_transformer import DataPipeline
from utils.training import (
    ForexDataset, collate_fn, train_mrat_model,
    generate_prediction_output, evaluate_by_regime
)
from torch.utils.data import DataLoader


def generate_synthetic_data(n_samples: int = 1000) -> tuple:
    """
    Generate synthetic forex data for demonstration.
    
    In production, this would load real market data from:
    - Price data provider (e.g., MetaTrader, Bloomberg)
    - Economic calendar API
    - Central bank data feeds
    - Commodity price feeds
    """
    print("Generating synthetic forex data...")
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # 1. OHLC Price Data
    np.random.seed(42)
    base_price = 1.0850  # EUR/USD starting price
    price_returns = np.random.randn(n_samples) * 0.0005
    close_prices = base_price * np.exp(np.cumsum(price_returns))
    
    ohlc = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(n_samples) * 0.0002),
        'high': close_prices * (1 + abs(np.random.randn(n_samples)) * 0.0005),
        'low': close_prices * (1 - abs(np.random.randn(n_samples)) * 0.0005),
        'close': close_prices
    }, index=dates)
    
    # 2. Economic Calendar Events
    event_types = ['NFP', 'CPI', 'FOMC', 'ECB', 'GDP', 'PMI', 'Retail_Sales']
    n_events = 50
    
    events = pd.DataFrame({
        'event_type': np.random.choice(event_types, n_events),
        'time': pd.date_range(start_date, periods=n_events, freq='7D'),
        'actual': np.random.randn(n_events) * 0.5 + 2.5,
        'forecast': np.random.randn(n_events) * 0.5 + 2.5,
        'previous': np.random.randn(n_events) * 0.5 + 2.5
    })
    
    # 3. Macro Indicators
    macro_data = {
        'rates': pd.DataFrame({
            'rate_a': 2.5 + np.cumsum(np.random.randn(n_samples) * 0.005),  # USD rate
            'rate_b': 1.5 + np.cumsum(np.random.randn(n_samples) * 0.005)   # EUR rate
        }, index=dates),
        'yields': pd.DataFrame({
            '2Y': 2.0 + np.random.randn(n_samples) * 0.1,
            '5Y': 2.5 + np.random.randn(n_samples) * 0.1,
            '10Y': 3.0 + np.random.randn(n_samples) * 0.1
        }, index=dates),
        'commodities': pd.DataFrame({
            'gold': 1800 + np.cumsum(np.random.randn(n_samples) * 2),
            'oil': 80 + np.cumsum(np.random.randn(n_samples) * 0.5),
            'copper': 4.0 + np.cumsum(np.random.randn(n_samples) * 0.02)
        }, index=dates),
        'fx': ohlc
    }
    
    # 4. Generate labels (direction: 0=LONG, 1=NEUTRAL, 2=SHORT)
    future_returns = np.diff(close_prices, prepend=close_prices[0])
    labels = np.where(future_returns > 0.0005, 0,  # LONG
                     np.where(future_returns < -0.0005, 2, 1))  # SHORT or NEUTRAL
    
    print(f"Generated {n_samples} samples with {n_events} events")
    
    return ohlc, events, macro_data, labels


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("MRAT Forex Prediction Model - Complete Demo")
    print("=" * 80)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Step 1: Generate Data
    print("\n" + "=" * 80)
    print("STEP 1: Data Generation")
    print("=" * 80)
    ohlc, events, macro_data, labels = generate_synthetic_data(n_samples=500)
    
    # Step 2: Transform Data
    print("\n" + "=" * 80)
    print("STEP 2: Data Transformation")
    print("=" * 80)
    pipeline = DataPipeline()
    transformed_samples = pipeline.prepare_training_data(
        ohlc, events, macro_data, window_size=100
    )
    print(f"Created {len(transformed_samples)} transformed samples")
    
    # Align labels with samples
    sample_labels = labels[100:]  # Skip first 100 (window size)
    sample_labels = sample_labels[:len(transformed_samples)]
    
    # Step 3: Create Dataset and DataLoader
    print("\n" + "=" * 80)
    print("STEP 3: Dataset Creation")
    print("=" * 80)
    
    # Split train/val
    split_idx = int(len(transformed_samples) * 0.8)
    train_samples = transformed_samples[:split_idx]
    val_samples = transformed_samples[split_idx:]
    train_labels = sample_labels[:split_idx]
    val_labels = sample_labels[split_idx:]
    
    train_dataset = ForexDataset(train_samples, train_labels)
    val_dataset = ForexDataset(val_samples, val_labels)
    
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Step 4: Create Model
    print("\n" + "=" * 80)
    print("STEP 4: Model Creation")
    print("=" * 80)
    config = MRATConfig()
    model = create_mrat_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Step 5: Training (abbreviated for demo)
    print("\n" + "=" * 80)
    print("STEP 5: Model Training (Demo - 5 epochs)")
    print("=" * 80)
    
    training_config = {
        'lr': 1e-4,
        'epochs': 5,  # Reduced for demo
        'patience': 3,
        'lambda_cal': 0.1,
        'lambda_att': 0.01,
        'lambda_div': 0.001,
        'lambda_regime': 0.05
    }
    
    history = train_mrat_model(
        model, train_loader, val_loader,
        training_config, device=device
    )
    
    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Step 6: Generate Prediction
    print("\n" + "=" * 80)
    print("STEP 6: Generate Prediction Example")
    print("=" * 80)
    
    # Get a sample from validation set
    sample_batch = next(iter(val_loader))
    
    # Generate prediction
    prediction = generate_prediction_output(
        model, sample_batch, pair_name="EUR/USD", device=device
    )
    
    # Display prediction
    print("\n" + "-" * 80)
    print("PREDICTION OUTPUT (JSON Format):")
    print("-" * 80)
    print(json.dumps(prediction, indent=2))
    
    # Step 7: Explainability Demonstration
    print("\n" + "=" * 80)
    print("STEP 7: Explainability Analysis")
    print("=" * 80)
    
    print("\nDirection Prediction:")
    print(f"  - Predicted: {prediction['prediction']['direction']}")
    print(f"  - Confidence: {prediction['prediction']['confidence']:.2%}")
    print(f"  - Probabilities:")
    for dir_label, prob in prediction['prediction']['probability_distribution'].items():
        print(f"      {dir_label}: {prob:.2%}")
    
    print("\nRegime Detection:")
    for regime_name, regime_info in prediction['regime_state'].items():
        print(f"  - {regime_name}: {regime_info['label']} ({regime_info['probability']:.2%})")
    
    print("\nFactor Attribution (What drove the prediction?):")
    sorted_factors = sorted(
        prediction['explainability']['factor_attribution'].items(),
        key=lambda x: x[1], reverse=True
    )
    for factor, contribution in sorted_factors:
        print(f"  - {factor}: {contribution:.2%}")
    
    print("\nRisk Assessment:")
    risk = prediction['risk_assessment']
    print(f"  - Volatility Regime: {risk['volatility_regime']}")
    print(f"  - Tail Risk Score: {risk['tail_risk_score']:.2%}")
    print(f"  - Liquidity Warning: {risk['liquidity_warning']}")
    print(f"  - Optimal Position Size: {risk['optimal_position_size']:.2%}")
    
    # Step 8: Key Innovations Summary
    print("\n" + "=" * 80)
    print("STEP 8: Key Innovations Summary")
    print("=" * 80)
    
    innovations = [
        "✓ Price Geometry Encoding: Transform OHLC into topological features",
        "✓ Event Embedding: Learned macro-event space with temporal decay",
        "✓ Macro State Encoder: Policy stance and yield curve geometry",
        "✓ Regime Detection: Multi-dimensional market state classification",
        "✓ Regime-Conditioned Attention: Adaptive fusion based on market regime",
        "✓ Hierarchical Reasoning: Macro → Meso → Micro time horizons",
        "✓ Full Explainability: Traceable predictions with factor attribution",
        "✓ Confidence Calibration: Regime-aware confidence scoring",
        "✓ Overfitting Prevention: Multi-currency training, regularization"
    ]
    
    for innovation in innovations:
        print(f"  {innovation}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nThe MRAT model demonstrates:")
    print("  1. Novel architecture combining multiple innovations")
    print("  2. Multi-modal data fusion with regime awareness")
    print("  3. Explainable predictions with factor attribution")
    print("  4. Robust training with overfitting prevention")
    print("  5. Complete prediction pipeline from data to output")
    
    print("\n" + "=" * 80)
    
    # Save example output
    output_file = "example_prediction_output.json"
    with open(output_file, 'w') as f:
        json.dump(prediction, f, indent=2)
    print(f"\nExample prediction saved to: {output_file}")
    
    return model, prediction


if __name__ == '__main__':
    try:
        model, prediction = main()
        print("\n✓ All steps completed successfully!")
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
