# Macro-Regime Adaptive Transformer (MRAT) - Forex Prediction Model

## Overview

The **Macro-Regime Adaptive Transformer (MRAT)** is a novel, conceptual AI/ML architecture designed for forex prediction that explicitly models macroeconomic structures and market regimes. This implementation goes far beyond standard retail approaches by conceptualizing the "market thinking" process through hierarchical regime-aware attention mechanisms and macro-economic reasoning layers.

## 🚀 Core Innovations

### 1. **Price Geometry Encoding**
- Transforms OHLC data into geometric manifolds
- Extracts curvature tensors (price acceleration)
- Computes volatility surfaces and fractal dimensions
- Detects support/resistance field equations

### 2. **Event Embedding Network**
- Embeds economic events in learned macro-event space
- Models event impact with temporal decay
- Handles pre-event anticipation and post-event drift
- Calculates normalized surprise factors

### 3. **Macro State Encoder**
- Encodes interest rate differential dynamics
- Performs yield curve PCA (level, slope, curvature)
- Models commodity-FX correlations
- Represents central bank policy stance

### 4. **Regime Detection Module**
- Multi-dimensional regime classification:
  - Risk Sentiment (Risk-On/Risk-Off)
  - Volatility (Low/Medium/High/Extreme)
  - Trend Strength (Range/Weak/Strong/Extreme)
  - CB Policy (Easing/Neutral/Tightening)
  - Market Stress (Orderly/Stressed/Crisis)

### 5. **Regime-Conditioned Cross-Attention**
- Adaptive attention weights based on detected regime
- Different modalities dominate in different market conditions
- Price-Macro, Event-Price, and Commodity-FX attention

### 6. **Hierarchical Reasoning**
- Three-tier reasoning mimicking macro trader thinking:
  - **Macro Layer**: Days to weeks (policy, yields, events)
  - **Meso Layer**: Hours to day (session flows, risk transitions)
  - **Micro Layer**: Minutes to hours (price action, execution)

### 7. **Full Explainability**
- Factor attribution (what drove the prediction)
- Attention weight visualization
- Regime contribution scores
- Counterfactual analysis capabilities

## 📁 Project Structure

```
forex_model/
├── models/
│   └── mrat_model.py          # Complete MRAT model implementation
├── transformers/
│   └── data_transformer.py    # Data transformation pipeline
├── utils/
│   └── training.py            # Training utilities and loss functions
├── docs/
│   └── ARCHITECTURE.md        # Detailed architecture documentation
├── example_usage.py           # Complete end-to-end demo
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Installation

```bash
# Navigate to forex_model directory
cd forex_model

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Running the Demo

```bash
python example_usage.py
```

This will:
1. Generate synthetic forex data
2. Transform data into model features
3. Train the MRAT model (5 epochs demo)
4. Generate predictions with full explainability
5. Save example output to JSON

### Expected Output

```json
{
  "pair": "EUR/USD",
  "prediction": {
    "direction": "LONG",
    "probability_distribution": {
      "LONG": 0.68,
      "NEUTRAL": 0.22,
      "SHORT": 0.10
    },
    "confidence": 0.82
  },
  "regime_state": {
    "risk_sentiment": {"label": "RISK_ON", "probability": 0.72},
    "volatility": {"label": "MEDIUM", "probability": 0.65}
  },
  "explainability": {
    "factor_attribution": {
      "interest_rate_differential": 0.42,
      "yield_curve_dynamics": 0.18,
      "price_momentum": 0.22
    }
  }
}
```

## 📚 Usage Examples

### 1. Training on Your Data

```python
from models.mrat_model import create_mrat_model, MRATConfig
from transformers.data_transformer import DataPipeline
from utils.training import train_mrat_model

# Prepare your data
pipeline = DataPipeline()
samples = pipeline.prepare_training_data(ohlc_df, events_df, macro_dict)

# Create model
config = MRATConfig()
model = create_mrat_model(config)

# Train
history = train_mrat_model(model, train_loader, val_loader, config)
```

### 2. Making Predictions

```python
from utils.training import generate_prediction_output

# Generate prediction
prediction = generate_prediction_output(
    model, batch, pair_name="EUR/USD"
)

print(f"Direction: {prediction['prediction']['direction']}")
print(f"Confidence: {prediction['prediction']['confidence']}")
```

### 3. Analyzing Explainability

```python
# Access factor attribution
factors = prediction['explainability']['factor_attribution']

# Most important factor
top_factor = max(factors.items(), key=lambda x: x[1])
print(f"Key driver: {top_factor[0]} ({top_factor[1]:.2%})")

# Regime state
regime = prediction['regime_state']
print(f"Market regime: {regime['volatility']['label']}")
```

## 🏗️ Architecture Details

### Model Components

1. **Encoders** (4 types)
   - Price Geometry Encoder (d=128)
   - Event Embedding Network (d=64)
   - Macro State Encoder (d=96)
   - Commodity Encoder (integrated with macro)

2. **Regime Detection**
   - Mixture of Experts architecture
   - 5 expert networks (one per regime dimension)
   - Gating network for weighted combination

3. **Cross-Attention Fusion**
   - Multi-head attention with regime modulation
   - Price-Macro, Event-Price attention
   - Regime-dependent scaling matrices

4. **Hierarchical Reasoning**
   - Macro reasoning (128-dim output)
   - Meso reasoning (96-dim output)
   - Micro execution (64-dim output)

5. **Output Layer**
   - Direction prediction (3 classes)
   - Confidence calibration
   - Risk assessment (4 risk types)
   - Factor attribution (6 factors)

### Training Strategy

**Overfitting Prevention:**
- Regime-conditional dropout
- Multi-currency pair training
- Temporal cross-validation
- Curriculum learning
- Meta-learning for adaptation

**Loss Function:**
```
Total Loss = Prediction Loss
           + λ_cal × Calibration Loss
           + λ_att × Attention Regularization
           + λ_div × Factor Diversity Loss
           + λ_regime × Regime Detection Loss
```

**Regularization Techniques:**
- L2 regularization on attention weights
- Macro-factor diversity loss (negative entropy)
- Counterfactual regularization
- Knowledge distillation from simpler models

## 📊 Data Requirements

### Required Data

1. **OHLC Price Data**
   - Multiple timeframes (5m, 15m, 1H, 4H, Daily)
   - Minimum 1000 candles for training

2. **Economic Calendar**
   - Event type, time, actual, forecast, previous
   - Major events: NFP, CPI, GDP, FOMC, ECB, etc.

3. **Macro Indicators**
   - Interest rates (central bank rates)
   - Bond yields (2Y, 5Y, 10Y, 30Y)
   - Commodity prices (Gold, Oil, Copper)

4. **Labels** (for supervised training)
   - Future price direction
   - Generated from forward returns

### Data Format

```python
# OHLC DataFrame
ohlc = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...]
}, index=DatetimeIndex)

# Events DataFrame
events = pd.DataFrame({
    'event_type': [...],
    'time': [...],
    'actual': [...],
    'forecast': [...],
    'previous': [...]
})

# Macro Dictionary
macro_data = {
    'rates': DataFrame,
    'yields': DataFrame,
    'commodities': DataFrame,
    'fx': DataFrame
}
```

## 🎓 Key Concepts

### Regime-Aware Modeling

The model doesn't treat all market conditions the same. In risk-off regimes:
- Commodity correlations get more weight
- Macro signals dominate
- Price momentum gets less weight

### Multi-Modal Fusion

Different data streams are fused through cross-attention:
- **Price + Macro**: How price reacts to macro signals
- **Event + Price**: Pre/post event price adjustment
- **Commodity + FX**: Dynamic correlation modeling

### Hierarchical Reasoning

Mimics how professional traders think:
1. **Macro**: What's the fundamental direction? (CB policy, yields)
2. **Meso**: What's happening in this session? (Flows, risk sentiment)
3. **Micro**: When to enter/exit? (Price action confirmation)

### Explainability

Every prediction is traceable:
- Which factors contributed most?
- What regime was detected?
- What would happen in counterfactual scenarios?

## 🔬 Advanced Features

### Online Learning & Adaptation

```python
# Fast adaptation to new regimes
model.adapt(recent_data, learning_rate=1e-5, steps=10)
```

### Ensemble Predictions

```python
# Train multiple models
models = [create_mrat_model(config) for _ in range(5)]

# Ensemble prediction with uncertainty
predictions = [model(batch) for model in models]
mean_pred = torch.mean([p['logits'] for p in predictions])
uncertainty = torch.std([p['logits'] for p in predictions])
```

### Regime-Specific Performance

```python
# Evaluate by regime
from utils.training import evaluate_by_regime

metrics = evaluate_by_regime(model, val_loader)
print(f"Risk-On Accuracy: {metrics['risk_on_accuracy']}")
print(f"High-Vol Accuracy: {metrics['high_vol_accuracy']}")
```

## 📈 Performance Metrics

The model tracks:
- **Prediction Accuracy** (overall and regime-specific)
- **Confidence Calibration** (predicted confidence vs actual win rate)
- **Sharpe Ratio** (if trading)
- **Factor Attribution Stability** (consistency of explanations)

## 🚧 Future Enhancements

- [ ] Add sentiment analysis from news/social media
- [ ] Implement order book features (if available)
- [ ] Add multi-step predictions (forecast path, not just direction)
- [ ] Implement reinforcement learning for position sizing
- [ ] Add volatility forecasting module
- [ ] Create real-time inference API
- [ ] Build visualization dashboard

## 📖 Documentation

For detailed architecture documentation, see:
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Complete architecture specification

## 🤝 Contributing

This is a conceptual model for educational and research purposes. Contributions welcome:
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## ⚠️ Disclaimer

This model is for **educational and research purposes only**. 

- Not financial advice
- Past performance doesn't guarantee future results
- Forex trading involves substantial risk
- Always test thoroughly before live trading
- Use proper risk management

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

This architecture draws inspiration from:
- Transformer architectures (Vaswani et al.)
- Mixture of Experts (Shazeer et al.)
- Meta-Learning (Finn et al.)
- Explainable AI research
- Professional forex trading practices

## 📧 Contact

For questions or discussions about the architecture, please open an issue.

---

**Built with PyTorch** 🔥 | **Powered by Novel AI/ML Concepts** 🚀
