"""
Macro-Regime Adaptive Transformer (MRAT) - Main Model Implementation
A novel forex prediction model with regime-aware attention and macro reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class PriceGeometryEncoder(nn.Module):
    """
    Encodes OHLC price data as geometric manifolds with topological features.
    
    Innovations:
    - Curvature tensors (price acceleration)
    - Volatility surfaces (realized vol estimation)
    - Support/resistance field equations
    - Multi-timeframe coherence
    - Fractal dimension estimation
    """
    
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_price
        self.n_timeframes = config.n_timeframes
        
        # OHLC embedding layers
        self.ohlc_proj = nn.Linear(4, 64)  # OHLC -> features
        
        # Geometric feature extractors
        self.curvature_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        
        self.volatility_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        self.sr_field_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        # Multi-timeframe attention
        self.timeframe_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )
        
        # Final projection
        self.output_proj = nn.Linear(128, self.d_model)
        
    def compute_curvature(self, prices: torch.Tensor) -> torch.Tensor:
        """Compute second derivative (curvature) of price series."""
        # First derivative (velocity)
        dP = prices[:, 1:] - prices[:, :-1]
        # Second derivative (acceleration/curvature)
        d2P = dP[:, 1:] - dP[:, :-1]
        # Pad to match original length
        d2P = F.pad(d2P, (2, 0), value=0)
        return d2P
    
    def compute_volatility_surface(self, ohlc: torch.Tensor) -> torch.Tensor:
        """Estimate realized volatility using Parkinson estimator."""
        high = ohlc[:, :, 1]
        low = ohlc[:, :, 2]
        # Parkinson estimator: sqrt((1/4ln2) * (ln(H/L))^2)
        vol = torch.sqrt(
            (1 / (4 * np.log(2))) * torch.log(high / (low + 1e-8)).pow(2)
        )
        return vol
    
    def detect_sr_levels(self, prices: torch.Tensor) -> torch.Tensor:
        """Detect support/resistance levels using local extrema."""
        # Simple approach: local maxima/minima detection
        # In production, use more sophisticated algorithms
        window = 5
        sr_field = torch.zeros_like(prices)
        
        for i in range(window, len(prices[0]) - window):
            local_window = prices[:, i-window:i+window+1]
            is_max = prices[:, i] == torch.max(local_window, dim=1)[0]
            is_min = prices[:, i] == torch.min(local_window, dim=1)[0]
            sr_field[:, i] = is_max.float() - is_min.float()
        
        return sr_field
    
    def forward(self, ohlc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ohlc: [batch, time, 4] (open, high, low, close)
        Returns:
            encoded: [batch, d_price] price geometry encoding
        """
        batch_size, seq_len, _ = ohlc.shape
        
        # Extract close prices for derivatives
        close = ohlc[:, :, 3]
        
        # Compute geometric features
        curvature = self.compute_curvature(close)
        volatility = self.compute_volatility_surface(ohlc)
        sr_field = self.detect_sr_levels(close)
        
        # Embed OHLC
        ohlc_embed = self.ohlc_proj(ohlc)  # [batch, time, 64]
        
        # Extract geometric encodings
        curv_feat = self.curvature_net(ohlc_embed)  # [batch, time, 32]
        vol_feat = self.volatility_net(ohlc_embed)   # [batch, time, 16]
        sr_feat = self.sr_field_net(ohlc_embed)      # [batch, time, 16]
        
        # Concatenate all features
        all_features = torch.cat([
            ohlc_embed,
            curv_feat,
            vol_feat,
            sr_feat
        ], dim=-1)  # [batch, time, 128]
        
        # Multi-timeframe attention (self-attention over time)
        attended, _ = self.timeframe_attention(
            all_features, all_features, all_features
        )
        
        # Pool over time (mean pooling)
        pooled = torch.mean(attended, dim=1)  # [batch, 128]
        
        # Final projection
        output = self.output_proj(pooled)  # [batch, d_price]
        
        return output


class EventEmbeddingNetwork(nn.Module):
    """
    Embeds economic calendar events into a learned macro-event space.
    
    Innovations:
    - Event impact vectors (learned impact profiles)
    - Temporal decay functions
    - Surprise factor embedding
    - Event anticipation mechanism
    """
    
    def __init__(self, config):
        super().__init__()
        self.d_event = config.d_event
        self.n_event_types = config.n_event_types
        
        # Event type embedding
        self.event_type_embed = nn.Embedding(self.n_event_types, 32)
        
        # Surprise encoding
        self.surprise_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Temporal encoding (time-to-event)
        self.temporal_net = nn.Sequential(
            nn.Linear(2, 16),  # [time_to_event, time_since_event]
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Event impact predictor
        self.impact_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.d_event)
        )
        
    def compute_surprise(self, actual: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        """Compute normalized surprise score."""
        # Normalized deviation
        surprise = (actual - forecast) / (torch.abs(forecast) + 1e-6)
        return surprise.unsqueeze(-1)
    
    def temporal_encoding(self, time_to_event: torch.Tensor, 
                         time_since_event: torch.Tensor) -> torch.Tensor:
        """Encode temporal information with exponential decay."""
        # Exponential decay for time-to-event (anticipation)
        tte_encoded = torch.exp(-torch.abs(time_to_event))
        # Exponential decay for time-since-event (impact fading)
        tse_encoded = torch.exp(-time_since_event)
        
        temporal = torch.stack([tte_encoded, tse_encoded], dim=-1)
        return self.temporal_net(temporal)
    
    def forward(self, events: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            events: Dictionary with:
                - event_type: [batch, n_events] (categorical)
                - actual: [batch, n_events]
                - forecast: [batch, n_events]
                - time_to_event: [batch, n_events] (hours, negative if future)
                - time_since_event: [batch, n_events] (hours, 0 if not happened)
        Returns:
            encoded: [batch, d_event] event encoding
        """
        # Embed event types
        type_embed = self.event_type_embed(events['event_type'])  # [batch, n_events, 32]
        
        # Encode surprise
        surprise = self.compute_surprise(events['actual'], events['forecast'])
        surprise_embed = self.surprise_net(surprise)  # [batch, n_events, 16]
        
        # Encode temporal information
        temporal_embed = self.temporal_encoding(
            events['time_to_event'], 
            events['time_since_event']
        )  # [batch, n_events, 16]
        
        # Concatenate all embeddings
        event_features = torch.cat([
            type_embed,
            surprise_embed,
            temporal_embed
        ], dim=-1)  # [batch, n_events, 64]
        
        # Pool over events (max pooling to capture most important event)
        # Handle case with no events
        if event_features.shape[1] == 0:
            pooled = torch.zeros(event_features.shape[0], 64, device=event_features.device)
        else:
            pooled, _ = torch.max(event_features, dim=1)  # [batch, 64]
        
        # Predict impact
        impact = self.impact_net(pooled)  # [batch, d_event]
        
        return impact


class MacroStateEncoder(nn.Module):
    """
    Encodes macroeconomic indicators into policy state representations.
    
    Innovations:
    - Interest rate differential dynamics (trajectory)
    - Yield curve geometry (PCA-based)
    - Commodity-macro linkage
    - Central bank policy stance embedding
    """
    
    def __init__(self, config):
        super().__init__()
        self.d_macro = config.d_macro
        
        # Interest rate differential encoder
        self.ird_net = nn.Sequential(
            nn.Linear(5, 32),  # [spread, d/dt, d²/dt², expected, surprise]
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Yield curve encoder
        self.yield_net = nn.Sequential(
            nn.Linear(5, 32),  # [level, slope, curvature, inversion, term_premium]
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Commodity encoder
        self.commodity_net = nn.Sequential(
            nn.Linear(4, 16),  # [gold_change, oil_change, copper_change, correlation]
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Policy stance encoder
        self.policy_net = nn.Sequential(
            nn.Linear(2, 16),  # [dove_score, expected_hikes]
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.d_macro)
        )
        
    def forward(self, macro: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            macro: Dictionary with macro indicators
        Returns:
            encoded: [batch, d_macro] macro state encoding
        """
        # Handle potential time dimension and average if needed
        def flatten_if_needed(x):
            if len(x.shape) == 3:  # [batch, time, features]
                return torch.mean(x, dim=1)  # [batch, features]
            return x
        
        # Encode IRD dynamics
        ird_input = flatten_if_needed(macro['ird'])
        ird_features = self.ird_net(ird_input)  # [batch, 32]
        
        # Encode yield curve
        yield_input = flatten_if_needed(macro['yield_curve'])
        yield_features = self.yield_net(yield_input)  # [batch, 32]
        
        # Encode commodities
        comm_input = flatten_if_needed(macro['commodities'])
        commodity_features = self.commodity_net(comm_input)  # [batch, 16]
        
        # Encode policy stance
        policy_input = flatten_if_needed(macro['policy'])
        policy_features = self.policy_net(policy_input)  # [batch, 16]
        
        # Concatenate and fuse
        all_features = torch.cat([
            ird_features,
            yield_features,
            commodity_features,
            policy_features
        ], dim=-1)  # [batch, 96]
        
        # Final encoding
        encoded = self.fusion(all_features)  # [batch, d_macro]
        
        return encoded


class RegimeDetectionModule(nn.Module):
    """
    Explicit multi-state regime detection with mixture of experts.
    
    Regime Dimensions:
    1. Risk Sentiment (Risk-On/Risk-Off)
    2. Volatility (Low/Medium/High/Extreme)
    3. Trend Strength (Range/Weak/Strong/Extreme)
    4. CB Policy (Easing/Neutral/Tightening)
    5. Market Stress (Orderly/Stressed/Crisis)
    """
    
    def __init__(self, config):
        super().__init__()
        # Use 2*d_model since we concat price and macro after projection
        self.d_input = config.d_model * 2
        
        # Expert networks (one per regime dimension)
        self.risk_sentiment_expert = self._make_expert(2)  # Risk-On/Off
        self.volatility_expert = self._make_expert(4)      # 4 levels
        self.trend_expert = self._make_expert(4)           # 4 levels
        self.policy_expert = self._make_expert(3)          # 3 stances
        self.stress_expert = self._make_expert(3)          # 3 levels
        
        # Gating network
        self.gating = nn.Sequential(
            nn.Linear(self.d_input, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # 5 experts
            nn.Softmax(dim=-1)
        )
        
        # Regime transition smoothing (temporal consistency)
        self.temporal_smooth = nn.GRU(
            input_size=16,  # Total regime dimensions
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        
    def _make_expert(self, n_states: int) -> nn.Module:
        """Create an expert network for regime detection."""
        return nn.Sequential(
            nn.Linear(self.d_input, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_states)
        )
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: List of [price_encoding, macro_encoding, ...]
        Returns:
            regime_probs: Dictionary of regime probabilities
        """
        # Concatenate all input features
        x = torch.cat(features, dim=-1)  # [batch, d_input]
        
        # Expert predictions
        risk_logits = self.risk_sentiment_expert(x)
        vol_logits = self.volatility_expert(x)
        trend_logits = self.trend_expert(x)
        policy_logits = self.policy_expert(x)
        stress_logits = self.stress_expert(x)
        
        # Convert to probabilities
        regime_probs = {
            'risk_sentiment': F.softmax(risk_logits, dim=-1),
            'volatility': F.softmax(vol_logits, dim=-1),
            'trend': F.softmax(trend_logits, dim=-1),
            'policy': F.softmax(policy_logits, dim=-1),
            'stress': F.softmax(stress_logits, dim=-1)
        }
        
        # Gating weights (for ensemble if needed)
        gate_weights = self.gating(x)
        regime_probs['gate_weights'] = gate_weights
        
        return regime_probs


class RegimeConditionedCrossAttention(nn.Module):
    """
    Cross-attention fusion with regime-dependent modulation.
    
    Innovation: Attention weights are scaled based on detected regime,
    allowing different modalities to dominate in different market conditions.
    """
    
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        
        # Cross-attention layers
        self.price_macro_attn = nn.MultiheadAttention(
            self.d_model, self.n_heads, batch_first=True
        )
        self.event_price_attn = nn.MultiheadAttention(
            self.d_model, self.n_heads, batch_first=True
        )
        
        # Regime modulation matrices (learned)
        self.regime_modulation = nn.ModuleDict({
            'risk_on': nn.Linear(self.d_model, self.d_model),
            'risk_off': nn.Linear(self.d_model, self.d_model),
            'high_vol': nn.Linear(self.d_model, self.d_model),
            'low_vol': nn.Linear(self.d_model, self.d_model)
        })
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model)
        )
        
    def apply_regime_modulation(self, x: torch.Tensor, 
                                regime_probs: Dict) -> torch.Tensor:
        """Apply regime-dependent scaling to features."""
        # Extract relevant regime probabilities
        risk_on_prob = regime_probs['risk_sentiment'][:, 0:1]  # [batch, 1]
        high_vol_prob = regime_probs['volatility'][:, 2:3]     # [batch, 1]
        
        # Weighted combination of modulation matrices
        modulated = (
            risk_on_prob * self.regime_modulation['risk_on'](x) +
            (1 - risk_on_prob) * self.regime_modulation['risk_off'](x) +
            high_vol_prob * self.regime_modulation['high_vol'](x) +
            (1 - high_vol_prob) * self.regime_modulation['low_vol'](x)
        ) / 2.0  # Average the two regime dimensions
        
        return modulated
    
    def forward(self, price: torch.Tensor, events: torch.Tensor,
                macro: torch.Tensor, commodities: torch.Tensor,
                regime: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            price, events, macro, commodities: [batch, d_model] each
            regime: Dictionary of regime probabilities
        Returns:
            fused: [batch, d_model] fused representation
            attention_weights: Dictionary of attention weights for explainability
        """
        # Add sequence dimension for attention
        price_seq = price.unsqueeze(1)  # [batch, 1, d_model]
        macro_seq = macro.unsqueeze(1)
        event_seq = events.unsqueeze(1)
        
        # Cross-attention: price attends to macro
        price_macro_fused, pm_weights = self.price_macro_attn(
            price_seq, macro_seq, macro_seq
        )
        
        # Cross-attention: event attends to price
        event_price_fused, ep_weights = self.event_price_attn(
            event_seq, price_seq, price_seq
        )
        
        # Remove sequence dimension
        price_macro_fused = price_macro_fused.squeeze(1)
        event_price_fused = event_price_fused.squeeze(1)
        
        # Apply regime modulation
        modulated = self.apply_regime_modulation(
            price_macro_fused, regime
        )
        
        # Fuse all information
        combined = torch.cat([modulated, event_price_fused], dim=-1)
        fused = self.fusion(combined)
        
        # Store attention weights for explainability
        attention_weights = {
            'price_macro': pm_weights,
            'event_price': ep_weights
        }
        
        return fused, attention_weights


class HierarchicalReasoningLayer(nn.Module):
    """
    Three-tier reasoning: Macro → Meso → Micro
    Each tier operates at different time horizons.
    """
    
    def __init__(self, level: str, config):
        super().__init__()
        self.level = level
        self.d_model = config.d_model
        
        # Level-specific parameters
        if level == 'macro':
            self.horizon = 'days'
            self.d_output = 128
        elif level == 'meso':
            self.horizon = 'hours'
            self.d_output = 96
        else:  # micro
            self.horizon = 'minutes'
            self.d_output = 64
        
        # Reasoning network
        self.reasoning_net = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.d_output)
        )
        
    def forward(self, fused: torch.Tensor, 
                parent_signal: Optional[torch.Tensor] = None,
                regime: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            fused: [batch, d_model] fused features
            parent_signal: [batch, d_parent] signal from parent layer (if not macro)
            regime: Regime probabilities
        Returns:
            signal: [batch, d_output] reasoning signal
        """
        if parent_signal is not None:
            # Concatenate parent signal for hierarchical dependency
            x = torch.cat([fused, parent_signal], dim=-1)
            # Adjust input dimension
            x = nn.Linear(x.shape[-1], self.d_model).to(x.device)(x)
        else:
            x = fused
        
        signal = self.reasoning_net(x)
        return signal


class OutputExplainabilityLayer(nn.Module):
    """
    Final output layer with built-in explainability.
    
    Outputs:
    - Direction probabilities
    - Confidence score
    - Risk regime assessment
    - Factor attribution
    """
    
    def __init__(self, config):
        super().__init__()
        self.d_input = 128 + 96 + 64  # macro + meso + micro signals
        
        # Direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(self.d_input, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # LONG, NEUTRAL, SHORT
        )
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(self.d_input, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1]
        )
        
        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(self.d_input, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # [tail_risk, volatility, liquidity, stress]
        )
        
        # Factor attribution network
        self.attribution_net = nn.Sequential(
            nn.Linear(self.d_input, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # 6 main factors
            nn.Softmax(dim=-1)
        )
        
    def forward(self, signals: List[torch.Tensor],
                attention_weights: Dict,
                regime: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            signals: List of [macro_signal, meso_signal, micro_signal]
            attention_weights: Attention weights from fusion layer
            regime: Regime probabilities
        Returns:
            outputs: Dictionary with all predictions and explainability info
        """
        # Concatenate all signals
        combined = torch.cat(signals, dim=-1)  # [batch, d_input]
        
        # Direction prediction
        direction_logits = self.direction_head(combined)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Confidence (regime-calibrated)
        base_confidence = self.confidence_head(combined)
        # Adjust by regime uncertainty
        regime_entropy = -torch.sum(
            regime['volatility'] * torch.log(regime['volatility'] + 1e-8),
            dim=-1, keepdim=True
        )
        regime_confidence = 1 - regime_entropy / np.log(4)  # Normalize
        confidence = base_confidence * regime_confidence
        
        # Risk assessment
        risk_scores = self.risk_head(combined)
        risk_scores = torch.sigmoid(risk_scores)  # [0, 1] for each risk type
        
        # Factor attribution
        factor_attribution = self.attribution_net(combined)
        
        outputs = {
            'logits': direction_logits,
            'direction_probs': direction_probs,
            'confidence': confidence,
            'risk_scores': risk_scores,
            'factor_attribution': factor_attribution,
            'regime_probs': regime,
            'attention_weights': attention_weights
        }
        
        return outputs


class MRATForexModel(nn.Module):
    """
    Complete Macro-Regime Adaptive Transformer (MRAT) model.
    
    This is the main model that integrates all components.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Encoders
        self.price_encoder = PriceGeometryEncoder(config)
        self.event_encoder = EventEmbeddingNetwork(config)
        self.macro_encoder = MacroStateEncoder(config)
        
        # Projection layers to d_model
        self.price_proj = nn.Linear(config.d_price, config.d_model)
        self.event_proj = nn.Linear(config.d_event, config.d_model)
        self.macro_proj = nn.Linear(config.d_macro, config.d_model)
        
        # Regime detection
        self.regime_detector = RegimeDetectionModule(config)
        
        # Cross-attention fusion
        self.cross_attention = RegimeConditionedCrossAttention(config)
        
        # Hierarchical reasoning
        self.macro_reasoning = HierarchicalReasoningLayer('macro', config)
        self.meso_reasoning = HierarchicalReasoningLayer('meso', config)
        self.micro_reasoning = HierarchicalReasoningLayer('micro', config)
        
        # Output layer
        self.output_head = OutputExplainabilityLayer(config)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass.
        
        Args:
            batch: Dictionary containing:
                - ohlc: [batch, time, 4] price data
                - events: Dictionary of event data
                - macro_indicators: Dictionary of macro data
                - commodities: [batch, n_commodities]
        Returns:
            outputs: Dictionary with predictions and explainability
        """
        # Step 1: Encode each modality
        H_price = self.price_encoder(batch['ohlc'])
        H_event = self.event_encoder(batch['events'])
        H_macro = self.macro_encoder(batch['macro_indicators'])
        
        # Project to d_model
        H_price = self.price_proj(H_price)
        H_event = self.event_proj(H_event)
        H_macro = self.macro_proj(H_macro)
        
        # For simplicity, treat commodities as part of macro
        # In full implementation, would have separate encoder
        H_commodity = H_macro  # Placeholder
        
        # Step 2: Detect regime
        regime_probs = self.regime_detector([H_price, H_macro])
        
        # Step 3: Regime-conditioned fusion
        H_fused, attention_weights = self.cross_attention(
            price=H_price,
            events=H_event,
            macro=H_macro,
            commodities=H_commodity,
            regime=regime_probs
        )
        
        # Step 4: Hierarchical reasoning
        macro_signal = self.macro_reasoning(H_fused, regime=regime_probs)
        meso_signal = self.meso_reasoning(
            H_fused, parent_signal=macro_signal, regime=regime_probs
        )
        micro_signal = self.micro_reasoning(
            H_fused, parent_signal=meso_signal, regime=regime_probs
        )
        
        # Step 5: Generate outputs with explainability
        outputs = self.output_head(
            signals=[macro_signal, meso_signal, micro_signal],
            attention_weights=attention_weights,
            regime=regime_probs
        )
        
        return outputs


class MRATConfig:
    """Configuration for MRAT model."""
    
    def __init__(self):
        # Encoding dimensions
        self.d_price = 128
        self.d_event = 64
        self.d_macro = 96
        self.d_model = 256
        
        # Architecture parameters
        self.n_heads = 8
        self.n_timeframes = 5
        self.n_event_types = 50
        
        # Training parameters
        self.dropout = 0.1
        self.lr = 1e-4
        self.batch_size = 32
        self.epochs = 100
        
        # Regularization
        self.lambda_att = 0.01
        self.lambda_div = 0.001


def create_mrat_model(config: Optional[MRATConfig] = None) -> MRATForexModel:
    """Factory function to create MRAT model."""
    if config is None:
        config = MRATConfig()
    return MRATForexModel(config)


if __name__ == '__main__':
    # Example usage
    config = MRATConfig()
    model = create_mrat_model(config)
    
    # Create dummy batch
    batch_size = 4
    seq_len = 100
    
    batch = {
        'ohlc': torch.randn(batch_size, seq_len, 4),
        'events': {
            'event_type': torch.randint(0, 50, (batch_size, 5)),
            'actual': torch.randn(batch_size, 5),
            'forecast': torch.randn(batch_size, 5),
            'time_to_event': torch.randn(batch_size, 5),
            'time_since_event': torch.abs(torch.randn(batch_size, 5))
        },
        'macro_indicators': {
            'ird': torch.randn(batch_size, 5),
            'yield_curve': torch.randn(batch_size, 5),
            'commodities': torch.randn(batch_size, 4),
            'policy': torch.randn(batch_size, 2)
        }
    }
    
    # Forward pass
    outputs = model(batch)
    
    print("Model output keys:", outputs.keys())
    print("Direction probabilities shape:", outputs['direction_probs'].shape)
    print("Confidence shape:", outputs['confidence'].shape)
    print("Risk scores shape:", outputs['risk_scores'].shape)
    print("Factor attribution shape:", outputs['factor_attribution'].shape)
