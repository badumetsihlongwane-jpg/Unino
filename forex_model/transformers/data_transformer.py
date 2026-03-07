"""
Data Transformation Pipeline for MRAT Forex Model
Handles transformation of raw forex data into model-ready features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


@dataclass
class TransformedData:
    """Container for transformed forex data."""
    ohlc: np.ndarray
    events: Dict[str, np.ndarray]
    macro_indicators: Dict[str, np.ndarray]
    timestamps: np.ndarray
    metadata: Dict


class PriceDataTransformer:
    """
    Transforms raw OHLC price data into geometric features.
    
    Features:
    - Normalized prices (log-returns)
    - Multi-horizon momentum
    - Realized volatility
    - Volume profile (if available)
    - Geometric features (curvature, fractal dimension)
    """
    
    def __init__(self, window_size: int = 100, timeframes: List[str] = None):
        self.window_size = window_size
        self.timeframes = timeframes or ['5m', '15m', '1H', '4H', 'D']
        self.scaler = StandardScaler()
        
    def compute_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """Compute log returns for stationarity."""
        log_prices = np.log(prices + 1e-8)
        returns = np.diff(log_prices)
        # Pad to match original length
        returns = np.concatenate([[returns[0]], returns])
        return returns
    
    def compute_momentum(self, prices: np.ndarray, 
                        horizons: List[int] = None) -> Dict[str, np.ndarray]:
        """Compute multi-horizon momentum."""
        if horizons is None:
            horizons = [5, 10, 20, 50, 100]  # Different lookback periods
        
        momentum_features = {}
        for h in horizons:
            if len(prices) > h:
                momentum = prices[h:] - prices[:-h]
                # Pad to match original length
                momentum = np.concatenate([
                    np.zeros(h),
                    momentum
                ])
                momentum_features[f'mom_{h}'] = momentum
            else:
                momentum_features[f'mom_{h}'] = np.zeros_like(prices)
        
        return momentum_features
    
    def compute_realized_volatility(self, ohlc: np.ndarray, 
                                   window: int = 20) -> np.ndarray:
        """
        Compute realized volatility using multiple estimators.
        
        Estimators:
        - Parkinson: Uses high-low range
        - Garman-Klass: Uses OHLC
        - Rogers-Satchell: For trending markets
        """
        high = ohlc[:, 1]
        low = ohlc[:, 2]
        close = ohlc[:, 3]
        open_price = ohlc[:, 0]
        
        # Parkinson estimator
        parkinson_vol = np.sqrt(
            (1 / (4 * np.log(2))) * np.log(high / (low + 1e-8))**2
        )
        
        # Garman-Klass estimator
        gk_vol = np.sqrt(
            0.5 * np.log(high / (low + 1e-8))**2 - 
            (2 * np.log(2) - 1) * np.log(close / (open_price + 1e-8))**2
        )
        
        # Rolling average
        vol = (parkinson_vol + gk_vol) / 2
        vol_rolled = pd.Series(vol).rolling(window, min_periods=1).mean().values
        
        return vol_rolled
    
    def compute_fractal_dimension(self, prices: np.ndarray, 
                                  window: int = 50) -> np.ndarray:
        """
        Estimate fractal dimension using variation ratio.
        Higher values indicate more complex, non-trending patterns.
        """
        fractal_dims = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            segment = prices[i-window:i]
            
            # Compute variation at different scales
            scales = [2, 4, 8, 16]
            variations = []
            
            for scale in scales:
                if window // scale > 1:
                    # Downsample
                    downsampled = segment[::scale]
                    # Compute total variation
                    variation = np.sum(np.abs(np.diff(downsampled)))
                    variations.append(variation / len(downsampled))
            
            if len(variations) > 1:
                # Estimate fractal dimension from log-log slope
                log_scales = np.log(scales[:len(variations)])
                log_vars = np.log(np.array(variations) + 1e-8)
                # Simple linear regression
                slope = np.polyfit(log_scales, log_vars, 1)[0]
                fractal_dim = 2 - slope  # Hurst exponent conversion
                fractal_dims[i] = np.clip(fractal_dim, 1.0, 2.0)
            else:
                fractal_dims[i] = 1.5  # Default
        
        return fractal_dims
    
    def detect_support_resistance(self, prices: np.ndarray,
                                  window: int = 20,
                                  threshold: float = 0.002) -> np.ndarray:
        """
        Detect support/resistance levels using local extrema.
        Returns a field indicating proximity to S/R levels.
        """
        sr_field = np.zeros(len(prices))
        
        # Find local maxima (resistance)
        for i in range(window, len(prices) - window):
            local_max = np.max(prices[i-window:i+window+1])
            local_min = np.min(prices[i-window:i+window+1])
            
            if prices[i] == local_max:
                sr_field[i] = 1.0  # Resistance
            elif prices[i] == local_min:
                sr_field[i] = -1.0  # Support
            else:
                # Distance to nearest S/R
                dist_to_max = abs(prices[i] - local_max) / (prices[i] + 1e-8)
                dist_to_min = abs(prices[i] - local_min) / (prices[i] + 1e-8)
                
                if dist_to_max < threshold:
                    sr_field[i] = 0.5  # Near resistance
                elif dist_to_min < threshold:
                    sr_field[i] = -0.5  # Near support
        
        return sr_field
    
    def transform(self, ohlc: pd.DataFrame) -> np.ndarray:
        """
        Transform raw OHLC data into model features.
        
        Args:
            ohlc: DataFrame with columns ['open', 'high', 'low', 'close']
        Returns:
            transformed: numpy array [time, features]
        """
        # Convert to numpy
        ohlc_np = ohlc[['open', 'high', 'low', 'close']].values
        close = ohlc['close'].values
        
        # Compute all features
        log_returns = self.compute_log_returns(close)
        momentum_features = self.compute_momentum(close)
        volatility = self.compute_realized_volatility(ohlc_np)
        fractal_dim = self.compute_fractal_dimension(close)
        sr_field = self.detect_support_resistance(close)
        
        # Normalize OHLC (z-score with rolling window)
        normalized_ohlc = self.scaler.fit_transform(ohlc_np)
        
        return normalized_ohlc


class EconomicCalendarTransformer:
    """
    Transforms economic calendar events into embeddings.
    
    Features:
    - Event type categorization
    - Surprise calculation (actual vs forecast)
    - Temporal encoding (time-to-event, time-since-event)
    - Historical impact estimation
    """
    
    def __init__(self):
        self.event_types = self._initialize_event_types()
        self.impact_profiles = self._initialize_impact_profiles()
        
    def _initialize_event_types(self) -> Dict[str, int]:
        """Map event names to categorical IDs."""
        events = [
            'NFP', 'CPI', 'GDP', 'FOMC', 'ECB', 'BOJ', 'BOE',
            'PMI', 'Retail_Sales', 'Unemployment', 'PPI',
            'Housing_Starts', 'Consumer_Confidence', 'Trade_Balance',
            'Industrial_Production', 'Durable_Goods', 'ADP',
            'ISM_Manufacturing', 'ISM_Services', 'PCE',
            'Jobs_Report', 'Jobless_Claims', 'Consumer_Spending',
            'Business_Confidence', 'Inflation', 'Interest_Rate_Decision',
            # ... (50 total event types)
        ]
        return {event: i for i, event in enumerate(events)}
    
    def _initialize_impact_profiles(self) -> Dict[str, Dict]:
        """
        Historical impact profiles for each event type.
        Based on typical market reactions.
        """
        return {
            'NFP': {
                'immediate_impact': 0.8,
                'decay_hours': 24,
                'pairs_affected': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
                'typical_move_bps': 50
            },
            'FOMC': {
                'immediate_impact': 1.0,
                'decay_hours': 48,
                'pairs_affected': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
                'typical_move_bps': 100
            },
            'CPI': {
                'immediate_impact': 0.7,
                'decay_hours': 12,
                'pairs_affected': ['EUR/USD', 'GBP/USD'],
                'typical_move_bps': 30
            },
            # ... (profiles for all event types)
        }
    
    def compute_surprise(self, actual: float, forecast: float,
                        historical_std: float = 1.0) -> float:
        """
        Calculate normalized surprise score.
        
        Surprise = (Actual - Forecast) / Historical_StdDev
        """
        if forecast == 0:
            return 0.0
        
        raw_surprise = (actual - forecast) / abs(forecast)
        normalized = raw_surprise / historical_std
        
        return np.clip(normalized, -3, 3)  # Clip to ±3 sigma
    
    def temporal_encoding(self, time_to_event_hours: float) -> Tuple[float, float]:
        """
        Encode temporal information with exponential decay.
        
        Returns:
            (anticipation_score, impact_score)
        """
        if time_to_event_hours > 0:  # Event is in future
            # Anticipation builds as event approaches
            anticipation = np.exp(-time_to_event_hours / 24)  # Decay over 24h
            impact = 0.0
        else:  # Event already happened (negative or zero)
            # Impact decays over time
            anticipation = 0.0
            impact = np.exp(time_to_event_hours / 12)  # Decay over 12h (time_to_event is negative)
        
        return anticipation, impact
    
    def transform(self, events_df: pd.DataFrame,
                 current_time: pd.Timestamp) -> Dict[str, np.ndarray]:
        """
        Transform economic calendar into model features.
        
        Args:
            events_df: DataFrame with columns:
                ['event_type', 'time', 'actual', 'forecast', 'previous']
            current_time: Reference timestamp for temporal encoding
        Returns:
            Dictionary with transformed event features
        """
        n_events = len(events_df)
        
        # Initialize arrays
        event_type_ids = np.zeros(n_events, dtype=int)
        surprises = np.zeros(n_events)
        time_to_events = np.zeros(n_events)
        time_since_events = np.zeros(n_events)
        
        for idx, row in enumerate(events_df.itertuples()):
            # Event type encoding
            event_name = row.event_type
            event_type_ids[idx] = self.event_types.get(event_name, 0)
            
            # Surprise calculation
            if pd.notna(row.actual) and pd.notna(row.forecast):
                surprises[idx] = self.compute_surprise(
                    row.actual, row.forecast
                )
            
            # Temporal encoding
            time_delta = (row.time - current_time).total_seconds() / 3600  # hours
            if time_delta > 0:  # Future event
                time_to_events[idx] = time_delta
                time_since_events[idx] = 0
            else:  # Past event (negative or zero time_delta)
                time_to_events[idx] = 0
                time_since_events[idx] = -time_delta  # Make positive
        
        return {
            'event_type': event_type_ids,
            'actual': events_df['actual'].fillna(0).values,
            'forecast': events_df['forecast'].fillna(0).values,
            'time_to_event': time_to_events,
            'time_since_event': time_since_events
        }


class MacroIndicatorTransformer:
    """
    Transforms macroeconomic indicators into policy state features.
    
    Features:
    - Interest rate differential dynamics
    - Yield curve geometry (PCA)
    - Commodity signals
    - Central bank policy stance
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def compute_ird_features(self, rate_a: pd.Series,
                            rate_b: pd.Series) -> np.ndarray:
        """
        Compute interest rate differential features.
        
        Returns:
            [spread, d/dt, d²/dt², expected, surprise]
        """
        # Current spread
        spread = (rate_a - rate_b).values
        
        # First derivative (velocity)
        velocity = np.diff(spread, prepend=spread[0])
        
        # Second derivative (acceleration)
        acceleration = np.diff(velocity, prepend=velocity[0])
        
        # Expected spread (placeholder - would use futures curve)
        expected = spread  # Simplified
        
        # Surprise
        surprise = spread - expected
        
        return np.column_stack([
            spread, velocity, acceleration, expected, surprise
        ])
    
    def compute_yield_curve_pca(self, yields: pd.DataFrame) -> np.ndarray:
        """
        Compute yield curve principal components.
        
        Args:
            yields: DataFrame with columns for different maturities
                    (e.g., '2Y', '5Y', '10Y', '30Y')
        Returns:
            [level, slope, curvature, inversion_flag, term_premium]
        """
        # Simple approximation of PCA components
        if '2Y' in yields.columns and '10Y' in yields.columns:
            level = yields.mean(axis=1).values
            slope = (yields['10Y'] - yields['2Y']).values
            
            # Curvature (belly vs wings)
            if '5Y' in yields.columns:
                curvature = (
                    2 * yields['5Y'] - yields['2Y'] - yields['10Y']
                ).values
            else:
                curvature = np.zeros(len(yields))
            
            # Inversion flag
            inversion = (yields['2Y'] > yields['10Y']).astype(float).values
            
            # Term premium (simplified)
            term_premium = slope / (level + 1e-8)
            
            return np.column_stack([
                level, slope, curvature, inversion, term_premium
            ])
        else:
            # Fallback if columns missing
            return np.zeros((len(yields), 5))
    
    def compute_commodity_signals(self, gold: pd.Series,
                                 oil: pd.Series,
                                 copper: pd.Series,
                                 fx_pair: pd.Series) -> np.ndarray:
        """
        Compute commodity-FX correlation signals.
        
        Returns:
            [gold_change, oil_change, copper_change, fx_correlation]
        """
        # Price changes (% returns)
        gold_change = gold.pct_change().fillna(0).values
        oil_change = oil.pct_change().fillna(0).values
        copper_change = copper.pct_change().fillna(0).values
        
        # Rolling correlation with FX pair
        fx_returns = fx_pair.pct_change().fillna(0)
        
        # Correlation with gold (20-period rolling)
        gold_corr = fx_returns.rolling(20).corr(gold.pct_change()).fillna(0).values
        
        return np.column_stack([
            gold_change, oil_change, copper_change, gold_corr
        ])
    
    def encode_policy_stance(self, cb_statements: List[str],
                            rate_trajectory: pd.Series) -> np.ndarray:
        """
        Encode central bank policy stance.
        
        Uses sentiment analysis on CB statements + rate trajectory.
        
        Returns:
            [dove_score, expected_hikes]
        """
        # Simplified: Use rate trajectory as proxy
        # In production, would use NLP on CB statements
        
        rate_change = rate_trajectory.diff().fillna(0).values
        
        # Dove score: negative if cutting, positive if hiking
        dove_score = -np.sign(rate_change)
        
        # Expected hikes (based on recent trajectory)
        expected_hikes = np.cumsum(rate_change > 0.1)  # Count significant hikes
        
        return np.column_stack([dove_score, expected_hikes])
    
    def transform(self, macro_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Transform all macro indicators.
        
        Args:
            macro_data: Dictionary containing:
                - 'rates': DataFrame with interest rates
                - 'yields': DataFrame with bond yields
                - 'commodities': DataFrame with commodity prices
                - 'fx': DataFrame with FX pair prices
        Returns:
            Dictionary with transformed features
        """
        transformed = {}
        
        # Interest rate differentials
        if 'rates' in macro_data:
            rates = macro_data['rates']
            if 'rate_a' in rates.columns and 'rate_b' in rates.columns:
                transformed['ird'] = self.compute_ird_features(
                    rates['rate_a'], rates['rate_b']
                )
        
        # Yield curve
        if 'yields' in macro_data:
            transformed['yield_curve'] = self.compute_yield_curve_pca(
                macro_data['yields']
            )
        
        # Commodities
        if 'commodities' in macro_data and 'fx' in macro_data:
            comm = macro_data['commodities']
            transformed['commodities'] = self.compute_commodity_signals(
                comm['gold'], comm['oil'], comm['copper'],
                macro_data['fx']['close']
            )
        
        # Policy stance
        if 'rates' in macro_data:
            transformed['policy'] = self.encode_policy_stance(
                [], macro_data['rates']['rate_a']
            )
        
        return transformed


class DataPipeline:
    """
    Complete data transformation pipeline for MRAT model.
    Orchestrates all transformers and produces model-ready data.
    """
    
    def __init__(self):
        self.price_transformer = PriceDataTransformer()
        self.event_transformer = EconomicCalendarTransformer()
        self.macro_transformer = MacroIndicatorTransformer()
        
    def prepare_training_data(self,
                             ohlc: pd.DataFrame,
                             events: pd.DataFrame,
                             macro_data: Dict[str, pd.DataFrame],
                             window_size: int = 100) -> List[TransformedData]:
        """
        Prepare data for model training with sliding windows.
        
        Args:
            ohlc: Price data
            events: Economic calendar
            macro_data: Macro indicators
            window_size: Lookback window
        Returns:
            List of TransformedData objects (one per window)
        """
        samples = []
        
        # Ensure data is aligned by timestamp
        ohlc = ohlc.sort_index()
        
        # Create sliding windows
        for i in range(window_size, len(ohlc)):
            window_ohlc = ohlc.iloc[i-window_size:i]
            current_time = ohlc.index[i]
            
            # Transform price data
            ohlc_transformed = self.price_transformer.transform(window_ohlc)
            
            # Get relevant events (within ±48 hours)
            time_window_start = current_time - pd.Timedelta(hours=48)
            time_window_end = current_time + pd.Timedelta(hours=48)
            relevant_events = events[
                (events['time'] >= time_window_start) &
                (events['time'] <= time_window_end)
            ]
            
            # Transform events
            events_transformed = self.event_transformer.transform(
                relevant_events, current_time
            )
            
            # Transform macro data
            macro_window = {
                key: df.iloc[i-window_size:i]
                for key, df in macro_data.items()
            }
            macro_transformed = self.macro_transformer.transform(macro_window)
            
            # Create sample
            sample = TransformedData(
                ohlc=ohlc_transformed,
                events=events_transformed,
                macro_indicators=macro_transformed,
                timestamps=window_ohlc.index.values,
                metadata={'current_time': current_time}
            )
            
            samples.append(sample)
        
        return samples


def example_usage():
    """Demonstrate data transformation pipeline."""
    # Create dummy data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    # OHLC data
    np.random.seed(42)
    close = 1.0 + np.cumsum(np.random.randn(1000) * 0.001)
    ohlc = pd.DataFrame({
        'open': close * (1 + np.random.randn(1000) * 0.0005),
        'high': close * (1 + abs(np.random.randn(1000)) * 0.001),
        'low': close * (1 - abs(np.random.randn(1000)) * 0.001),
        'close': close
    }, index=dates)
    
    # Events
    events = pd.DataFrame({
        'event_type': ['NFP', 'CPI', 'FOMC'] * 10,
        'time': pd.date_range('2023-01-05', periods=30, freq='D'),
        'actual': np.random.randn(30),
        'forecast': np.random.randn(30),
        'previous': np.random.randn(30)
    })
    
    # Macro data
    macro_data = {
        'rates': pd.DataFrame({
            'rate_a': 2.5 + np.cumsum(np.random.randn(1000) * 0.01),
            'rate_b': 1.5 + np.cumsum(np.random.randn(1000) * 0.01)
        }, index=dates),
        'yields': pd.DataFrame({
            '2Y': 2.0 + np.random.randn(1000) * 0.1,
            '5Y': 2.5 + np.random.randn(1000) * 0.1,
            '10Y': 3.0 + np.random.randn(1000) * 0.1,
        }, index=dates),
        'commodities': pd.DataFrame({
            'gold': 1800 + np.cumsum(np.random.randn(1000)),
            'oil': 80 + np.cumsum(np.random.randn(1000) * 0.5),
            'copper': 4.0 + np.cumsum(np.random.randn(1000) * 0.02)
        }, index=dates),
        'fx': ohlc
    }
    
    # Transform data
    pipeline = DataPipeline()
    samples = pipeline.prepare_training_data(ohlc, events, macro_data)
    
    print(f"Generated {len(samples)} training samples")
    print(f"Sample OHLC shape: {samples[0].ohlc.shape}")
    print(f"Sample events keys: {samples[0].events.keys()}")
    print(f"Sample macro keys: {samples[0].macro_indicators.keys()}")
    
    return samples


if __name__ == '__main__':
    samples = example_usage()
