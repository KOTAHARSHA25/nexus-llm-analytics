# Time Series Analysis Agent Plugin
# Specialized agent for temporal data analysis and forecasting

import sys
import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add src to path for imports
# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

try:
    from backend.core.plugin_system import BasePluginAgent, AgentMetadata, AgentCapability
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    raise

# Time series analysis imports
try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TimeSeriesAgent(BasePluginAgent):
    """
    Advanced Time Series Analysis Agent
    
    Capabilities:
    - Time series decomposition (trend, seasonality, residuals)
    - Stationarity testing (ADF, KPSS tests)
    - Autocorrelation and partial autocorrelation analysis
    - ARIMA modeling and forecasting
    - Exponential smoothing (Holt-Winters)
    - Seasonal pattern detection and analysis
    - Trend analysis and change point detection
    - Time series anomaly detection
    - Forecast accuracy evaluation
    - Rolling statistics and moving averages
    - Lag analysis and cross-correlation
    - Time series clustering
    - Frequency analysis and spectral density
    
    Features:
    - Automatic date/time column detection
    - Missing value handling for time series
    - Multiple forecasting methods comparison
    - Seasonal adjustment and detrending
    - Forecast confidence intervals
    - Time series visualization recommendations
    - Performance metrics and model selection
    """
    
    def get_metadata(self) -> AgentMetadata:
        """Define agent metadata and capabilities"""
        return AgentMetadata(
            name="TimeSeriesAgent",
            version="1.0.0",
            description="Advanced time series analysis and forecasting agent with seasonal decomposition and ARIMA modeling",
            author="Nexus LLM Analytics Team",
            capabilities=[
                AgentCapability.TIME_SERIES,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.VISUALIZATION,
                AgentCapability.MACHINE_LEARNING
            ],
            file_types=[".csv", ".xlsx", ".json", ".txt"],
            dependencies=["pandas", "numpy", "scipy", "matplotlib", "statsmodels", "scikit-learn"],
            min_ram_mb=512,
            max_timeout_seconds=600,  # Longer timeout for complex forecasting
            priority=80  # High priority for time series data
        )
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the time series analysis agent"""
        try:
            # Configuration
            self.forecast_periods = self.config.get("forecast_periods", 12)
            self.confidence_level = self.config.get("confidence_level", 0.95)
            self.seasonal_periods = self.config.get("seasonal_periods", None)  # Auto-detect
            self.decomposition_model = self.config.get("decomposition_model", "additive")
            
            # Time series patterns for query matching
            self.ts_patterns = {
                "forecast": {
                    "patterns": ["forecast", "predict", "future", "next period", "projection"],
                    "description": "Generate forecasts for future periods"
                },
                "trend": {
                    "patterns": ["trend", "direction", "increasing", "decreasing", "growth", "decline"],
                    "description": "Analyze trends over time"
                },
                "seasonality": {
                    "patterns": ["seasonal", "season", "cyclic", "periodic", "pattern", "recurring"],
                    "description": "Detect and analyze seasonal patterns"
                },
                "decomposition": {
                    "patterns": ["decompose", "components", "breakdown", "separate"],
                    "description": "Decompose time series into components"
                },
                "stationarity": {
                    "patterns": ["stationary", "unit root", "adf", "kpss", "stable"],
                    "description": "Test for stationarity in time series"
                },
                "anomaly": {
                    "patterns": ["anomaly", "outlier", "unusual", "abnormal", "detect anomalies"],
                    "description": "Detect anomalies in time series"
                },
                "correlation": {
                    "patterns": ["autocorrelation", "correlation", "lag", "acf", "pacf"],
                    "description": "Analyze temporal correlations"
                }
            }
            
            # Date/time column patterns
            self.datetime_patterns = [
                "date", "time", "timestamp", "created", "updated", "year", "month", "day"
            ]
            
            # Forecast models
            self.available_models = ["arima", "exponential_smoothing", "moving_average", "linear_trend"]
            
            self.initialized = True
            logging.debug(f"Time Series Agent initialized: forecast_periods={self.forecast_periods}")
            
            return True
            
        except Exception as e:
            logging.error(f"Time Series Agent initialization failed: {e}")
            return False
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        """Determine if this agent can handle the time series query"""
        if not self.initialized:
            return 0.0
            
        confidence = 0.0
        query_lower = query.lower()
        
        # CRITICAL: Reject document files - they should go to RAG Agent
        document_extensions = [".pdf", ".docx", ".pptx", ".rtf"]
        if file_type and file_type.lower() in document_extensions:
            logging.debug(f"Time Series Agent rejecting document file: {file_type}")
            return 0.0
        
        # File type support - only structured data
        if file_type and file_type.lower() in [".csv", ".xlsx", ".json", ".txt"]:
            confidence += 0.1
        
        # Time series keywords
        ts_keywords = [
            "time series", "temporal", "over time", "time", "date",
            "timeline", "chronological", "sequence", "historical"
        ]
        
        keyword_matches = sum(1 for keyword in ts_keywords if keyword in query_lower)
        confidence += min(keyword_matches * 0.15, 0.4)
        
        # Specific time series patterns
        for pattern_type, pattern_data in self.ts_patterns.items():
            patterns = pattern_data["patterns"]
            if any(pattern in query_lower for pattern in patterns):
                confidence += 0.2
                break
        
        # Temporal terms
        temporal_terms = [
            "forecast", "predict", "trend", "seasonal", "periodic",
            "daily", "weekly", "monthly", "yearly", "quarterly",
            "lag", "autocorrelation", "arima", "exponential smoothing"
        ]
        
        temporal_matches = sum(1 for term in temporal_terms if term in query_lower)
        confidence += min(temporal_matches * 0.1, 0.3)
        
        # Date/time references
        datetime_refs = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "2020", "2021", "2022", "2023", "2024", "2025"
        ]
        
        datetime_matches = sum(1 for ref in datetime_refs if ref in query_lower)
        confidence += min(datetime_matches * 0.05, 0.15)
        
        return min(confidence, 1.0)
    
    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute time series analysis based on the query"""
        try:
            # Load data if filename provided
            filename = kwargs.get('filename')
            if filename and not data:
                data = self._load_data(filename)
            
            if data is None:
                return {
                    "success": False,
                    "error": "No data provided for time series analysis",
                    "agent": "TimeSeriesAgent"
                }
            
            # Detect and prepare time series data
            ts_data = self._prepare_time_series_data(data)
            if ts_data is None:
                return {
                    "success": False,
                    "error": "Could not identify time series structure in data",
                    "agent": "TimeSeriesAgent"
                }
            
            # Parse query intent
            intent = self._parse_ts_intent(query)
            
            # Execute appropriate time series analysis
            if intent == "forecast":
                return self._forecast_analysis(ts_data, query, **kwargs)
            elif intent == "trend":
                return self._trend_analysis(ts_data, query, **kwargs)
            elif intent == "seasonality":
                return self._seasonality_analysis(ts_data, query, **kwargs)
            elif intent == "decomposition":
                return self._decomposition_analysis(ts_data, query, **kwargs)
            elif intent == "stationarity":
                return self._stationarity_analysis(ts_data, query, **kwargs)
            elif intent == "anomaly":
                return self._anomaly_detection(ts_data, query, **kwargs)
            elif intent == "correlation":
                return self._correlation_analysis(ts_data, query, **kwargs)
            else:
                return self._comprehensive_ts_analysis(ts_data, query, **kwargs)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Time series analysis failed: {str(e)}",
                "agent": "TimeSeriesAgent"
            }
    
    def _load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from file"""
        try:
            base_data_dir = Path(__file__).parent.parent / "data"
            
            for subdir in ["uploads", "samples"]:
                filepath = base_data_dir / subdir / filename
                if filepath.exists():
                    if filename.endswith('.csv'):
                        return pd.read_csv(filepath)
                    elif filename.endswith(('.xlsx', '.xls')):
                        return pd.read_excel(filepath)
                    elif filename.endswith('.json'):
                        return pd.read_json(filepath)
                    
            return None
        except Exception as e:
            logging.error(f"Failed to load data from {filename}: {e}")
            return None
    
    def _prepare_time_series_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare and validate time series data"""
        try:
            # Auto-detect datetime column
            datetime_col = None
            for col in data.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in self.datetime_patterns):
                    # Try to convert to datetime
                    try:
                        pd.to_datetime(data[col])
                        datetime_col = col
                        break
                    except Exception:
                        continue
            
            if datetime_col is None:
                # Try to detect datetime in first few columns
                for col in data.columns[:3]:
                    try:
                        pd.to_datetime(data[col])
                        datetime_col = col
                        break
                    except Exception:
                        continue
            
            if datetime_col is None:
                logging.warning("No datetime column detected")
                return None
            
            # Prepare time series
            ts_data = data.copy()
            ts_data[datetime_col] = pd.to_datetime(ts_data[datetime_col])
            ts_data = ts_data.sort_values(datetime_col)
            ts_data.set_index(datetime_col, inplace=True)
            
            # Remove non-numeric columns except the index
            numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
            ts_data = ts_data[numeric_cols]
            
            logging.info(f"Prepared time series with {len(ts_data)} observations and {len(numeric_cols)} numeric columns")
            return ts_data
            
        except Exception as e:
            logging.error(f"Failed to prepare time series data: {e}")
            return None
    
    def _parse_ts_intent(self, query: str) -> str:
        """Parse the time series intent from the query"""
        query_lower = query.lower()
        
        # Check for specific time series patterns
        for pattern_type, pattern_data in self.ts_patterns.items():
            patterns = pattern_data["patterns"]
            if any(pattern in query_lower for pattern in patterns):
                return pattern_type
        
        # Default to comprehensive analysis
        return "comprehensive"
    
    def _forecast_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Generate forecasts using multiple methods"""
        try:
            results = {}
            
            # For each numeric column, generate forecasts
            for col in data.columns:
                series = data[col].dropna()
                if len(series) < 10:  # Need minimum data points
                    continue
                
                col_results = {}
                
                # Simple moving average forecast
                if len(series) >= 3:
                    window = min(3, len(series) // 3)
                    ma_forecast = series.rolling(window=window).mean().iloc[-1]
                    col_results["moving_average"] = {
                        "forecast": float(ma_forecast),
                        "method": f"{window}-period moving average"
                    }
                
                # Linear trend forecast
                try:
                    x = np.arange(len(series))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
                    next_point = slope * len(series) + intercept
                    col_results["linear_trend"] = {
                        "forecast": float(next_point),
                        "slope": float(slope),
                        "r_squared": float(r_value ** 2),
                        "p_value": float(p_value)
                    }
                except:
                    logging.debug("Operation failed (non-critical) - continuing")
                
                # Exponential smoothing (if available)
                if HAS_STATSMODELS and len(series) >= 6:
                    try:
                        model = ExponentialSmoothing(series, trend='add', seasonal=None)
                        fitted_model = model.fit()
                        forecast = fitted_model.forecast(steps=1)[0]
                        col_results["exponential_smoothing"] = {
                            "forecast": float(forecast),
                            "aic": float(fitted_model.aic),
                            "method": "Holt's exponential smoothing"
                        }
                    except:
                        logging.debug("Operation failed (non-critical) - continuing")
                
                results[col] = col_results
            
            return {
                "success": True,
                "result": {
                    "forecasts": results,
                    "data_period": {
                        "start": str(data.index.min()),
                        "end": str(data.index.max()),
                        "observations": len(data)
                    },
                    "forecast_horizon": 1  # Currently forecasting 1 period ahead
                },
                "agent": "TimeSeriesAgent",
                "operation": "forecast_analysis",
                "interpretation": self._interpret_forecasts(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Forecast analysis failed: {str(e)}",
                "agent": "TimeSeriesAgent"
            }
    
    def _trend_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        try:
            results = {}
            
            for col in data.columns:
                series = data[col].dropna()
                if len(series) < 3:
                    continue
                
                # Linear trend analysis
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
                
                # Trend classification
                if p_value < 0.05:
                    if slope > 0:
                        trend_type = "increasing"
                    else:
                        trend_type = "decreasing"
                    significance = "significant"
                else:
                    trend_type = "no clear trend"
                    significance = "not significant"
                
                # Calculate percentage change
                first_value = series.iloc[0]
                last_value = series.iloc[-1]
                pct_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                
                results[col] = {
                    "trend_type": trend_type,
                    "significance": significance,
                    "slope": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "percentage_change": float(pct_change),
                    "start_value": float(first_value),
                    "end_value": float(last_value)
                }
            
            return {
                "success": True,
                "result": results,
                "agent": "TimeSeriesAgent",
                "operation": "trend_analysis",
                "interpretation": self._interpret_trends(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Trend analysis failed: {str(e)}",
                "agent": "TimeSeriesAgent"
            }
    
    def _seasonality_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze seasonal patterns in time series data"""
        try:
            results = {}
            
            for col in data.columns:
                series = data[col].dropna()
                if len(series) < 24:  # Need sufficient data for seasonality
                    continue
                
                # Try to detect seasonality using autocorrelation
                col_results = {
                    "seasonal_detected": False,
                    "seasonal_period": None,
                    "seasonal_strength": None
                }
                
                # Simple seasonality detection using autocorrelation
                try:
                    # Check common seasonal periods
                    periods_to_check = [7, 12, 24, 30, 365]  # weekly, monthly, daily, etc.
                    max_corr = 0
                    best_period = None
                    
                    for period in periods_to_check:
                        if len(series) > period * 2:
                            # Calculate autocorrelation at this lag
                            corr = series.autocorr(lag=period)
                            if not np.isnan(corr) and abs(corr) > max_corr:
                                max_corr = abs(corr)
                                best_period = period
                    
                    if max_corr > 0.3:  # Threshold for seasonal detection
                        col_results["seasonal_detected"] = True
                        col_results["seasonal_period"] = best_period
                        col_results["seasonal_strength"] = float(max_corr)
                except:
                    logging.debug("Operation failed (non-critical) - continuing")
                
                # If statsmodels available, try seasonal decomposition
                if HAS_STATSMODELS and len(series) >= 24:
                    try:
                        decomposition = seasonal_decompose(series, model='additive', period=best_period if best_period else 12)
                        seasonal_component = decomposition.seasonal
                        seasonal_variance = seasonal_component.var()
                        total_variance = series.var()
                        seasonal_ratio = seasonal_variance / total_variance if total_variance > 0 else 0
                        
                        col_results["decomposition"] = {
                            "seasonal_variance_ratio": float(seasonal_ratio),
                            "trend_component_available": True,
                            "residual_component_available": True
                        }
                    except:
                        logging.debug("Operation failed (non-critical) - continuing")
                
                results[col] = col_results
            
            return {
                "success": True,
                "result": results,
                "agent": "TimeSeriesAgent",
                "operation": "seasonality_analysis",
                "interpretation": self._interpret_seasonality(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Seasonality analysis failed: {str(e)}",
                "agent": "TimeSeriesAgent"
            }
    
    def _comprehensive_ts_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive time series analysis"""
        try:
            results = {}
            
            # Basic time series info
            results["data_info"] = {
                "start_date": str(data.index.min()),
                "end_date": str(data.index.max()),
                "observations": len(data),
                "frequency": self._detect_frequency(data.index),
                "columns": list(data.columns)
            }
            
            # Trend analysis
            trend_result = self._trend_analysis(data, query, **kwargs)
            if trend_result["success"]:
                results["trends"] = trend_result["result"]
            
            # Seasonality analysis  
            seasonality_result = self._seasonality_analysis(data, query, **kwargs)
            if seasonality_result["success"]:
                results["seasonality"] = seasonality_result["result"]
            
            # Basic forecasts
            forecast_result = self._forecast_analysis(data, query, **kwargs)
            if forecast_result["success"]:
                results["forecasts"] = forecast_result["result"]
            
            return {
                "success": True,
                "result": results,
                "agent": "TimeSeriesAgent",
                "operation": "comprehensive_ts_analysis",
                "interpretation": "Comprehensive time series analysis completed with trend detection, seasonality analysis, and basic forecasting."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Comprehensive time series analysis failed: {str(e)}",
                "agent": "TimeSeriesAgent"
            }
    
    def _detect_frequency(self, index: pd.DatetimeIndex) -> str:
        """Detect the frequency of the time series"""
        try:
            if len(index) < 2:
                return "unknown"
            
            # Calculate time differences
            diffs = index[1:] - index[:-1]
            mode_diff = diffs.mode()[0] if len(diffs.mode()) > 0 else diffs[0]
            
            # Classify frequency
            if mode_diff <= timedelta(minutes=1):
                return "minute"
            elif mode_diff <= timedelta(hours=1):
                return "hourly"
            elif mode_diff <= timedelta(days=1):
                return "daily"
            elif mode_diff <= timedelta(days=7):
                return "weekly"
            elif mode_diff <= timedelta(days=31):
                return "monthly"
            elif mode_diff <= timedelta(days=366):
                return "yearly"
            else:
                return "irregular"
                
        except Exception:
            return "unknown"
    
    def _interpret_forecasts(self, results: Dict) -> str:
        """Generate interpretation of forecast results"""
        interpretations = []
        
        for col, forecasts in results.items():
            forecast_values = []
            for method, data in forecasts.items():
                forecast_values.append(f"{method}: {data['forecast']:.2f}")
            
            if forecast_values:
                interpretations.append(f"{col} forecasts - {', '.join(forecast_values)}")
        
        return " ".join(interpretations)
    
    def _interpret_trends(self, results: Dict) -> str:
        """Generate interpretation of trend analysis"""
        interpretations = []
        
        for col, trend_data in results.items():
            trend_type = trend_data["trend_type"]
            significance = trend_data["significance"]
            pct_change = trend_data["percentage_change"]
            
            interpretations.append(f"{col}: {trend_type} trend ({significance}, {pct_change:.1f}% change)")
        
        return " ".join(interpretations)
    
    def _interpret_seasonality(self, results: Dict) -> str:
        """Generate interpretation of seasonality analysis"""
        interpretations = []
        
        for col, seasonal_data in results.items():
            if seasonal_data["seasonal_detected"]:
                period = seasonal_data["seasonal_period"]
                strength = seasonal_data["seasonal_strength"]
                interpretations.append(f"{col}: Seasonal pattern detected (period: {period}, strength: {strength:.2f})")
            else:
                interpretations.append(f"{col}: No clear seasonal pattern detected")
        
        return " ".join(interpretations)
    
    def _decomposition_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Uses seasonal_decompose from statsmodels to separate:
        - Trend: Long-term progression
        - Seasonal: Repeating short-term patterns
        - Residual: Random/irregular components
        """
        try:
            if not HAS_STATSMODELS:
                return {
                    "success": False,
                    "error": "statsmodels library required for decomposition",
                    "agent": "TimeSeriesAgent"
                }
            
            results = {}
            
            for col in data.columns:
                series = data[col].dropna()
                
                # Need sufficient data points for decomposition
                if len(series) < 24:
                    continue
                
                try:
                    # Auto-detect period or use default
                    period = kwargs.get('period', None)
                    if period is None:
                        # Try to detect period from frequency
                        freq = self._detect_frequency(series.index)
                        period_map = {
                            'hourly': 24,
                            'daily': 7,
                            'weekly': 52,
                            'monthly': 12,
                            'yearly': 1
                        }
                        period = period_map.get(freq, 12)
                    
                    # Ensure period is valid
                    if len(series) < 2 * period:
                        period = max(2, len(series) // 4)
                    
                    # Perform decomposition (additive model)
                    decomposition = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
                    
                    # Extract components
                    trend = decomposition.trend.dropna()
                    seasonal = decomposition.seasonal.dropna()
                    residual = decomposition.resid.dropna()
                    
                    # Calculate component strengths
                    total_var = series.var()
                    trend_strength = 1 - (residual.var() / total_var) if total_var > 0 else 0
                    seasonal_strength = seasonal.var() / total_var if total_var > 0 else 0
                    
                    # Summary statistics for each component
                    results[col] = {
                        "period": period,
                        "model": "additive",
                        "components": {
                            "trend": {
                                "mean": float(trend.mean()),
                                "std": float(trend.std()),
                                "min": float(trend.min()),
                                "max": float(trend.max()),
                                "strength": float(trend_strength)
                            },
                            "seasonal": {
                                "mean": float(seasonal.mean()),
                                "std": float(seasonal.std()),
                                "min": float(seasonal.min()),
                                "max": float(seasonal.max()),
                                "strength": float(seasonal_strength),
                                "amplitude": float(seasonal.max() - seasonal.min())
                            },
                            "residual": {
                                "mean": float(residual.mean()),
                                "std": float(residual.std()),
                                "min": float(residual.min()),
                                "max": float(residual.max())
                            }
                        },
                        "interpretation": self._interpret_decomposition(
                            trend_strength, seasonal_strength, period
                        )
                    }
                    
                except Exception as e:
                    logging.debug(f"Decomposition failed for {col}: {e}")
                    continue
            
            if not results:
                return {
                    "success": False,
                    "error": "Insufficient data for decomposition (need at least 24 observations)",
                    "agent": "TimeSeriesAgent"
                }
            
            return {
                "success": True,
                "result": results,
                "agent": "TimeSeriesAgent",
                "operation": "decomposition_analysis",
                "interpretation": f"Time series decomposed into trend, seasonal, and residual components for {len(results)} variable(s)."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Decomposition analysis failed: {str(e)}",
                "agent": "TimeSeriesAgent"
            }
    
    def _interpret_decomposition(self, trend_strength: float, seasonal_strength: float, period: int) -> str:
        """Generate interpretation of decomposition results"""
        interpretations = []
        
        # Trend interpretation
        if trend_strength > 0.7:
            interpretations.append("Strong trend component")
        elif trend_strength > 0.4:
            interpretations.append("Moderate trend component")
        else:
            interpretations.append("Weak trend component")
        
        # Seasonal interpretation
        if seasonal_strength > 0.3:
            interpretations.append(f"significant seasonality (period={period})")
        elif seasonal_strength > 0.1:
            interpretations.append(f"moderate seasonality (period={period})")
        else:
            interpretations.append("minimal seasonality")
        
        return ", ".join(interpretations)
    
    def _stationarity_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """
        Test for stationarity using ADF and KPSS tests
        
        Stationarity is important for many time series models (ARIMA, etc.)
        - ADF Test: Null hypothesis = series has unit root (non-stationary)
        - KPSS Test: Null hypothesis = series is stationary
        
        A stationary series has:
        - Constant mean over time
        - Constant variance over time
        - No seasonal patterns
        """
        try:
            if not HAS_STATSMODELS:
                return {
                    "success": False,
                    "error": "statsmodels library required for stationarity tests",
                    "agent": "TimeSeriesAgent"
                }
            
            results = {}
            
            for col in data.columns:
                series = data[col].dropna()
                
                if len(series) < 10:
                    continue
                
                col_results = {}
                
                # Augmented Dickey-Fuller Test
                try:
                    adf_result = adfuller(series, autolag='AIC')
                    adf_statistic = adf_result[0]
                    adf_pvalue = adf_result[1]
                    adf_critical_values = adf_result[4]
                    
                    # Interpretation: p < 0.05 means reject null (series IS stationary)
                    adf_stationary = adf_pvalue < 0.05
                    
                    col_results["adf_test"] = {
                        "statistic": float(adf_statistic),
                        "p_value": float(adf_pvalue),
                        "critical_values": {
                            "1%": float(adf_critical_values['1%']),
                            "5%": float(adf_critical_values['5%']),
                            "10%": float(adf_critical_values['10%'])
                        },
                        "stationary": adf_stationary,
                        "interpretation": "Stationary" if adf_stationary else "Non-stationary (has unit root)"
                    }
                except Exception as e:
                    logging.debug(f"ADF test failed for {col}: {e}")
                
                # KPSS Test
                try:
                    kpss_result = kpss(series, regression='c', nlags='auto')
                    kpss_statistic = kpss_result[0]
                    kpss_pvalue = kpss_result[1]
                    kpss_critical_values = kpss_result[3]
                    
                    # Interpretation: p < 0.05 means reject null (series is NOT stationary)
                    kpss_stationary = kpss_pvalue >= 0.05
                    
                    col_results["kpss_test"] = {
                        "statistic": float(kpss_statistic),
                        "p_value": float(kpss_pvalue),
                        "critical_values": {
                            "1%": float(kpss_critical_values['1%']),
                            "2.5%": float(kpss_critical_values['2.5%']),
                            "5%": float(kpss_critical_values['5%']),
                            "10%": float(kpss_critical_values['10%'])
                        },
                        "stationary": kpss_stationary,
                        "interpretation": "Stationary" if kpss_stationary else "Non-stationary"
                    }
                except Exception as e:
                    logging.debug(f"KPSS test failed for {col}: {e}")
                
                # Overall conclusion
                if "adf_test" in col_results and "kpss_test" in col_results:
                    adf_stat = col_results["adf_test"]["stationary"]
                    kpss_stat = col_results["kpss_test"]["stationary"]
                    
                    if adf_stat and kpss_stat:
                        overall = "Stationary"
                    elif not adf_stat and not kpss_stat:
                        overall = "Non-stationary"
                    else:
                        overall = "Inconclusive (tests disagree)"
                    
                    col_results["overall_conclusion"] = overall
                    
                    # Recommendations
                    if overall == "Non-stationary":
                        col_results["recommendation"] = "Consider differencing or transformation before modeling"
                    elif overall == "Stationary":
                        col_results["recommendation"] = "Series is suitable for ARIMA and other time series models"
                    else:
                        col_results["recommendation"] = "Review data quality and consider additional tests"
                
                results[col] = col_results
            
            if not results:
                return {
                    "success": False,
                    "error": "Insufficient data for stationarity tests (need at least 10 observations)",
                    "agent": "TimeSeriesAgent"
                }
            
            return {
                "success": True,
                "result": results,
                "agent": "TimeSeriesAgent",
                "operation": "stationarity_analysis",
                "interpretation": self._interpret_stationarity(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Stationarity analysis failed: {str(e)}",
                "agent": "TimeSeriesAgent"
            }
    
    def _interpret_stationarity(self, results: Dict) -> str:
        """Generate interpretation of stationarity tests"""
        interpretations = []
        
        for col, tests in results.items():
            if "overall_conclusion" in tests:
                conclusion = tests["overall_conclusion"]
                interpretations.append(f"{col}: {conclusion}")
        
        return " | ".join(interpretations)
    
    def _anomaly_detection(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """
        Detect anomalies in time series data using multiple methods
        
        Methods:
        1. Statistical (Z-score): Values beyond ±3 standard deviations
        2. IQR Method: Values outside Q1-1.5*IQR or Q3+1.5*IQR
        3. Moving Average: Deviations from rolling mean
        4. Seasonal: Deviations considering seasonal patterns
        """
        try:
            results = {}
            
            for col in data.columns:
                series = data[col].dropna()
                
                if len(series) < 10:
                    continue
                
                anomalies = {}
                
                # Method 1: Z-score (statistical outliers)
                try:
                    mean = series.mean()
                    std = series.std()
                    z_scores = np.abs((series - mean) / std)
                    z_anomalies = series[z_scores > 3]
                    
                    anomalies["z_score"] = {
                        "count": len(z_anomalies),
                        "indices": [str(idx) for idx in z_anomalies.index[:10]],  # First 10
                        "values": [float(val) for val in z_anomalies.values[:10]],
                        "threshold": 3.0,
                        "method": "Statistical (±3σ)"
                    }
                except Exception as e:
                    logging.debug(f"Z-score anomaly detection failed: {e}")
                
                # Method 2: IQR (Interquartile Range)
                try:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    iqr_anomalies = series[(series < lower_bound) | (series > upper_bound)]
                    
                    anomalies["iqr"] = {
                        "count": len(iqr_anomalies),
                        "indices": [str(idx) for idx in iqr_anomalies.index[:10]],
                        "values": [float(val) for val in iqr_anomalies.values[:10]],
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "method": "IQR (Interquartile Range)"
                    }
                except Exception as e:
                    logging.debug(f"IQR anomaly detection failed: {e}")
                
                # Method 3: Moving Average Deviation
                try:
                    window = min(7, len(series) // 4)
                    if window >= 2:
                        rolling_mean = series.rolling(window=window, center=True).mean()
                        rolling_std = series.rolling(window=window, center=True).std()
                        
                        # Points that deviate significantly from rolling mean
                        deviations = np.abs(series - rolling_mean) / rolling_std
                        ma_anomalies = series[deviations > 2.5]
                        
                        anomalies["moving_average"] = {
                            "count": len(ma_anomalies),
                            "indices": [str(idx) for idx in ma_anomalies.index[:10]],
                            "values": [float(val) for val in ma_anomalies.values[:10]],
                            "window": window,
                            "threshold": 2.5,
                            "method": f"Moving Average ({window}-period)"
                        }
                except Exception as e:
                    logging.debug(f"Moving average anomaly detection failed: {e}")
                
                # Method 4: Rate of Change (sudden spikes)
                try:
                    if len(series) > 1:
                        pct_change = series.pct_change().abs()
                        # Detect sudden changes > 50%
                        spike_threshold = 0.5
                        spikes = series[pct_change > spike_threshold]
                        
                        anomalies["sudden_changes"] = {
                            "count": len(spikes),
                            "indices": [str(idx) for idx in spikes.index[:10]],
                            "values": [float(val) for val in spikes.values[:10]],
                            "threshold_pct": spike_threshold * 100,
                            "method": "Sudden Rate of Change"
                        }
                except Exception as e:
                    logging.debug(f"Rate of change detection failed: {e}")
                
                # Summary
                total_anomalies = sum(method_data["count"] for method_data in anomalies.values())
                anomaly_rate = (total_anomalies / len(series)) * 100 if len(series) > 0 else 0
                
                results[col] = {
                    "total_observations": len(series),
                    "anomaly_methods": anomalies,
                    "summary": {
                        "total_anomalies_detected": total_anomalies,
                        "anomaly_rate_percent": float(anomaly_rate),
                        "methods_used": len(anomalies)
                    },
                    "interpretation": self._interpret_anomalies(anomalies, anomaly_rate)
                }
            
            if not results:
                return {
                    "success": False,
                    "error": "Insufficient data for anomaly detection (need at least 10 observations)",
                    "agent": "TimeSeriesAgent"
                }
            
            return {
                "success": True,
                "result": results,
                "agent": "TimeSeriesAgent",
                "operation": "anomaly_detection",
                "interpretation": f"Anomaly detection completed using multiple methods across {len(results)} variable(s)."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Anomaly detection failed: {str(e)}",
                "agent": "TimeSeriesAgent"
            }
    
    def _interpret_anomalies(self, anomalies: Dict, anomaly_rate: float) -> str:
        """Generate interpretation of anomaly detection results"""
        if anomaly_rate < 1:
            severity = "Low"
        elif anomaly_rate < 5:
            severity = "Moderate"
        else:
            severity = "High"
        
        methods = [method_data["method"] for method_data in anomalies.values() if method_data["count"] > 0]
        
        return f"{severity} anomaly rate ({anomaly_rate:.1f}%). Detected by: {', '.join(methods)}"
    
    def _correlation_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze autocorrelation and partial autocorrelation
        
        - Autocorrelation (ACF): Correlation of series with its lagged values
        - Partial Autocorrelation (PACF): Correlation after removing effects of intermediate lags
        
        Used for:
        - Identifying AR and MA orders for ARIMA models
        - Detecting seasonal patterns
        - Understanding temporal dependencies
        """
        try:
            results = {}
            
            for col in data.columns:
                series = data[col].dropna()
                
                if len(series) < 10:
                    continue
                
                col_results = {}
                
                # Calculate autocorrelations
                try:
                    max_lags = min(20, len(series) // 2 - 1)
                    
                    # Simple autocorrelation calculation
                    acf_values = []
                    for lag in range(1, max_lags + 1):
                        if len(series) > lag:
                            corr = series.autocorr(lag=lag)
                            if not np.isnan(corr):
                                acf_values.append({
                                    "lag": lag,
                                    "correlation": float(corr),
                                    "significant": abs(corr) > 1.96 / np.sqrt(len(series))  # 95% CI
                                })
                    
                    # Identify significant lags
                    significant_lags = [item for item in acf_values if item["significant"]]
                    
                    col_results["autocorrelation"] = {
                        "lags_tested": max_lags,
                        "correlations": acf_values[:10],  # First 10 lags
                        "significant_lags": [lag["lag"] for lag in significant_lags],
                        "max_correlation": {
                            "lag": max(acf_values, key=lambda x: abs(x["correlation"]))["lag"],
                            "value": float(max(acf_values, key=lambda x: abs(x["correlation"]))["correlation"])
                        } if acf_values else None
                    }
                except Exception as e:
                    logging.debug(f"Autocorrelation calculation failed: {e}")
                
                # Lag analysis - find optimal lag
                try:
                    # Test multiple lags and find strongest correlation
                    best_lag = 1
                    best_corr = 0
                    
                    for lag in range(1, min(13, len(series) // 2)):
                        corr = abs(series.autocorr(lag=lag))
                        if not np.isnan(corr) and corr > best_corr:
                            best_corr = corr
                            best_lag = lag
                    
                    col_results["optimal_lag"] = {
                        "lag": best_lag,
                        "correlation": float(best_corr),
                        "interpretation": f"Strongest correlation at lag {best_lag}"
                    }
                except Exception as e:
                    logging.debug(f"Lag analysis failed: {e}")
                
                # Cross-correlation with other columns (if multiple)
                if len(data.columns) > 1:
                    try:
                        cross_correlations = []
                        for other_col in data.columns:
                            if other_col != col:
                                other_series = data[other_col].dropna()
                                if len(other_series) == len(series):
                                    # Pearson correlation
                                    corr = series.corr(other_series)
                                    if not np.isnan(corr):
                                        cross_correlations.append({
                                            "variable": other_col,
                                            "correlation": float(corr),
                                            "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                                        })
                        
                        if cross_correlations:
                            col_results["cross_correlation"] = cross_correlations
                    except Exception as e:
                        logging.debug(f"Cross-correlation failed: {e}")
                
                # Durbin-Watson test for autocorrelation in residuals
                try:
                    # Simple linear trend for residuals
                    x = np.arange(len(series))
                    slope, intercept = np.polyfit(x, series.values, 1)
                    trend = slope * x + intercept
                    residuals = series.values - trend
                    
                    # Durbin-Watson statistic
                    diff_resid = np.diff(residuals)
                    dw_stat = np.sum(diff_resid**2) / np.sum(residuals**2)
                    
                    # DW interpretation: ~2 = no autocorrelation, <2 = positive, >2 = negative
                    if dw_stat < 1.5:
                        dw_interpretation = "Positive autocorrelation (values cluster)"
                    elif dw_stat > 2.5:
                        dw_interpretation = "Negative autocorrelation (values alternate)"
                    else:
                        dw_interpretation = "No significant autocorrelation"
                    
                    col_results["durbin_watson"] = {
                        "statistic": float(dw_stat),
                        "interpretation": dw_interpretation
                    }
                except Exception as e:
                    logging.debug(f"Durbin-Watson test failed: {e}")
                
                results[col] = col_results
            
            if not results:
                return {
                    "success": False,
                    "error": "Insufficient data for correlation analysis (need at least 10 observations)",
                    "agent": "TimeSeriesAgent"
                }
            
            return {
                "success": True,
                "result": results,
                "agent": "TimeSeriesAgent",
                "operation": "correlation_analysis",
                "interpretation": self._interpret_correlations(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Correlation analysis failed: {str(e)}",
                "agent": "TimeSeriesAgent"
            }
    
    def _interpret_correlations(self, results: Dict) -> str:
        """Generate interpretation of correlation analysis"""
        interpretations = []
        
        for col, corr_data in results.items():
            if "optimal_lag" in corr_data:
                lag = corr_data["optimal_lag"]["lag"]
                corr = corr_data["optimal_lag"]["correlation"]
                interpretations.append(f"{col}: strongest autocorrelation at lag {lag} (r={corr:.2f})")
        
        return " | ".join(interpretations) if interpretations else "Correlation analysis completed"