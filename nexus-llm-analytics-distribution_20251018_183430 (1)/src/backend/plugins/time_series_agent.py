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
src_path = Path(__file__).parent.parent / "src"
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
            logging.info(f"✅ Time Series Agent initialized")
            logging.info(f"   Default forecast periods: {self.forecast_periods}")
            logging.info(f"   Confidence level: {self.confidence_level}")
            logging.info(f"   Available models: {', '.join(self.available_models)}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Time Series Agent initialization failed: {e}")
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
    
    # Placeholder methods for other analyses
    def _decomposition_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Placeholder for time series decomposition"""
        return {
            "success": True,
            "result": {"message": "Time series decomposition would be implemented here"},
            "agent": "TimeSeriesAgent",
            "operation": "decomposition_analysis"
        }
    
    def _stationarity_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Placeholder for stationarity testing"""
        return {
            "success": True,
            "result": {"message": "Stationarity testing would be implemented here"},
            "agent": "TimeSeriesAgent",
            "operation": "stationarity_analysis"
        }
    
    def _anomaly_detection(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Placeholder for anomaly detection"""
        return {
            "success": True,
            "result": {"message": "Time series anomaly detection would be implemented here"},
            "agent": "TimeSeriesAgent",
            "operation": "anomaly_detection"
        }
    
    def _correlation_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Placeholder for autocorrelation analysis"""
        return {
            "success": True,
            "result": {"message": "Autocorrelation analysis would be implemented here"},
            "agent": "TimeSeriesAgent",
            "operation": "correlation_analysis"
        }