# Advanced Statistical Analysis Agent Plugin
# Specialized agent for comprehensive statistical analysis and hypothesis testing

import sys
import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

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

# Statistical analysis imports
try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    import scipy.stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None
    scipy_stats = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class StatisticalAgent(BasePluginAgent):
    """
    Advanced Statistical Analysis Agent
    
    Capabilities:
    - Descriptive statistics and data profiling
    - Hypothesis testing (t-tests, chi-square, ANOVA, etc.)  
    - Correlation and regression analysis
    - Distribution analysis and fitting
    - Statistical significance testing
    - Confidence intervals and effect sizes
    - Outlier detection and handling
    - Sample size calculations
    - Power analysis
    - Non-parametric tests
    - Time series statistical analysis
    - Multivariate analysis
    - Principal Component Analysis (PCA)
    - Cluster analysis for statistical grouping
    
    Features:
    - Automatic assumption checking
    - Effect size calculations
    - Multiple comparison corrections
    - Bootstrap confidence intervals
    - Robust statistical methods
    - Statistical report generation
    - Visualization recommendations
    """
    
    def get_metadata(self) -> AgentMetadata:
        """Define agent metadata and capabilities"""
        return AgentMetadata(
            name="StatisticalAgent",
            version="1.0.0",
            description="Advanced statistical analysis agent with hypothesis testing, regression, and multivariate analysis capabilities",
            author="Nexus LLM Analytics Team",
            capabilities=[
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.VISUALIZATION,
                AgentCapability.REPORTING,
                AgentCapability.MACHINE_LEARNING
            ],
            file_types=[".csv", ".xlsx", ".json", ".txt"],
            dependencies=["pandas", "numpy", "scipy", "matplotlib", "seaborn", "scikit-learn"],
            min_ram_mb=512,
            max_timeout_seconds=300,
            priority=75  # High priority for statistical analysis
        )
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the statistical analysis agent"""
        try:
            # Check required dependencies
            if not HAS_SCIPY:
                logging.error("SciPy not available - required for statistical tests")
                return False
                
            # Configuration
            self.confidence_level = self.config.get("confidence_level", 0.95)
            self.alpha = 1 - self.confidence_level
            self.effect_size_threshold = self.config.get("effect_size_threshold", 0.5)
            self.outlier_method = self.config.get("outlier_method", "iqr")  # iqr, zscore, modified_zscore
            
            # Statistical test patterns
            self.test_patterns = {
                "ttest": {
                    "patterns": ["t-test", "t test", "mean difference", "compare means", "two groups"],
                    "description": "Compare means between two groups"
                },
                "anova": {
                    "patterns": ["anova", "analysis of variance", "multiple groups", "three groups", "more than two"],
                    "description": "Compare means across multiple groups"
                },
                "correlation": {
                    "patterns": ["correlation", "relationship", "association", "correl", "pearson", "spearman"],
                    "description": "Measure association between variables"
                },
                "regression": {
                    "patterns": ["regression", "predict", "linear model", "relationship", "dependent variable"],
                    "description": "Model relationships between variables"
                },
                "chi_square": {
                    "patterns": ["chi-square", "chi square", "independence", "categorical", "frequency"],
                    "description": "Test independence in categorical data"
                },
                "normality": {
                    "patterns": ["normal", "normality", "distribution", "shapiro", "kolmogorov"],
                    "description": "Test for normal distribution"
                },
                "outliers": {
                    "patterns": ["outlier", "anomaly", "extreme value", "unusual", "detect outliers"],
                    "description": "Identify and analyze outliers"
                }
            }
            
            # Analysis history
            self.analysis_history = []
            
            self.initialized = True
            logging.debug(f"Statistical Agent initialized: confidence={self.confidence_level}, alpha={self.alpha}")
            
            return True
            
        except Exception as e:
            logging.error(f"Statistical Agent initialization failed: {e}")
            return False
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        """Determine if this agent can handle the statistical query"""
        if not self.initialized:
            return 0.0
            
        confidence = 0.0
        query_lower = query.lower()
        
        # CRITICAL: Reject document files - they should go to RAG Agent
        document_extensions = [".pdf", ".docx", ".pptx", ".rtf"]
        if file_type and file_type.lower() in document_extensions:
            logging.debug(f"Statistical Agent rejecting document file: {file_type}")
            return 0.0
        
        # File type support - only structured data
        if file_type and file_type.lower() in [".csv", ".xlsx", ".json", ".txt"]:
            confidence += 0.2
        
        # Statistical keywords
        stat_keywords = [
            "statistics", "statistical", "stat", "analysis", "test",
            "hypothesis", "significance", "p-value", "confidence",
            "mean", "median", "variance", "standard deviation"
        ]
        
        keyword_matches = sum(1 for keyword in stat_keywords if keyword in query_lower)
        confidence += min(keyword_matches * 0.1, 0.3)
        
        # Specific statistical tests and methods
        for test_type, test_data in self.test_patterns.items():
            patterns = test_data["patterns"]
            if any(pattern in query_lower for pattern in patterns):
                confidence += 0.2
                break
        
        # Advanced statistical terms
        advanced_terms = [
            "effect size", "power analysis", "bootstrap", "parametric",
            "non-parametric", "distribution", "probability", "sampling",
            "inference", "multivariate", "pca", "cluster", "regression"
        ]
        
        advanced_matches = sum(1 for term in advanced_terms if term in query_lower)
        confidence += min(advanced_matches * 0.08, 0.25)
        
        # Statistical operations
        operations = [
            "compare", "test", "analyze", "examine", "investigate",
            "correlate", "predict", "model", "estimate", "calculate"
        ]
        
        operation_matches = sum(1 for op in operations if op in query_lower)
        confidence += min(operation_matches * 0.05, 0.15)
        
        return min(confidence, 1.0)
    
    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute statistical analysis based on the query"""
        try:
            # Load data if filename provided
            filename = kwargs.get('filename')
            if filename and not data:
                data = self._load_data(filename)
            
            if data is None:
                return {
                    "success": False,
                    "error": "No data provided for statistical analysis",
                    "agent": "StatisticalAgent"
                }
            
            # Parse query intent
            intent = self._parse_statistical_intent(query)
            
            # Execute appropriate statistical analysis
            if intent == "descriptive":
                return self._descriptive_statistics(data, query, **kwargs)
            elif intent == "ttest":
                return self._t_test_analysis(data, query, **kwargs)
            elif intent == "anova":
                return self._anova_analysis(data, query, **kwargs)
            elif intent == "correlation":
                return self._correlation_analysis(data, query, **kwargs)
            elif intent == "regression":
                return self._regression_analysis(data, query, **kwargs)
            elif intent == "chi_square":
                return self._chi_square_analysis(data, query, **kwargs)
            elif intent == "normality":
                return self._normality_test(data, query, **kwargs)
            elif intent == "outliers":
                return self._outlier_analysis(data, query, **kwargs)
            else:
                return self._comprehensive_analysis(data, query, **kwargs)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Statistical analysis failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from file"""
        try:
            # Look for files in uploads and samples directories
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
    
    def _parse_statistical_intent(self, query: str) -> str:
        """Parse the statistical intent from the query"""
        query_lower = query.lower()
        
        # Check for specific statistical tests
        for test_type, test_data in self.test_patterns.items():
            patterns = test_data["patterns"]
            if any(pattern in query_lower for pattern in patterns):
                return test_type
        
        # Default to comprehensive analysis
        if any(term in query_lower for term in ["summary", "describe", "overview", "profile"]):
            return "descriptive"
        
        return "comprehensive"
    
    def _descriptive_statistics(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Generate comprehensive descriptive statistics"""
        try:
            results = {
                "basic_stats": data.describe().to_dict(),
                "data_info": {
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": data.dtypes.to_dict(),
                    "missing_values": data.isnull().sum().to_dict()
                },
                "numeric_summary": {},
                "categorical_summary": {}
            }
            
            # Analyze numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].notna().sum() > 0:  # Skip empty columns
                    col_data = data[col].dropna()
                    results["numeric_summary"][col] = {
                        "count": len(col_data),
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "var": float(col_data.var()),
                        "skewness": float(scipy_stats.skew(col_data)),
                        "kurtosis": float(scipy_stats.kurtosis(col_data)),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "range": float(col_data.max() - col_data.min()),
                        "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                        "cv": float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0
                    }
            
            # Analyze categorical columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if data[col].notna().sum() > 0:
                    value_counts = data[col].value_counts()
                    results["categorical_summary"][col] = {
                        "unique_count": data[col].nunique(),
                        "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                        "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "value_counts": value_counts.head(10).to_dict()
                    }
            
            return {
                "success": True,
                "result": results,
                "agent": "StatisticalAgent",
                "operation": "descriptive_statistics",
                "interpretation": self._interpret_descriptive_stats(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Descriptive statistics failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _correlation_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform correlation analysis"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 numeric columns for correlation analysis",
                    "agent": "StatisticalAgent"
                }
            
            # Calculate different correlation coefficients
            pearson_corr = numeric_data.corr(method='pearson')
            spearman_corr = numeric_data.corr(method='spearman')
            
            # Statistical significance testing
            correlation_tests = {}
            for i, col1 in enumerate(numeric_data.columns):
                for j, col2 in enumerate(numeric_data.columns):
                    if i < j:  # Avoid duplicate pairs
                        x = numeric_data[col1].dropna()
                        y = numeric_data[col2].dropna()
                        
                        # Align the data (remove pairs where either is NaN)
                        combined = pd.DataFrame({'x': x, 'y': y}).dropna()
                        if len(combined) > 2:
                            x_clean = combined['x']
                            y_clean = combined['y']
                            
                            # Pearson correlation test
                            pearson_r, pearson_p = scipy_stats.pearsonr(x_clean, y_clean)
                            
                            # Spearman correlation test
                            spearman_r, spearman_p = scipy_stats.spearmanr(x_clean, y_clean)
                            
                            correlation_tests[f"{col1}_vs_{col2}"] = {
                                "pearson": {"r": float(pearson_r), "p_value": float(pearson_p)},
                                "spearman": {"r": float(spearman_r), "p_value": float(spearman_p)},
                                "sample_size": len(x_clean)
                            }
            
            return {
                "success": True,
                "result": {
                    "pearson_correlation": pearson_corr.to_dict(),
                    "spearman_correlation": spearman_corr.to_dict(),
                    "significance_tests": correlation_tests,
                    "strong_correlations": self._identify_strong_correlations(pearson_corr),
                },
                "agent": "StatisticalAgent", 
                "operation": "correlation_analysis",
                "interpretation": self._interpret_correlations(correlation_tests)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Correlation analysis failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _identify_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Identify strong correlations above threshold"""
        strong_corr = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlation
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= threshold:
                        strong_corr.append({
                            "variable1": col1,
                            "variable2": col2,
                            "correlation": float(corr_value),
                            "strength": "very strong" if abs(corr_value) >= 0.9 else "strong"
                        })
        
        return strong_corr
    
    def _interpret_descriptive_stats(self, results: Dict) -> str:
        """Generate interpretation of descriptive statistics"""
        interpretations = []
        
        # Dataset overview
        shape = results["data_info"]["shape"]
        interpretations.append(f"Dataset contains {shape[0]} rows and {shape[1]} columns.")
        
        # Missing data analysis
        missing = results["data_info"]["missing_values"]
        total_missing = sum(missing.values())
        if total_missing > 0:
            interpretations.append(f"Found {total_missing} missing values across the dataset.")
        
        # Numeric variables analysis
        numeric_summary = results["numeric_summary"]
        for col, stats in numeric_summary.items():
            skew = stats["skewness"]
            if abs(skew) > 1:
                skew_desc = "highly skewed" if abs(skew) > 2 else "moderately skewed"
                direction = "right" if skew > 0 else "left"
                interpretations.append(f"{col} is {skew_desc} to the {direction} (skewness: {skew:.2f}).")
            
            cv = stats["cv"]
            if cv > 1:
                interpretations.append(f"{col} shows high variability (CV: {cv:.2f}).")
        
        return " ".join(interpretations)
    
    def _interpret_correlations(self, correlation_tests: Dict) -> str:
        """Generate interpretation of correlation results"""
        interpretations = []
        
        significant_correlations = []
        for pair, tests in correlation_tests.items():
            pearson = tests["pearson"]
            if pearson["p_value"] < 0.05:
                strength = self._correlation_strength(abs(pearson["r"]))
                direction = "positive" if pearson["r"] > 0 else "negative"
                significant_correlations.append(f"{pair.replace('_vs_', ' and ')}: {strength} {direction} correlation (r={pearson['r']:.3f}, p={pearson['p_value']:.3f})")
        
        if significant_correlations:
            interpretations.append("Significant correlations found:")
            interpretations.extend(significant_correlations)
        else:
            interpretations.append("No significant correlations found at α = 0.05 level.")
        
        return " ".join(interpretations)
    
    def _correlation_strength(self, r: float) -> str:
        """Classify correlation strength"""
        if r >= 0.7:
            return "strong"
        elif r >= 0.3:
            return "moderate"
        else:
            return "weak"
    
    def _comprehensive_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        try:
            # Combine multiple analyses
            results = {
                "descriptive": self._descriptive_statistics(data, query, **kwargs)["result"],
                "correlations": None,
                "outliers": None,
                "normality_tests": None
            }
            
            # Add correlation analysis if enough numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_result = self._correlation_analysis(data, query, **kwargs)
                if corr_result["success"]:
                    results["correlations"] = corr_result["result"]
            
            # Add outlier detection
            outlier_result = self._outlier_analysis(data, query, **kwargs)
            if outlier_result["success"]:
                results["outliers"] = outlier_result["result"]
            
            # Add normality tests for numeric columns
            normality_result = self._normality_test(data, query, **kwargs)
            if normality_result["success"]:
                results["normality_tests"] = normality_result["result"]
            
            return {
                "success": True,
                "result": results,
                "agent": "StatisticalAgent",
                "operation": "comprehensive_analysis",
                "interpretation": "Comprehensive statistical analysis completed with descriptive statistics, correlation analysis, outlier detection, and normality testing."
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Comprehensive analysis failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _outlier_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Detect and analyze outliers using multiple methods"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            outlier_results = {}
            
            for col in numeric_data.columns:
                col_data = numeric_data[col].dropna()
                if len(col_data) < 4:  # Need minimum data points
                    continue
                
                # IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                # Z-score method
                z_scores = np.abs(scipy_stats.zscore(col_data))
                zscore_outliers = col_data[z_scores > 3]
                
                # Modified Z-score method
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                modified_z_scores = 0.6745 * (col_data - median) / mad
                modified_zscore_outliers = col_data[np.abs(modified_z_scores) > 3.5]
                
                outlier_results[col] = {
                    "iqr_method": {
                        "outliers": iqr_outliers.tolist(),
                        "count": len(iqr_outliers),
                        "percentage": len(iqr_outliers) / len(col_data) * 100,
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                    },
                    "zscore_method": {
                        "outliers": zscore_outliers.tolist(),
                        "count": len(zscore_outliers),
                        "percentage": len(zscore_outliers) / len(col_data) * 100
                    },
                    "modified_zscore_method": {
                        "outliers": modified_zscore_outliers.tolist(),
                        "count": len(modified_zscore_outliers),
                        "percentage": len(modified_zscore_outliers) / len(col_data) * 100
                    }
                }
            
            return {
                "success": True,
                "result": outlier_results,
                "agent": "StatisticalAgent",
                "operation": "outlier_analysis",
                "interpretation": self._interpret_outliers(outlier_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Outlier analysis failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _normality_test(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Test for normality using multiple statistical tests"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            normality_results = {}
            
            for col in numeric_data.columns:
                col_data = numeric_data[col].dropna()
                if len(col_data) < 8:  # Need minimum sample size
                    continue
                
                results = {}
                
                # Shapiro-Wilk test (best for n < 5000)
                if len(col_data) <= 5000:
                    shapiro_stat, shapiro_p = scipy_stats.shapiro(col_data)
                    results["shapiro_wilk"] = {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > self.alpha
                    }
                
                # D'Agostino and Pearson's test
                try:
                    dagostino_stat, dagostino_p = scipy_stats.normaltest(col_data)
                    results["dagostino_pearson"] = {
                        "statistic": float(dagostino_stat),
                        "p_value": float(dagostino_p),
                        "is_normal": dagostino_p > self.alpha
                    }
                except:
                    logging.debug("Operation failed (non-critical) - continuing")
                
                # Kolmogorov-Smirnov test
                try:
                    # Compare against normal distribution with same mean and std
                    ks_stat, ks_p = scipy_stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                    results["kolmogorov_smirnov"] = {
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "is_normal": ks_p > self.alpha
                    }
                except:
                    logging.debug("Operation failed (non-critical) - continuing")
                
                # Anderson-Darling test
                try:
                    ad_result = scipy_stats.anderson(col_data, dist='norm')
                    results["anderson_darling"] = {
                        "statistic": float(ad_result.statistic),
                        "critical_values": ad_result.critical_values.tolist(),
                        "significance_levels": ad_result.significance_level.tolist()
                    }
                except:
                    logging.debug("Operation failed (non-critical) - continuing")
                
                normality_results[col] = results
            
            return {
                "success": True,
                "result": normality_results,
                "agent": "StatisticalAgent",
                "operation": "normality_testing",
                "interpretation": self._interpret_normality(normality_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Normality testing failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _interpret_outliers(self, outlier_results: Dict) -> str:
        """Generate interpretation of outlier analysis"""
        interpretations = []
        
        for col, methods in outlier_results.items():
            iqr_count = methods["iqr_method"]["count"]
            iqr_pct = methods["iqr_method"]["percentage"]
            
            if iqr_count > 0:
                interpretations.append(f"{col}: {iqr_count} outliers detected ({iqr_pct:.1f}%) using IQR method.")
            else:
                interpretations.append(f"{col}: No outliers detected using IQR method.")
        
        return " ".join(interpretations)
    
    def _interpret_normality(self, normality_results: Dict) -> str:
        """Generate interpretation of normality tests"""
        interpretations = []
        
        for col, tests in normality_results.items():
            if "shapiro_wilk" in tests:
                sw_result = tests["shapiro_wilk"]
                if sw_result["is_normal"]:
                    interpretations.append(f"{col}: Passes normality test (Shapiro-Wilk p={sw_result['p_value']:.3f}).")
                else:
                    interpretations.append(f"{col}: Fails normality test (Shapiro-Wilk p={sw_result['p_value']:.3f}).")
        
        return " ".join(interpretations)

    # Additional statistical test implementations
    
    def _t_test_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform t-test analysis (independent samples, paired, or one-sample)"""
        try:
            # Extract parameters from query or kwargs
            group_column = kwargs.get('group_column')
            value_column = kwargs.get('value_column')
            test_type = kwargs.get('test_type', 'independent')  # independent, paired, one_sample
            alternative = kwargs.get('alternative', 'two-sided')  # two-sided, less, greater
            
            # Auto-detect columns if not specified
            if not group_column or not value_column:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if len(numeric_cols) > 0:
                    value_column = numeric_cols[0]
                if len(categorical_cols) > 0:
                    group_column = categorical_cols[0]
            
            if not value_column:
                return {
                    "success": False,
                    "error": "No numeric column found for t-test",
                    "agent": "StatisticalAgent"
                }
            
            results = {}
            
            if test_type == 'independent' and group_column:
                # Independent samples t-test (two groups)
                groups = data[group_column].unique()
                
                if len(groups) != 2:
                    return {
                        "success": False,
                        "error": f"Independent t-test requires exactly 2 groups, found {len(groups)}",
                        "agent": "StatisticalAgent"
                    }
                
                group1_data = data[data[group_column] == groups[0]][value_column].dropna()
                group2_data = data[data[group_column] == groups[1]][value_column].dropna()
                
                # Perform t-test
                t_stat, p_value = scipy_stats.ttest_ind(group1_data, group2_data, alternative=alternative)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(group1_data)-1)*group1_data.std()**2 + 
                                     (len(group2_data)-1)*group2_data.std()**2) / 
                                    (len(group1_data) + len(group2_data) - 2))
                cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
                
                # Confidence interval for mean difference
                se_diff = pooled_std * np.sqrt(1/len(group1_data) + 1/len(group2_data))
                ci_lower = (group1_data.mean() - group2_data.mean()) - scipy_stats.t.ppf(1-self.alpha/2, len(group1_data)+len(group2_data)-2) * se_diff
                ci_upper = (group1_data.mean() - group2_data.mean()) + scipy_stats.t.ppf(1-self.alpha/2, len(group1_data)+len(group2_data)-2) * se_diff
                
                results = {
                    "test_type": "Independent Samples T-Test",
                    "groups": {
                        groups[0]: {
                            "n": len(group1_data),
                            "mean": float(group1_data.mean()),
                            "std": float(group1_data.std()),
                            "se": float(group1_data.std() / np.sqrt(len(group1_data)))
                        },
                        groups[1]: {
                            "n": len(group2_data),
                            "mean": float(group2_data.mean()),
                            "std": float(group2_data.std()),
                            "se": float(group2_data.std() / np.sqrt(len(group2_data)))
                        }
                    },
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "degrees_of_freedom": len(group1_data) + len(group2_data) - 2,
                    "cohens_d": float(cohens_d),
                    "effect_size_interpretation": self._interpret_cohens_d(cohens_d),
                    "mean_difference": float(group1_data.mean() - group2_data.mean()),
                    "confidence_interval": {
                        "level": self.confidence_level,
                        "lower": float(ci_lower),
                        "upper": float(ci_upper)
                    },
                    "significant": p_value < self.alpha,
                    "alternative": alternative
                }
                
            elif test_type == 'paired':
                # Paired samples t-test
                # Assumes data has two columns to compare
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) < 2:
                    return {
                        "success": False,
                        "error": "Paired t-test requires at least 2 numeric columns",
                        "agent": "StatisticalAgent"
                    }
                
                col1, col2 = numeric_cols[0], numeric_cols[1]
                paired_data = data[[col1, col2]].dropna()
                
                t_stat, p_value = scipy_stats.ttest_rel(paired_data[col1], paired_data[col2], alternative=alternative)
                
                differences = paired_data[col1] - paired_data[col2]
                cohens_d = differences.mean() / differences.std()
                
                results = {
                    "test_type": "Paired Samples T-Test",
                    "columns_compared": [col1, col2],
                    "n_pairs": len(paired_data),
                    "mean_difference": float(differences.mean()),
                    "std_difference": float(differences.std()),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "degrees_of_freedom": len(paired_data) - 1,
                    "cohens_d": float(cohens_d),
                    "effect_size_interpretation": self._interpret_cohens_d(cohens_d),
                    "significant": p_value < self.alpha,
                    "alternative": alternative
                }
                
            else:
                # One-sample t-test
                test_value = kwargs.get('population_mean', 0)
                sample_data = data[value_column].dropna()
                
                t_stat, p_value = scipy_stats.ttest_1samp(sample_data, test_value, alternative=alternative)
                
                cohens_d = (sample_data.mean() - test_value) / sample_data.std()
                
                results = {
                    "test_type": "One-Sample T-Test",
                    "sample_column": value_column,
                    "n": len(sample_data),
                    "sample_mean": float(sample_data.mean()),
                    "sample_std": float(sample_data.std()),
                    "test_value": float(test_value),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "degrees_of_freedom": len(sample_data) - 1,
                    "cohens_d": float(cohens_d),
                    "effect_size_interpretation": self._interpret_cohens_d(cohens_d),
                    "significant": p_value < self.alpha,
                    "alternative": alternative
                }
            
            return {
                "success": True,
                "result": results,
                "agent": "StatisticalAgent",
                "operation": "t_test_analysis",
                "interpretation": self._interpret_t_test(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"T-test analysis failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_t_test(self, results: Dict) -> str:
        """Generate interpretation of t-test results"""
        test_type = results["test_type"]
        p_value = results["p_value"]
        significant = results["significant"]
        effect_size = results["effect_size_interpretation"]
        
        if significant:
            interpretation = f"{test_type}: Significant difference found (p={p_value:.4f} < {self.alpha}). "
            interpretation += f"Effect size is {effect_size} (Cohen's d={results['cohens_d']:.3f})."
        else:
            interpretation = f"{test_type}: No significant difference found (p={p_value:.4f} ≥ {self.alpha})."
        
        return interpretation
    
    def _anova_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform one-way ANOVA analysis to compare means across multiple groups"""
        try:
            # Extract parameters from query or kwargs
            group_column = kwargs.get('group_column')
            value_column = kwargs.get('value_column')
            
            # Auto-detect columns if not specified
            if not group_column or not value_column:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if len(numeric_cols) > 0:
                    value_column = numeric_cols[0]
                if len(categorical_cols) > 0:
                    group_column = categorical_cols[0]
            
            if not value_column or not group_column:
                return {
                    "success": False,
                    "error": "Need both categorical (group) and numeric (value) columns for ANOVA",
                    "agent": "StatisticalAgent"
                }
            
            # Get groups
            groups = data[group_column].unique()
            
            if len(groups) < 2:
                return {
                    "success": False,
                    "error": f"ANOVA requires at least 2 groups, found {len(groups)}",
                    "agent": "StatisticalAgent"
                }
            
            # Extract data for each group
            group_data = []
            group_stats = {}
            
            for group in groups:
                group_values = data[data[group_column] == group][value_column].dropna()
                group_data.append(group_values)
                group_stats[str(group)] = {
                    "n": len(group_values),
                    "mean": float(group_values.mean()),
                    "std": float(group_values.std()),
                    "se": float(group_values.std() / np.sqrt(len(group_values))),
                    "min": float(group_values.min()),
                    "max": float(group_values.max())
                }
            
            # Perform one-way ANOVA
            f_stat, p_value = scipy_stats.f_oneway(*group_data)
            
            # Calculate eta-squared (effect size for ANOVA)
            all_data = data[[group_column, value_column]].dropna()
            grand_mean = all_data[value_column].mean()
            
            ss_between = sum(len(group_data[i]) * (group_stats[str(groups[i])]["mean"] - grand_mean)**2 
                           for i in range(len(groups)))
            ss_total = sum((all_data[value_column] - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Post-hoc pairwise comparisons (Tukey HSD) if significant
            posthoc_results = None
            if p_value < self.alpha and len(groups) > 2:
                posthoc_results = {}
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        # Pairwise t-test with Bonferroni correction
                        t_stat, t_p = scipy_stats.ttest_ind(group_data[i], group_data[j])
                        bonferroni_p = min(t_p * (len(groups) * (len(groups)-1) / 2), 1.0)  # Bonferroni correction
                        
                        posthoc_results[f"{groups[i]}_vs_{groups[j]}"] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(t_p),
                            "bonferroni_p": float(bonferroni_p),
                            "significant": bonferroni_p < self.alpha,
                            "mean_difference": float(group_stats[str(groups[i])]["mean"] - group_stats[str(groups[j])]["mean"])
                        }
            
            results = {
                "test_type": "One-Way ANOVA",
                "group_column": group_column,
                "value_column": value_column,
                "n_groups": len(groups),
                "groups": list(groups),
                "group_statistics": group_stats,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": {
                    "between": len(groups) - 1,
                    "within": len(all_data) - len(groups),
                    "total": len(all_data) - 1
                },
                "eta_squared": float(eta_squared),
                "effect_size_interpretation": self._interpret_eta_squared(eta_squared),
                "significant": p_value < self.alpha,
                "posthoc_comparisons": posthoc_results
            }
            
            return {
                "success": True,
                "result": results,
                "agent": "StatisticalAgent",
                "operation": "anova_analysis",
                "interpretation": self._interpret_anova(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"ANOVA analysis failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size"""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_anova(self, results: Dict) -> str:
        """Generate interpretation of ANOVA results"""
        p_value = results["p_value"]
        significant = results["significant"]
        effect_size = results["effect_size_interpretation"]
        n_groups = results["n_groups"]
        
        interpretation = f"One-Way ANOVA comparing {n_groups} groups: "
        
        if significant:
            interpretation += f"Significant difference found (F={results['f_statistic']:.3f}, p={p_value:.4f}). "
            interpretation += f"Effect size is {effect_size} (η²={results['eta_squared']:.3f})."
            
            if results["posthoc_comparisons"]:
                sig_comparisons = [k for k, v in results["posthoc_comparisons"].items() if v["significant"]]
                if sig_comparisons:
                    interpretation += f" Post-hoc tests reveal {len(sig_comparisons)} significant pairwise difference(s)."
        else:
            interpretation += f"No significant difference found (F={results['f_statistic']:.3f}, p={p_value:.4f})."
        
        return interpretation
    
    def _regression_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform linear regression analysis"""
        try:
            # Extract parameters
            dependent_var = kwargs.get('dependent_variable') or kwargs.get('y_variable')
            independent_vars = kwargs.get('independent_variables') or kwargs.get('x_variables')
            
            # Auto-detect if not specified
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not dependent_var and len(numeric_cols) > 0:
                # Last numeric column as dependent variable (common convention)
                dependent_var = numeric_cols[-1]
                independent_vars = numeric_cols[:-1] if len(numeric_cols) > 1 else None
            
            if not dependent_var:
                return {
                    "success": False,
                    "error": "No numeric column found for regression analysis",
                    "agent": "StatisticalAgent"
                }
            
            if not independent_vars:
                # Use all other numeric columns as predictors
                independent_vars = [col for col in numeric_cols if col != dependent_var]
            
            if not independent_vars:
                return {
                    "success": False,
                    "error": "Need at least one independent variable for regression",
                    "agent": "StatisticalAgent"
                }
            
            # Prepare data (remove missing values)
            regression_data = data[[dependent_var] + independent_vars].dropna()
            
            if len(regression_data) < len(independent_vars) + 2:
                return {
                    "success": False,
                    "error": f"Insufficient data points for regression (need at least {len(independent_vars) + 2})",
                    "agent": "StatisticalAgent"
                }
            
            y = regression_data[dependent_var].values
            X = regression_data[independent_vars].values
            
            # Add intercept term
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            
            # Calculate coefficients using OLS
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            # Predictions
            y_pred = X_with_intercept @ coefficients
            residuals = y - y_pred
            
            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Adjusted R-squared
            n = len(y)
            p = len(independent_vars)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            
            # Standard error of residuals
            mse = ss_res / (n - p - 1)
            rmse = np.sqrt(mse)
            
            # F-statistic for overall model significance
            ms_reg = ss_tot - ss_res
            if p > 0:
                f_statistic = (ms_reg / p) / mse
                f_pvalue = 1 - scipy_stats.f.cdf(f_statistic, p, n - p - 1)
            else:
                f_statistic = 0
                f_pvalue = 1
            
            # Coefficient statistics
            # Standard errors of coefficients
            X_transpose_X_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se_coefficients = np.sqrt(mse * np.diag(X_transpose_X_inv))
            
            # T-statistics and p-values for each coefficient
            t_statistics = coefficients / se_coefficients
            p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_statistics), n - p - 1))
            
            # Build coefficient results
            coefficient_results = {
                "intercept": {
                    "coefficient": float(coefficients[0]),
                    "std_error": float(se_coefficients[0]),
                    "t_statistic": float(t_statistics[0]),
                    "p_value": float(p_values[0]),
                    "significant": p_values[0] < self.alpha
                }
            }
            
            for i, var in enumerate(independent_vars):
                coefficient_results[var] = {
                    "coefficient": float(coefficients[i+1]),
                    "std_error": float(se_coefficients[i+1]),
                    "t_statistic": float(t_statistics[i+1]),
                    "p_value": float(p_values[i+1]),
                    "significant": p_values[i+1] < self.alpha
                }
            
            # Residual analysis
            residual_analysis = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "skewness": float(scipy_stats.skew(residuals)),
                "kurtosis": float(scipy_stats.kurtosis(residuals))
            }
            
            results = {
                "dependent_variable": dependent_var,
                "independent_variables": independent_vars,
                "n_observations": n,
                "coefficients": coefficient_results,
                "model_statistics": {
                    "r_squared": float(r_squared),
                    "adjusted_r_squared": float(adj_r_squared),
                    "rmse": float(rmse),
                    "mse": float(mse),
                    "f_statistic": float(f_statistic),
                    "f_pvalue": float(f_pvalue),
                    "model_significant": f_pvalue < self.alpha
                },
                "residual_analysis": residual_analysis,
                "predictions": {
                    "predicted_values": y_pred.tolist()[:10],  # First 10 predictions
                    "residuals": residuals.tolist()[:10]  # First 10 residuals
                }
            }
            
            return {
                "success": True,
                "result": results,
                "agent": "StatisticalAgent",
                "operation": "regression_analysis",
                "interpretation": self._interpret_regression(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Regression analysis failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _interpret_regression(self, results: Dict) -> str:
        """Generate interpretation of regression results"""
        r_squared = results["model_statistics"]["r_squared"]
        adj_r_squared = results["model_statistics"]["adjusted_r_squared"]
        f_pvalue = results["model_statistics"]["f_pvalue"]
        model_sig = results["model_statistics"]["model_significant"]
        
        interpretation = f"Linear Regression Analysis: "
        
        if model_sig:
            interpretation += f"Overall model is significant (F p-value={f_pvalue:.4f}). "
            interpretation += f"Model explains {r_squared*100:.1f}% of variance (R²={r_squared:.3f}, Adjusted R²={adj_r_squared:.3f}). "
            
            sig_predictors = [k for k, v in results["coefficients"].items() 
                            if k != "intercept" and v["significant"]]
            if sig_predictors:
                interpretation += f"Significant predictors: {', '.join(sig_predictors)}."
        else:
            interpretation += f"Overall model is not significant (F p-value={f_pvalue:.4f})."
        
        return interpretation
    
    def _chi_square_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform chi-square test of independence for categorical variables"""
        try:
            # Extract parameters
            var1 = kwargs.get('variable1') or kwargs.get('row_variable')
            var2 = kwargs.get('variable2') or kwargs.get('column_variable')
            
            # Auto-detect categorical columns if not specified
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not var1 or not var2:
                if len(categorical_cols) >= 2:
                    var1 = categorical_cols[0]
                    var2 = categorical_cols[1]
                else:
                    return {
                        "success": False,
                        "error": "Need at least 2 categorical columns for chi-square test",
                        "agent": "StatisticalAgent"
                    }
            
            # Create contingency table
            contingency_table = pd.crosstab(data[var1], data[var2])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected_freq = scipy_stats.chi2_contingency(contingency_table)
            
            # Calculate effect size (Cramér's V)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
            cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
            
            # Calculate standardized residuals to identify which cells contribute most
            standardized_residuals = (contingency_table.values - expected_freq) / np.sqrt(expected_freq)
            
            # Find cells with largest standardized residuals (|z| > 2 is noteworthy)
            significant_cells = []
            for i in range(contingency_table.shape[0]):
                for j in range(contingency_table.shape[1]):
                    std_res = standardized_residuals[i, j]
                    if abs(std_res) > 2:
                        significant_cells.append({
                            "row": contingency_table.index[i],
                            "column": contingency_table.columns[j],
                            "observed": int(contingency_table.iloc[i, j]),
                            "expected": float(expected_freq[i, j]),
                            "standardized_residual": float(std_res),
                            "interpretation": "more than expected" if std_res > 0 else "less than expected"
                        })
            
            results = {
                "test_type": "Chi-Square Test of Independence",
                "variables": [var1, var2],
                "contingency_table": contingency_table.to_dict(),
                "table_shape": contingency_table.shape,
                "chi2_statistic": float(chi2_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "cramers_v": float(cramers_v),
                "effect_size_interpretation": self._interpret_cramers_v(cramers_v, min_dim),
                "significant": p_value < self.alpha,
                "expected_frequencies": pd.DataFrame(expected_freq, 
                                                     index=contingency_table.index,
                                                     columns=contingency_table.columns).to_dict(),
                "significant_cells": significant_cells
            }
            
            return {
                "success": True,
                "result": results,
                "agent": "StatisticalAgent",
                "operation": "chi_square_analysis",
                "interpretation": self._interpret_chi_square(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Chi-square analysis failed: {str(e)}",
                "agent": "StatisticalAgent"
            }
    
    def _interpret_cramers_v(self, v: float, min_dim: int) -> str:
        """Interpret Cramér's V effect size (depends on degrees of freedom)"""
        if min_dim == 1:
            # For 2x2 tables
            if v < 0.1:
                return "negligible"
            elif v < 0.3:
                return "small"
            elif v < 0.5:
                return "medium"
            else:
                return "large"
        else:
            # For larger tables
            if v < 0.07:
                return "negligible"
            elif v < 0.21:
                return "small"
            elif v < 0.35:
                return "medium"
            else:
                return "large"
    
    def _interpret_chi_square(self, results: Dict) -> str:
        """Generate interpretation of chi-square results"""
        p_value = results["p_value"]
        significant = results["significant"]
        effect_size = results["effect_size_interpretation"]
        var1, var2 = results["variables"]
        
        interpretation = f"Chi-Square Test: Testing independence between {var1} and {var2}. "
        
        if significant:
            interpretation += f"Variables are significantly associated (χ²={results['chi2_statistic']:.3f}, p={p_value:.4f}). "
            interpretation += f"Effect size is {effect_size} (Cramér's V={results['cramers_v']:.3f})."
            
            if results["significant_cells"]:
                interpretation += f" {len(results['significant_cells'])} cell(s) show notable deviations from expected frequencies."
        else:
            interpretation += f"No significant association found (χ²={results['chi2_statistic']:.3f}, p={p_value:.4f})."
        
        return interpretation
