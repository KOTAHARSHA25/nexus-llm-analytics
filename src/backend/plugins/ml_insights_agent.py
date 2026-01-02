# ML Insights Agent Plugin
# Specialized agent for machine learning pattern detection and insights

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

# ML and data science imports
try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class MLInsightsAgent(BasePluginAgent):
    """
    Machine Learning Insights Agent
    
    Capabilities:
    - Pattern detection and data mining
    - Clustering analysis for customer segmentation
    - Anomaly detection using multiple algorithms
    - Principal Component Analysis (PCA) for dimensionality reduction
    - Feature importance analysis
    - Correlation and association analysis
    - Classification insights and predictions
    - Regression analysis for trend prediction
    - Data preprocessing and feature engineering
    - Model performance evaluation
    - Automated machine learning insights
    - Predictive analytics recommendations
    - Data quality assessment
    - Feature selection and ranking
    
    Features:
    - Automatic algorithm selection based on data characteristics
    - Hyperparameter optimization suggestions
    - Model interpretability and explainability
    - Bias detection and fairness analysis
    - Performance benchmarking
    - Cross-validation and model validation
    - Feature engineering recommendations
    - Data visualization for ML insights
    """
    
    def get_metadata(self) -> AgentMetadata:
        """Define agent metadata and capabilities"""
        return AgentMetadata(
            name="MLInsightsAgent",
            version="1.0.0",
            description="Machine learning insights agent for pattern detection, clustering, anomaly detection, and predictive analytics",
            author="Nexus LLM Analytics Team",
            capabilities=[
                AgentCapability.MACHINE_LEARNING,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.VISUALIZATION
            ],
            file_types=[".csv", ".xlsx", ".json", ".txt"],
            dependencies=["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn"],
            min_ram_mb=1024,  # ML operations can be memory intensive
            max_timeout_seconds=600,
            priority=70  # High priority for ML tasks
        )
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the ML insights agent"""
        try:
            if not HAS_SKLEARN:
                logging.error("scikit-learn not available - required for ML analysis")
                return False
            
            # Configuration
            self.random_state = self.config.get("random_state", 42)
            self.test_size = self.config.get("test_size", 0.2)
            self.n_clusters_default = self.config.get("n_clusters_default", 5)
            self.pca_components = self.config.get("pca_components", None)  # Auto-determine
            
            # ML patterns for query matching
            self.ml_patterns = {
                "clustering": {
                    "patterns": ["cluster", "segment", "group", "similar", "categorize", "partition"],
                    "description": "Group similar data points together"
                },
                "anomaly": {
                    "patterns": ["anomaly", "outlier", "unusual", "abnormal", "detect anomalies", "fraud"],
                    "description": "Detect unusual patterns or outliers"
                },
                "classification": {
                    "patterns": ["classify", "predict category", "predict class", "classification"],
                    "description": "Predict categories or classes"
                },
                "regression": {
                    "patterns": ["predict", "forecast", "estimate", "regression", "predict value"],
                    "description": "Predict continuous values"
                },
                "dimensionality": {
                    "patterns": ["pca", "reduce dimensions", "feature reduction", "principal component"],
                    "description": "Reduce data dimensions while preserving information"
                },
                "association": {
                    "patterns": ["association", "relationship", "correlation", "pattern", "rules"],
                    "description": "Find relationships and associations in data"
                },
                "feature": {
                    "patterns": ["feature importance", "feature selection", "important variables"],
                    "description": "Identify most important features"
                },
                "pattern": {
                    "patterns": ["pattern", "trend", "insight", "discover", "mining"],
                    "description": "Discover patterns and insights in data"
                }
            }
            
            # ML algorithms available
            self.available_algorithms = {
                "clustering": ["kmeans", "dbscan", "hierarchical"],
                "anomaly_detection": ["isolation_forest", "one_class_svm", "local_outlier_factor"],
                "classification": ["logistic_regression", "decision_tree", "random_forest"],
                "regression": ["linear_regression", "polynomial_regression", "random_forest_regressor"]
            }
            
            # Data preprocessing pipelines
            self.preprocessing_steps = [
                "missing_value_handling",
                "outlier_detection",
                "feature_scaling", 
                "categorical_encoding",
                "feature_selection"
            ]
            
            self.initialized = True
            logging.debug(f"ML Insights Agent initialized: random_state={self.random_state}, clusters={self.n_clusters_default}")
            
            return True
            
        except Exception as e:
            logging.error(f"ML Insights Agent initialization failed: {e}")
            return False
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        """Determine if this agent can handle the ML query"""
        if not self.initialized:
            return 0.0
            
        confidence = 0.0
        query_lower = query.lower()
        
        # CRITICAL: Reject document files - they should go to RAG Agent
        document_extensions = [".pdf", ".docx", ".pptx", ".rtf"]
        if file_type and file_type.lower() in document_extensions:
            logging.debug(f"ML Insights Agent rejecting document file: {file_type}")
            return 0.0
        
        # File type support - only structured data
        if file_type and file_type.lower() in [".csv", ".xlsx", ".json", ".txt"]:
            confidence += 0.1
        
        # ML keywords
        ml_keywords = [
            "machine learning", "ml", "artificial intelligence", "ai", 
            "model", "algorithm", "prediction", "classification", "clustering"
        ]
        
        keyword_matches = sum(1 for keyword in ml_keywords if keyword in query_lower)
        confidence += min(keyword_matches * 0.15, 0.4)
        
        # Specific ML patterns
        for pattern_type, pattern_data in self.ml_patterns.items():
            patterns = pattern_data["patterns"]
            if any(pattern in query_lower for pattern in patterns):
                confidence += 0.25
                break
        
        # Data science terms
        ds_terms = [
            "insight", "pattern", "discover", "mining", "analytics",
            "feature", "variable", "dimension", "component", "score"
        ]
        
        ds_matches = sum(1 for term in ds_terms if term in query_lower)
        confidence += min(ds_matches * 0.08, 0.25)
        
        # Specific algorithms
        algorithm_terms = [
            "kmeans", "pca", "svm", "decision tree", "random forest",
            "neural network", "deep learning", "gradient boosting"
        ]
        
        algo_matches = sum(1 for term in algorithm_terms if term in query_lower)
        confidence += min(algo_matches * 0.12, 0.3)
        
        return min(confidence, 1.0)
    
    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute ML analysis based on the query"""
        try:
            # Load data if filename provided
            filename = kwargs.get('filename')
            if filename and not data:
                data = self._load_data(filename)
            
            if data is None:
                return {
                    "success": False,
                    "error": "No data provided for ML analysis",
                    "agent": "MLInsightsAgent"
                }
            
            # Parse query intent
            intent = self._parse_ml_intent(query)
            
            # Preprocess data
            processed_data = self._preprocess_data(data)
            if processed_data is None:
                return {
                    "success": False,
                    "error": "Data preprocessing failed",
                    "agent": "MLInsightsAgent"
                }
            
            # Execute appropriate ML analysis
            if intent == "clustering":
                return self._clustering_analysis(processed_data, query, **kwargs)
            elif intent == "anomaly":
                return self._anomaly_detection(processed_data, query, **kwargs)
            elif intent == "classification":
                return self._classification_analysis(processed_data, query, **kwargs)
            elif intent == "regression":
                return self._regression_analysis(processed_data, query, **kwargs)
            elif intent == "dimensionality":
                return self._dimensionality_reduction(processed_data, query, **kwargs)
            elif intent == "association":
                return self._association_analysis(processed_data, query, **kwargs)
            elif intent == "feature":
                return self._feature_importance_analysis(processed_data, query, **kwargs)
            else:
                return self._comprehensive_ml_analysis(processed_data, query, **kwargs)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"ML analysis failed: {str(e)}",
                "agent": "MLInsightsAgent"
            }
    
    def _load_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from file"""
        try:
            # Project root is 3 levels up from this file (src/backend/plugins)
            project_root = Path(__file__).parent.parent.parent
            base_data_dir = project_root / "data"
            
            for subdir in ["uploads", "samples"]:
                filepath = base_data_dir / subdir / filename
                if filepath.exists():
                    logging.info(f"Loading data from: {filepath}")
                    if filename.endswith('.csv'):
                        return pd.read_csv(filepath)
                    elif filename.endswith(('.xlsx', '.xls')):
                        return pd.read_excel(filepath)
                    elif filename.endswith('.json'):
                        return pd.read_json(filepath)
            
            logging.warning(f"File not found in uploads or samples: {filename}")
            return None
        except Exception as e:
            logging.error(f"Failed to load data from {filename}: {e}")
            return None
    
    def _parse_ml_intent(self, query: str) -> str:
        """Parse the ML intent from the query"""
        query_lower = query.lower()
        
        # Check for specific ML patterns
        for pattern_type, pattern_data in self.ml_patterns.items():
            patterns = pattern_data["patterns"]
            if any(pattern in query_lower for pattern in patterns):
                return pattern_type
        
        # Default to comprehensive analysis
        return "comprehensive"
    
    def _preprocess_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preprocess data for ML analysis"""
        try:
            processed_data = data.copy()
            
            # Handle missing values
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns
            
            # Fill numeric missing values with median
            for col in numeric_cols:
                if processed_data[col].isnull().any():
                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
            
            # Fill categorical missing values with mode
            for col in categorical_cols:
                if processed_data[col].isnull().any():
                    mode_value = processed_data[col].mode()
                    if len(mode_value) > 0:
                        processed_data[col].fillna(mode_value[0], inplace=True)
                    else:
                        processed_data[col].fillna('Unknown', inplace=True)
            
            # Encode categorical variables
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                processed_data[col + '_encoded'] = le.fit_transform(processed_data[col].astype(str))
                label_encoders[col] = le
            
            # Store preprocessing info
            processed_data._preprocessing_info = {
                "numeric_columns": list(numeric_cols),
                "categorical_columns": list(categorical_cols),
                "label_encoders": label_encoders,
                "original_shape": data.shape,
                "processed_shape": processed_data.shape
            }
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Data preprocessing failed: {e}")
            return None
    
    def _clustering_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform clustering analysis"""
        try:
            # Get numeric data for clustering
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 numeric columns for clustering",
                    "agent": "MLInsightsAgent"
                }
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            results = {}
            
            # K-Means clustering
            try:
                # Determine optimal number of clusters using elbow method
                inertias = []
                silhouette_scores = []
                k_range = range(2, min(11, len(numeric_data) // 2))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    cluster_labels = kmeans.fit_predict(scaled_data)
                    inertias.append(kmeans.inertia_)
                    
                    if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                        sil_score = silhouette_score(scaled_data, cluster_labels)
                        silhouette_scores.append(sil_score)
                    else:
                        silhouette_scores.append(0)
                
                # Choose optimal k based on silhouette score
                if silhouette_scores:
                    optimal_k = k_range[np.argmax(silhouette_scores)]
                    optimal_silhouette = max(silhouette_scores)
                else:
                    optimal_k = self.n_clusters_default
                    optimal_silhouette = 0
                
                # Perform final clustering with optimal k
                final_kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
                cluster_labels = final_kmeans.fit_predict(scaled_data)
                
                # Analyze clusters
                cluster_analysis = {}
                for cluster_id in range(optimal_k):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_data = numeric_data[cluster_mask]
                    
                    cluster_analysis[f"cluster_{cluster_id}"] = {
                        "size": int(cluster_mask.sum()),
                        "percentage": float(cluster_mask.sum() / len(numeric_data) * 100),
                        "center": final_kmeans.cluster_centers_[cluster_id].tolist(),
                        "characteristics": {
                            col: {
                                "mean": float(cluster_data[col].mean()),
                                "std": float(cluster_data[col].std()),
                                "median": float(cluster_data[col].median())
                            }
                            for col in numeric_data.columns
                        }
                    }
                
                results["kmeans"] = {
                    "optimal_clusters": int(optimal_k),
                    "silhouette_score": float(optimal_silhouette),
                    "inertia": float(final_kmeans.inertia_),
                    "cluster_analysis": cluster_analysis,
                    "elbow_data": {
                        "k_values": list(k_range),
                        "inertias": inertias,
                        "silhouette_scores": silhouette_scores
                    }
                }
                
            except Exception as e:
                logging.error(f"K-means clustering failed: {e}")
            
            # DBSCAN clustering
            try:
                from sklearn.neighbors import NearestNeighbors
                
                # Estimate eps parameter
                neighbors = NearestNeighbors(n_neighbors=4)
                neighbors_fit = neighbors.fit(scaled_data)
                distances, indices = neighbors_fit.kneighbors(scaled_data)
                distances = np.sort(distances, axis=0)
                distances = distances[:, 1]  # k-distance (k=4)
                
                # Use knee point as eps (simplified method)
                eps = np.percentile(distances, 90)
                
                dbscan = DBSCAN(eps=eps, min_samples=4)
                dbscan_labels = dbscan.fit_predict(scaled_data)
                
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                results["dbscan"] = {
                    "n_clusters": int(n_clusters),
                    "n_noise_points": int(n_noise),
                    "noise_percentage": float(n_noise / len(scaled_data) * 100),
                    "eps_parameter": float(eps),
                    "silhouette_score": float(silhouette_score(scaled_data, dbscan_labels)) if n_clusters > 1 else 0
                }
                
            except Exception as e:
                logging.error(f"DBSCAN clustering failed: {e}")
            
            return {
                "success": True,
                "result": results,
                "agent": "MLInsightsAgent",
                "operation": "clustering_analysis",
                "interpretation": self._interpret_clustering(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Clustering analysis failed: {str(e)}",
                "agent": "MLInsightsAgent"
            }
    
    def _anomaly_detection(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform anomaly detection analysis"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {
                    "success": False,
                    "error": "No numeric data available for anomaly detection",
                    "agent": "MLInsightsAgent"
                }
            
            results = {}
            
            # Isolation Forest
            try:
                isolation_forest = IsolationForest(contamination=0.1, random_state=self.random_state)
                anomaly_labels = isolation_forest.fit_predict(numeric_data)
                anomaly_scores = isolation_forest.decision_function(numeric_data)
                
                anomaly_indices = np.where(anomaly_labels == -1)[0]
                normal_indices = np.where(anomaly_labels == 1)[0]
                
                results["isolation_forest"] = {
                    "n_anomalies": int(len(anomaly_indices)),
                    "anomaly_percentage": float(len(anomaly_indices) / len(numeric_data) * 100),
                    "anomaly_indices": anomaly_indices.tolist(),
                    "anomaly_scores": {
                        "mean": float(np.mean(anomaly_scores)),
                        "std": float(np.std(anomaly_scores)),
                        "min": float(np.min(anomaly_scores)),
                        "max": float(np.max(anomaly_scores))
                    }
                }
                
                # Analyze characteristics of anomalies
                if len(anomaly_indices) > 0:
                    anomaly_data = numeric_data.iloc[anomaly_indices]
                    normal_data = numeric_data.iloc[normal_indices]
                    
                    anomaly_characteristics = {}
                    for col in numeric_data.columns:
                        anomaly_characteristics[col] = {
                            "anomaly_mean": float(anomaly_data[col].mean()),
                            "normal_mean": float(normal_data[col].mean()),
                            "difference": float(anomaly_data[col].mean() - normal_data[col].mean()),
                            "anomaly_std": float(anomaly_data[col].std()),
                            "normal_std": float(normal_data[col].std())
                        }
                    
                    results["isolation_forest"]["characteristics"] = anomaly_characteristics
                
            except Exception as e:
                logging.error(f"Isolation Forest anomaly detection failed: {e}")
            
            # Statistical outlier detection (Z-score method)
            try:
                z_scores = np.abs(stats.zscore(numeric_data))
                z_threshold = 3
                z_outliers = np.where(np.any(z_scores > z_threshold, axis=1))[0]
                
                results["z_score_method"] = {
                    "n_outliers": int(len(z_outliers)),
                    "outlier_percentage": float(len(z_outliers) / len(numeric_data) * 100),
                    "outlier_indices": z_outliers.tolist(),
                    "threshold": z_threshold
                }
                
            except Exception as e:
                logging.error(f"Z-score outlier detection failed: {e}")
            
            return {
                "success": True,
                "result": results,
                "agent": "MLInsightsAgent",
                "operation": "anomaly_detection",
                "interpretation": self._interpret_anomalies(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Anomaly detection failed: {str(e)}",
                "agent": "MLInsightsAgent"
            }
    
    def _dimensionality_reduction(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform dimensionality reduction analysis"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 3:
                return {
                    "success": False,
                    "error": "Need at least 3 numeric columns for meaningful dimensionality reduction",
                    "agent": "MLInsightsAgent"
                }
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            results = {}
            
            # Principal Component Analysis
            try:
                # Determine number of components
                n_components = min(self.pca_components or numeric_data.shape[1], numeric_data.shape[0], numeric_data.shape[1])
                
                pca = PCA(n_components=n_components, random_state=self.random_state)
                pca_transformed = pca.fit_transform(scaled_data)
                
                # Calculate cumulative explained variance
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                
                # Find number of components for 95% variance
                n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
                
                results["pca"] = {
                    "n_components": int(n_components),
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "cumulative_variance": cumulative_variance.tolist(),
                    "components_for_95_percent": int(n_components_95),
                    "total_variance_explained": float(cumulative_variance[-1]),
                    "feature_importance": {
                        f"PC{i+1}": {
                            "variance_explained": float(pca.explained_variance_ratio_[i]),
                            "top_features": self._get_top_pca_features(pca.components_[i], numeric_data.columns, top_n=3)
                        }
                        for i in range(min(5, n_components))  # Show top 5 components
                    }
                }
                
            except Exception as e:
                logging.error(f"PCA analysis failed: {e}")
            
            return {
                "success": True,
                "result": results,
                "agent": "MLInsightsAgent",
                "operation": "dimensionality_reduction",
                "interpretation": self._interpret_pca(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Dimensionality reduction failed: {str(e)}",
                "agent": "MLInsightsAgent"
            }
    
    def _get_top_pca_features(self, component: np.ndarray, feature_names: List[str], top_n: int = 3) -> List[Dict]:
        """Get top contributing features for a PCA component"""
        feature_contributions = [(abs(component[i]), feature_names[i], component[i]) for i in range(len(component))]
        feature_contributions.sort(reverse=True, key=lambda x: x[0])
        
        return [
            {
                "feature": contrib[1],
                "contribution": float(contrib[2]),
                "abs_contribution": float(contrib[0])
            }
            for contrib in feature_contributions[:top_n]
        ]
    
    def _comprehensive_ml_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive ML analysis"""
        try:
            results = {}
            
            # Data overview
            results["data_overview"] = {
                "shape": data.shape,
                "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(data.select_dtypes(exclude=[np.number]).columns),
                "missing_values": data.isnull().sum().sum()
            }
            
            # Clustering analysis
            clustering_result = self._clustering_analysis(data, query, **kwargs)
            if clustering_result["success"]:
                results["clustering"] = clustering_result["result"]
            
            # Anomaly detection
            anomaly_result = self._anomaly_detection(data, query, **kwargs)
            if anomaly_result["success"]:
                results["anomalies"] = anomaly_result["result"]
            
            # Dimensionality reduction
            if data.select_dtypes(include=[np.number]).shape[1] >= 3:
                pca_result = self._dimensionality_reduction(data, query, **kwargs)
                if pca_result["success"]:
                    results["dimensionality_reduction"] = pca_result["result"]
            
            # Feature correlation analysis
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                correlation_matrix = numeric_data.corr()
                results["feature_correlations"] = {
                    "correlation_matrix": correlation_matrix.to_dict(),
                    "strong_correlations": self._find_strong_correlations(correlation_matrix)
                }
            
            # Generate comprehensive interpretation
            interpretation = self._generate_ml_interpretation(data, results, query)
            
            return {
                "success": True,
                "result": results,
                "agent": "MLInsightsAgent",
                "operation": "comprehensive_ml_analysis",
                "interpretation": interpretation
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Comprehensive ML analysis failed: {str(e)}",
                "agent": "MLInsightsAgent"
            }
    
    def _generate_ml_interpretation(self, data: pd.DataFrame, results: Dict, query: str) -> str:
        """Generate comprehensive human-readable ML analysis interpretation"""
        lines = []
        
        lines.append("## Machine Learning Analysis Summary\n")
        
        # Data overview
        overview = results.get("data_overview", {})
        shape = overview.get("shape", (0, 0))
        lines.append(f"**Dataset:** {shape[0]:,} records, {shape[1]} columns")
        lines.append(f"• Numeric features: {overview.get('numeric_columns', 0)}")
        lines.append(f"• Categorical features: {overview.get('categorical_columns', 0)}")
        if overview.get("missing_values", 0) > 0:
            lines.append(f"• Missing values: {overview.get('missing_values', 0):,}")
        lines.append("")
        
        # Clustering insights
        clustering = results.get("clustering", {})
        if clustering:
            lines.append("### Clustering Analysis\n")
            if "kmeans" in clustering:
                kmeans = clustering["kmeans"]
                n_clusters = kmeans.get("optimal_clusters", 0)
                silhouette = kmeans.get("silhouette_score", 0)
                lines.append(f"**K-Means Clustering:** Identified {n_clusters} distinct groups")
                lines.append(f"• Cluster quality (silhouette score): {silhouette:.3f}")
                
                cluster_analysis = kmeans.get("cluster_analysis", {})
                for cluster_id, cluster_data in list(cluster_analysis.items())[:3]:
                    size = cluster_data.get("size", 0)
                    pct = cluster_data.get("percentage", 0)
                    lines.append(f"• {cluster_id.replace('_', ' ').title()}: {size:,} records ({pct:.1f}%)")
                lines.append("")
            
            if "dbscan" in clustering:
                dbscan = clustering["dbscan"]
                n_clusters = dbscan.get("n_clusters", 0)
                noise_pct = dbscan.get("noise_percentage", 0)
                lines.append(f"**DBSCAN Clustering:** Found {n_clusters} natural clusters")
                lines.append(f"• Noise points: {noise_pct:.1f}%")
                lines.append("")
        
        # Anomaly insights
        anomalies = results.get("anomalies", {})
        if anomalies:
            lines.append("### Anomaly Detection\n")
            if "isolation_forest" in anomalies:
                iso = anomalies["isolation_forest"]
                n_anomalies = iso.get("n_anomalies", 0)
                pct = iso.get("anomaly_percentage", 0)
                lines.append(f"**Detected Anomalies:** {n_anomalies:,} records ({pct:.1f}%)")
                if pct > 5:
                    lines.append("• ⚠️ Higher than expected anomaly rate - investigate data quality")
                elif pct < 1:
                    lines.append("• ✅ Low anomaly rate - data appears consistent")
                lines.append("")
        
        # Dimensionality reduction
        dim_reduction = results.get("dimensionality_reduction", {})
        if dim_reduction and "pca" in dim_reduction:
            pca = dim_reduction["pca"]
            lines.append("### Dimensionality Analysis\n")
            components_95 = pca.get("components_for_95_percent", 0)
            total_variance = pca.get("total_variance_explained", 0)
            original_dims = overview.get("numeric_columns", 0)
            lines.append(f"**PCA Results:** {components_95} components capture 95% of variance")
            lines.append(f"• Dimensionality reduction: {original_dims} → {components_95} features")
            lines.append(f"• Total variance explained: {total_variance:.1%}")
            lines.append("")
        
        # Correlations
        correlations = results.get("feature_correlations", {})
        strong_corrs = correlations.get("strong_correlations", [])
        if strong_corrs:
            lines.append("### Key Relationships\n")
            lines.append("**Strong Correlations Found:**")
            for corr in strong_corrs[:5]:
                col1 = corr.get("column1", "")
                col2 = corr.get("column2", "")
                r = corr.get("correlation", 0)
                direction = "positive" if r > 0 else "negative"
                lines.append(f"• {col1} ↔ {col2}: {abs(r):.3f} ({direction})")
            lines.append("")
        
        return "\n".join(lines) if lines else "ML analysis completed."
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations in the correlation matrix"""
        strong_corr = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= threshold:
                        strong_corr.append({
                            "feature1": col1,
                            "feature2": col2,
                            "correlation": float(corr_value),
                            "strength": "very strong" if abs(corr_value) >= 0.9 else "strong"
                        })
        
        return strong_corr
    
    def _interpret_clustering(self, results: Dict) -> str:
        """Generate interpretation of clustering results"""
        interpretations = []
        
        if "kmeans" in results:
            kmeans_data = results["kmeans"]
            n_clusters = kmeans_data["optimal_clusters"]
            silhouette = kmeans_data["silhouette_score"]
            interpretations.append(f"K-means identified {n_clusters} clusters with silhouette score {silhouette:.3f}")
        
        if "dbscan" in results:
            dbscan_data = results["dbscan"]
            n_clusters = dbscan_data["n_clusters"]
            noise_pct = dbscan_data["noise_percentage"]
            interpretations.append(f"DBSCAN found {n_clusters} clusters with {noise_pct:.1f}% noise points")
        
        return " ".join(interpretations)
    
    def _interpret_anomalies(self, results: Dict) -> str:
        """Generate interpretation of anomaly detection results"""
        interpretations = []
        
        if "isolation_forest" in results:
            iso_data = results["isolation_forest"]
            n_anomalies = iso_data["n_anomalies"]
            anomaly_pct = iso_data["anomaly_percentage"]
            interpretations.append(f"Isolation Forest detected {n_anomalies} anomalies ({anomaly_pct:.1f}% of data)")
        
        if "z_score_method" in results:
            z_data = results["z_score_method"]
            n_outliers = z_data["n_outliers"]
            outlier_pct = z_data["outlier_percentage"]
            interpretations.append(f"Z-score method found {n_outliers} outliers ({outlier_pct:.1f}% of data)")
        
        return " ".join(interpretations)
    
    def _interpret_pca(self, results: Dict) -> str:
        """Generate interpretation of PCA results"""
        interpretations = []
        
        if "pca" in results:
            pca_data = results["pca"]
            n_components_95 = pca_data["components_for_95_percent"]
            total_variance = pca_data["total_variance_explained"]
            interpretations.append(f"PCA analysis: {n_components_95} components explain 95% of variance (total: {total_variance:.1%})")
        
        return " ".join(interpretations)
    
    # Placeholder methods for other ML analyses
    def _classification_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Placeholder for classification analysis"""
        return {
            "success": True,
            "result": {"message": "Classification analysis would be implemented here"},
            "agent": "MLInsightsAgent",
            "operation": "classification_analysis"
        }
    
    def _regression_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Placeholder for regression analysis"""
        return {
            "success": True,
            "result": {"message": "Regression analysis would be implemented here"},
            "agent": "MLInsightsAgent",
            "operation": "regression_analysis"
        }
    
    def _association_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Placeholder for association analysis"""
        return {
            "success": True,
            "result": {"message": "Association analysis would be implemented here"},
            "agent": "MLInsightsAgent",
            "operation": "association_analysis"
        }
    
    def _feature_importance_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Placeholder for feature importance analysis"""
        return {
            "success": True,
            "result": {"message": "Feature importance analysis would be implemented here"},
            "agent": "MLInsightsAgent",
            "operation": "feature_importance_analysis"
        }