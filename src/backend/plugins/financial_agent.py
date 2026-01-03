# Financial Analysis Agent Plugin
# Specialized agent for financial data analysis and business metrics

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

# Financial analysis imports
try:
    import pandas as pd
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class FinancialAgent(BasePluginAgent):
    """
    Advanced Financial Analysis Agent
    
    Capabilities:
    - Revenue and profitability analysis
    - Cash flow analysis and forecasting
    - Financial ratio calculations (liquidity, profitability, efficiency)
    - ROI and ROE analysis
    - Cost analysis and optimization
    - Budget vs actual performance
    - Customer lifetime value (CLV) calculation
    - Churn rate and retention analysis
    - Sales performance metrics
    - Inventory turnover analysis
    - Working capital management
    - Financial risk assessment
    - Break-even analysis
    - Market share analysis
    - Price sensitivity analysis
    - Financial forecasting and budgeting
    
    Features:
    - Automatic financial metric calculation
    - Industry benchmark comparisons
    - Financial health scoring
    - Risk assessment and alerts
    - Profitability optimization suggestions
    - Cash flow projections
    - Performance dashboards
    - Financial report generation
    """
    
    def get_metadata(self) -> AgentMetadata:
        """Define agent metadata and capabilities"""
        return AgentMetadata(
            name="FinancialAgent",
            version="1.0.0",
            description="Comprehensive financial analysis agent with business metrics, ratio analysis, and profitability assessment",
            author="Nexus LLM Analytics Team",
            capabilities=[
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.REPORTING,
                AgentCapability.VISUALIZATION
            ],
            file_types=[".csv", ".xlsx", ".json", ".txt"],
            dependencies=["pandas", "numpy", "scipy", "matplotlib"],
            min_ram_mb=512,
            max_timeout_seconds=300,
            priority=75  # High priority for financial data
        )
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the financial analysis agent"""
        try:
            # Configuration
            self.currency = self.config.get("currency", "USD")
            self.fiscal_year_start = self.config.get("fiscal_year_start", 1)  # January
            self.industry_benchmarks = self.config.get("industry_benchmarks", {})
            
            # Financial patterns for query matching
            self.financial_patterns = {
                "profitability": {
                    "patterns": ["profit", "profitability", "margin", "earnings", "income", "revenue"],
                    "description": "Analyze profitability metrics and margins"
                },
                "liquidity": {
                    "patterns": ["liquidity", "cash flow", "working capital", "current ratio", "quick ratio"],
                    "description": "Assess liquidity and cash position"
                },
                "efficiency": {
                    "patterns": ["efficiency", "turnover", "utilization", "productivity", "performance"],
                    "description": "Analyze operational efficiency metrics"
                },
                "growth": {
                    "patterns": ["growth", "growth rate", "expansion", "increase", "yoy", "year over year"],
                    "description": "Calculate growth rates and trends"
                },
                "roi": {
                    "patterns": ["roi", "return on investment", "return on equity", "roe", "roa", "return"],
                    "description": "Calculate return on investment metrics"
                },
                "cost": {
                    "patterns": ["cost", "expense", "spending", "budget", "cost analysis"],
                    "description": "Analyze costs and expenses"
                },
                "customer": {
                    "patterns": ["customer", "clv", "lifetime value", "churn", "retention", "acquisition"],
                    "description": "Customer financial metrics and analysis"
                },
                "forecast": {
                    "patterns": ["forecast", "budget", "projection", "plan", "predict", "future"],
                    "description": "Financial forecasting and budgeting"
                }
            }
            
            # Financial metrics definitions
            self.financial_metrics = {
                "profitability": {
                    "gross_margin": {"formula": "(Revenue - COGS) / Revenue", "benchmark": 0.4},
                    "net_margin": {"formula": "Net Income / Revenue", "benchmark": 0.1},
                    "operating_margin": {"formula": "Operating Income / Revenue", "benchmark": 0.15},
                    "ebitda_margin": {"formula": "EBITDA / Revenue", "benchmark": 0.2}
                },
                "liquidity": {
                    "current_ratio": {"formula": "Current Assets / Current Liabilities", "benchmark": 2.0},
                    "quick_ratio": {"formula": "(Current Assets - Inventory) / Current Liabilities", "benchmark": 1.0},
                    "cash_ratio": {"formula": "Cash / Current Liabilities", "benchmark": 0.2}
                },
                "efficiency": {
                    "asset_turnover": {"formula": "Revenue / Total Assets", "benchmark": 1.0},
                    "inventory_turnover": {"formula": "COGS / Average Inventory", "benchmark": 6.0},
                    "receivables_turnover": {"formula": "Revenue / Average Receivables", "benchmark": 12.0}
                },
                "leverage": {
                    "debt_to_equity": {"formula": "Total Debt / Total Equity", "benchmark": 0.5},
                    "debt_to_assets": {"formula": "Total Debt / Total Assets", "benchmark": 0.3},
                    "interest_coverage": {"formula": "EBIT / Interest Expense", "benchmark": 5.0}
                }
            }
            
            # Common financial column patterns
            self.financial_columns = {
                "revenue": ["revenue", "sales", "income", "turnover", "receipts"],
                "cost": ["cost", "expense", "cogs", "cost of goods sold", "expenditure"],
                "profit": ["profit", "net income", "earnings", "margin"],
                "assets": ["assets", "total assets", "current assets", "fixed assets"],
                "liabilities": ["liabilities", "debt", "payables", "current liabilities"],
                "equity": ["equity", "stockholders equity", "owners equity"],
                "cash": ["cash", "cash flow", "operating cash flow", "free cash flow"]
            }
            
            self.initialized = True
            logging.debug(f"Financial Agent initialized: Currency={self.currency}, Fiscal year start=Month {self.fiscal_year_start}")
            
            return True
            
        except Exception as e:
            logging.error(f"Financial Agent initialization failed: {e}")
            return False
    
    def can_handle(self, query: str, file_type: Optional[str] = None, **kwargs) -> float:
        """Determine if this agent can handle the financial query"""
        if not self.initialized:
            return 0.0
            
        confidence = 0.0
        query_lower = query.lower()
        
        # CRITICAL: Reject document files - they should go to RAG Agent
        document_extensions = [".pdf", ".docx", ".pptx", ".rtf"]
        if file_type and file_type.lower() in document_extensions:
            logging.debug(f"Financial Agent rejecting document file: {file_type}")
            return 0.0
        
        # Reject simple calculation queries (let DataAnalyst handle)
        simple_calc_patterns = ["calculate total", "total revenue by category", "sum by"]
        if any(pattern in query_lower for pattern in simple_calc_patterns):
            if not any(word in query_lower for word in ["roi", "profitability", "margin", "ratio"]):
                logging.debug(f"Financial Agent deferring simple calculation to DataAnalyst")
                return 0.2  # Low confidence for simple totals
        
        # File type support - only structured data
        if file_type and file_type.lower() in [".csv", ".xlsx", ".json", ".txt"]:
            confidence += 0.1
        
        # STRICT FINANCIAL CONTEXT - Only handle queries with clear financial domain indicators
        strict_financial_keywords = [
            "financial", "finance", "investment", "portfolio", "stock", "bond",
            "equity", "debt", "asset", "liability", "balance sheet", "income statement",
            "cash flow", "fiscal", "treasury", "securities", "trading"
        ]
        
        strict_financial_matches = sum(1 for keyword in strict_financial_keywords if keyword in query_lower)
        
        # Only boost confidence if query has STRONG financial context
        if strict_financial_matches >= 2:
            confidence += 0.4
        elif strict_financial_matches == 1:
            # Single financial keyword - check if it's truly financial context
            financial_ratios = [
                "current ratio", "quick ratio", "debt to equity", "return on equity",
                "return on assets", "asset turnover", "liquidity ratio"
            ]
            if any(ratio in query_lower for ratio in financial_ratios):
                confidence += 0.3
            else:
                confidence += 0.15  # Lower boost for single keyword
        
        # Currency indicators (strong financial signal)
        monetary_indicators = ["$", "€", "£", "¥"]
        if any(indicator in query_lower for indicator in monetary_indicators):
            confidence += 0.2
        
        # Specific financial analysis patterns
        for pattern_type, pattern_data in self.financial_patterns.items():
            patterns = pattern_data["patterns"]
            if any(pattern in query_lower for pattern in patterns):
                # Only boost if also has financial context
                if strict_financial_matches > 0 or any(indicator in query_lower for indicator in monetary_indicators):
                    confidence += 0.15
                break
        
        return min(confidence, 1.0)
    
    def execute(self, query: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Execute financial analysis based on the query"""
        try:
            # Load data if filename provided
            filename = kwargs.get('filename')
            if filename and not data:
                data = self._load_data(filename)
            
            if data is None:
                return {
                    "success": False,
                    "error": "No data provided for financial analysis",
                    "agent": "FinancialAgent"
                }
            
            # Parse query intent
            intent = self._parse_financial_intent(query)
            
            # Execute appropriate financial analysis
            if intent == "profitability":
                return self._profitability_analysis(data, query, **kwargs)
            elif intent == "liquidity":
                return self._liquidity_analysis(data, query, **kwargs)
            elif intent == "efficiency":
                return self._efficiency_analysis(data, query, **kwargs)
            elif intent == "growth":
                return self._growth_analysis(data, query, **kwargs)
            elif intent == "roi":
                return self._roi_analysis(data, query, **kwargs)
            elif intent == "cost":
                return self._cost_analysis(data, query, **kwargs)
            elif intent == "customer":
                return self._customer_analysis(data, query, **kwargs)
            elif intent == "forecast":
                return self._financial_forecast(data, query, **kwargs)
            else:
                return self._comprehensive_financial_analysis(data, query, **kwargs)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Financial analysis failed: {str(e)}",
                "agent": "FinancialAgent"
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
    
    def _parse_financial_intent(self, query: str) -> str:
        """Parse the financial intent from the query"""
        query_lower = query.lower()
        
        # Check for specific financial patterns
        for pattern_type, pattern_data in self.financial_patterns.items():
            patterns = pattern_data["patterns"]
            if any(pattern in query_lower for pattern in patterns):
                return pattern_type
        
        # Default to comprehensive analysis
        return "comprehensive"
    
    def _identify_financial_columns(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify financial columns in the dataset"""
        column_mapping = {}
        
        for category, patterns in self.financial_columns.items():
            matching_cols = []
            for col in data.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in patterns):
                    matching_cols.append(col)
            
            if matching_cols:
                column_mapping[category] = matching_cols
        
        return column_mapping
    
    def _profitability_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze profitability metrics"""
        try:
            financial_cols = self._identify_financial_columns(data)
            results = {}
            
            # Basic profitability metrics
            if "revenue" in financial_cols and "cost" in financial_cols:
                revenue_cols = financial_cols["revenue"]
                cost_cols = financial_cols["cost"]
                
                for rev_col in revenue_cols:
                    for cost_col in cost_cols:
                        if rev_col in data.columns and cost_col in data.columns:
                            revenue = data[rev_col].sum()
                            costs = data[cost_col].sum()
                            gross_profit = revenue - costs
                            gross_margin = (gross_profit / revenue) * 100 if revenue > 0 else 0
                            
                            results[f"{rev_col}_vs_{cost_col}"] = {
                                "revenue": float(revenue),
                                "costs": float(costs),
                                "gross_profit": float(gross_profit),
                                "gross_margin_percent": float(gross_margin),
                                "profitability_status": "profitable" if gross_profit > 0 else "unprofitable"
                            }
            
            # Profitability by period (if date column exists)
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols and "revenue" in financial_cols:
                try:
                    date_col = date_cols[0]
                    data[date_col] = pd.to_datetime(data[date_col])
                    
                    # Monthly profitability
                    monthly_data = data.groupby(data[date_col].dt.to_period('M')).agg({
                        financial_cols["revenue"][0]: 'sum',
                        financial_cols["cost"][0]: 'sum' if "cost" in financial_cols else lambda x: 0
                    })
                    
                    monthly_data['profit'] = monthly_data[financial_cols["revenue"][0]] - monthly_data.get(financial_cols["cost"][0], 0)
                    monthly_data['margin'] = (monthly_data['profit'] / monthly_data[financial_cols["revenue"][0]]) * 100
                    
                    results["monthly_trends"] = {
                        "data": monthly_data.to_dict(),
                        "avg_monthly_revenue": float(monthly_data[financial_cols["revenue"][0]].mean()),
                        "avg_monthly_margin": float(monthly_data['margin'].mean())
                    }
                except (ValueError, TypeError, KeyError) as e:
                    logging.debug(f"Monthly trends calculation failed: {e}")
            
            # Profitability benchmarking
            benchmark_analysis = {}
            for metric, benchmark_value in [("gross_margin_percent", 40), ("net_margin_percent", 10)]:
                if any(metric in result for result in results.values() if isinstance(result, dict)):
                    avg_metric = np.mean([result.get(metric, 0) for result in results.values() if isinstance(result, dict)])
                    benchmark_analysis[metric] = {
                        "actual": float(avg_metric),
                        "benchmark": benchmark_value,
                        "performance": "above" if avg_metric > benchmark_value else "below",
                        "gap": float(avg_metric - benchmark_value)
                    }
            
            results["benchmark_analysis"] = benchmark_analysis
            
            return {
                "success": True,
                "result": results,
                "agent": "FinancialAgent",
                "operation": "profitability_analysis",
                "interpretation": self._interpret_profitability(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Profitability analysis failed: {str(e)}",
                "agent": "FinancialAgent"
            }
    
    def _growth_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze growth rates and trends"""
        try:
            results = {}
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Find date column for time-based growth analysis
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_cols and len(numeric_cols) > 0:
                date_col = date_cols[0]
                data[date_col] = pd.to_datetime(data[date_col])
                data_sorted = data.sort_values(date_col)
                
                for col in numeric_cols:
                    if data_sorted[col].notna().sum() > 1:
                        # Calculate various growth metrics
                        first_value = data_sorted[col].dropna().iloc[0]
                        last_value = data_sorted[col].dropna().iloc[-1]
                        
                        # Total growth
                        total_growth = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                        
                        # Period-over-period growth
                        values = data_sorted[col].dropna()
                        if len(values) > 1:
                            period_growth = values.pct_change().mean() * 100
                            growth_volatility = values.pct_change().std() * 100
                        else:
                            period_growth = 0
                            growth_volatility = 0
                        
                        # Compound Annual Growth Rate (CAGR) if we have time span
                        try:
                            time_span = (data_sorted[date_col].max() - data_sorted[date_col].min()).days / 365.25
                            if time_span > 0 and first_value > 0:
                                cagr = (((last_value / first_value) ** (1/time_span)) - 1) * 100
                            else:
                                cagr = 0
                        except Exception:
                            cagr = 0
                        
                        results[col] = {
                            "total_growth_percent": float(total_growth),
                            "average_period_growth_percent": float(period_growth),
                            "growth_volatility_percent": float(growth_volatility),
                            "cagr_percent": float(cagr),
                            "start_value": float(first_value),
                            "end_value": float(last_value),
                            "growth_classification": self._classify_growth(total_growth)
                        }
            else:
                # Simple growth analysis without time dimension
                for col in numeric_cols:
                    values = data[col].dropna()
                    if len(values) > 1:
                        growth_rate = values.pct_change().mean() * 100
                        results[col] = {
                            "average_growth_percent": float(growth_rate),
                            "growth_classification": self._classify_growth(growth_rate)
                        }
            
            return {
                "success": True,
                "result": results,
                "agent": "FinancialAgent",
                "operation": "growth_analysis",
                "interpretation": self._interpret_growth(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Growth analysis failed: {str(e)}",
                "agent": "FinancialAgent"
            }
    
    def _classify_growth(self, growth_rate: float) -> str:
        """Classify growth rate"""
        if growth_rate > 20:
            return "high_growth"
        elif growth_rate > 5:
            return "moderate_growth"
        elif growth_rate > 0:
            return "slow_growth"
        elif growth_rate > -5:
            return "slight_decline"
        else:
            return "significant_decline"
    
    def _comprehensive_financial_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive financial analysis"""
        try:
            results = {}
            
            # Identify financial columns
            financial_cols = self._identify_financial_columns(data)
            results["identified_columns"] = financial_cols
            
            # Basic financial summary
            numeric_data = data.select_dtypes(include=[np.number])
            results["summary_statistics"] = {
                col: {
                    "total": float(numeric_data[col].sum()),
                    "average": float(numeric_data[col].mean()),
                    "median": float(numeric_data[col].median()),
                    "std_dev": float(numeric_data[col].std())
                }
                for col in numeric_data.columns
            }
            
            # Profitability analysis
            profitability_result = self._profitability_analysis(data, query, **kwargs)
            if profitability_result["success"]:
                results["profitability"] = profitability_result["result"]
            
            # Growth analysis
            growth_result = self._growth_analysis(data, query, **kwargs)
            if growth_result["success"]:
                results["growth"] = growth_result["result"]
            
            # Financial health assessment
            results["financial_health"] = self._assess_financial_health(data, financial_cols)
            
            # Generate comprehensive interpretation
            interpretation = self._generate_comprehensive_interpretation(data, results, query)
            
            return {
                "success": True,
                "result": results,
                "agent": "FinancialAgent",
                "operation": "comprehensive_financial_analysis",
                "interpretation": interpretation
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Comprehensive financial analysis failed: {str(e)}",
                "agent": "FinancialAgent"
            }
    
    def _assess_financial_health(self, data: pd.DataFrame, financial_cols: Dict) -> Dict[str, Any]:
        """Assess overall financial health"""
        health_score = 0
        max_score = 0
        factors = {}
        
        # Revenue stability
        if "revenue" in financial_cols:
            revenue_col = financial_cols["revenue"][0]
            revenue_data = data[revenue_col].dropna()
            if len(revenue_data) > 1:
                cv = revenue_data.std() / revenue_data.mean() if revenue_data.mean() != 0 else float('inf')
                if cv < 0.2:
                    health_score += 25
                    factors["revenue_stability"] = "excellent"
                elif cv < 0.5:
                    health_score += 15
                    factors["revenue_stability"] = "good"
                else:
                    factors["revenue_stability"] = "volatile"
                max_score += 25
        
        # Positive cash flow/profit
        if "profit" in financial_cols or ("revenue" in financial_cols and "cost" in financial_cols):
            if "profit" in financial_cols:
                profit_data = data[financial_cols["profit"][0]].sum()
            else:
                revenue = data[financial_cols["revenue"][0]].sum()
                costs = data[financial_cols["cost"][0]].sum()
                profit_data = revenue - costs
            
            if profit_data > 0:
                health_score += 25
                factors["profitability"] = "profitable"
            else:
                factors["profitability"] = "unprofitable"
            max_score += 25
        
        # Growth trend
        if "revenue" in financial_cols:
            revenue_data = data[financial_cols["revenue"][0]].dropna()
            if len(revenue_data) > 1:
                growth_rate = revenue_data.pct_change().mean()
                if growth_rate > 0.05:
                    health_score += 25
                    factors["growth_trend"] = "growing"
                elif growth_rate > 0:
                    health_score += 15
                    factors["growth_trend"] = "stable"
                else:
                    factors["growth_trend"] = "declining"
                max_score += 25
        
        # Calculate final health score
        final_score = (health_score / max_score) * 100 if max_score > 0 else 0
        
        return {
            "health_score": float(final_score),
            "health_grade": self._grade_financial_health(final_score),
            "factors": factors,
            "recommendations": self._generate_health_recommendations(factors)
        }
    
    def _grade_financial_health(self, score: float) -> str:
        """Grade financial health based on score"""
        if score >= 80:
            return "A - Excellent"
        elif score >= 70:
            return "B - Good"
        elif score >= 60:
            return "C - Average"
        elif score >= 50:
            return "D - Below Average"
        else:
            return "F - Poor"
    
    def _generate_health_recommendations(self, factors: Dict) -> List[str]:
        """Generate recommendations based on financial health factors"""
        recommendations = []
        
        if factors.get("revenue_stability") == "volatile":
            recommendations.append("Consider diversifying revenue streams to reduce volatility")
        
        if factors.get("profitability") == "unprofitable":
            recommendations.append("Focus on cost reduction or revenue optimization to achieve profitability")
        
        if factors.get("growth_trend") == "declining":
            recommendations.append("Investigate market opportunities and consider strategic initiatives to reverse decline")
        
        if not recommendations:
            recommendations.append("Financial health appears strong - maintain current strategies")
        
        return recommendations
    
    def _generate_comprehensive_interpretation(self, data: pd.DataFrame, results: Dict, query: str) -> str:
        """Generate comprehensive human-readable interpretation of financial analysis"""
        lines = []
        query_lower = query.lower()
        
        lines.append("## Financial Analysis Summary\n")
        lines.append(f"**Dataset:** {data.shape[0]:,} records, {data.shape[1]} columns\n")
        
        # Summary statistics
        summary = results.get("summary_statistics", {})
        if summary:
            lines.append("### Key Metrics\n")
            for col, stats in list(summary.items())[:5]:
                col_display = col.replace('_', ' ').title()
                lines.append(f"**{col_display}:**")
                lines.append(f"• Total: {stats['total']:,.2f}")
                lines.append(f"• Average: {stats['average']:,.2f}")
                lines.append(f"• Median: {stats['median']:,.2f}")
                lines.append("")
        
        # Financial health
        health = results.get("financial_health", {})
        if health:
            score = health.get("health_score", 0)
            rating = health.get("health_rating", "unknown")
            lines.append("### Financial Health\n")
            lines.append(f"**Overall Score:** {score:.1f}/100 ({rating})")
            
            factors = health.get("factors", {})
            if factors:
                lines.append("\n**Key Factors:**")
                for factor, status in factors.items():
                    factor_display = factor.replace('_', ' ').title()
                    lines.append(f"• {factor_display}: {status}")
            
            recommendations = health.get("recommendations", [])
            if recommendations:
                lines.append("\n**Recommendations:**")
                for rec in recommendations[:3]:
                    lines.append(f"• {rec}")
            lines.append("")
        
        # Profitability insights
        profitability = results.get("profitability", {})
        if profitability:
            lines.append("### Profitability Analysis\n")
            for key, data_item in profitability.items():
                if isinstance(data_item, dict) and "gross_margin_percent" in data_item:
                    margin = data_item["gross_margin_percent"]
                    status = data_item.get("profitability_status", "unknown")
                    revenue = data_item.get("revenue", 0)
                    profit = data_item.get("gross_profit", 0)
                    lines.append(f"**{key.replace('_', ' ').title()}:**")
                    lines.append(f"• Revenue: {revenue:,.2f}")
                    lines.append(f"• Gross Profit: {profit:,.2f}")
                    lines.append(f"• Gross Margin: {margin:.1f}%")
                    lines.append(f"• Status: {status.title()}")
                    lines.append("")
        
        # Growth insights
        growth = results.get("growth", {})
        if growth:
            lines.append("### Growth Analysis\n")
            for col, growth_data in growth.items():
                if isinstance(growth_data, dict) and "total_growth_percent" in growth_data:
                    total_growth = growth_data["total_growth_percent"]
                    classification = growth_data.get("growth_classification", "unknown")
                    lines.append(f"**{col.replace('_', ' ').title()}:**")
                    lines.append(f"• Total Growth: {total_growth:.1f}%")
                    lines.append(f"• Classification: {classification.replace('_', ' ').title()}")
                    lines.append("")
        
        return "\n".join(lines) if lines else "Financial analysis completed."
    
    def _interpret_profitability(self, results: Dict) -> str:
        """Generate interpretation of profitability analysis"""
        interpretations = []
        
        for key, data in results.items():
            if isinstance(data, dict) and "gross_margin_percent" in data and "profitability_status" in data:
                margin = data["gross_margin_percent"]
                status = data["profitability_status"]
                interpretations.append(f"{key}: {status} with {margin:.1f}% gross margin")
        
        return " ".join(interpretations) if interpretations else "Profitability analysis completed"
    
    def _interpret_growth(self, results: Dict) -> str:
        """Generate interpretation of growth analysis"""
        interpretations = []
        
        for col, growth_data in results.items():
            if "total_growth_percent" in growth_data:
                growth = growth_data["total_growth_percent"]
                classification = growth_data["growth_classification"]
                interpretations.append(f"{col}: {growth:.1f}% total growth ({classification})")
        
        return " ".join(interpretations)
    
    # Placeholder methods for other financial analyses
    def _liquidity_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze liquidity metrics (Current Ratio, Quick Ratio, Cash Ratio)"""
        try:
            financial_cols = self._identify_financial_columns(data)
            results = {}
            
            # Helper to get sum of column if it exists
            def get_val(category):
                if category in financial_cols:
                    col = financial_cols[category][0]
                    return data[col].sum()
                return 0.0

            # Calculate metrics
            current_assets = get_val("assets")  # Assuming 'assets' column maps to Current Assets for simplicity or user provides 'Current Assets'
            current_liabilities = get_val("liabilities")
            inventory = get_val("inventory") if "inventory" in financial_cols else 0.0
            cash = get_val("cash")

            metrics = {}
            
            if current_liabilities > 0:
                metrics["current_ratio"] = {
                    "value": float(current_assets / current_liabilities),
                    "benchmark": 2.0,
                    "status": "healthy" if (current_assets / current_liabilities) >= 1.5 else "caution"
                }
                
                metrics["quick_ratio"] = {
                    "value": float((current_assets - inventory) / current_liabilities),
                    "benchmark": 1.0,
                    "status": "healthy" if ((current_assets - inventory) / current_liabilities) >= 1.0 else "caution"
                }

                if cash > 0:
                     metrics["cash_ratio"] = {
                        "value": float(cash / current_liabilities),
                        "benchmark": 0.5,
                        "status": "healthy" if (cash / current_liabilities) >= 0.5 else "caution"
                     }
            
            results["liquidity_metrics"] = metrics
            results["raw_values"] = {
                "current_assets": float(current_assets),
                "current_liabilities": float(current_liabilities),
                "cash": float(cash),
                "inventory": float(inventory)
            }

            return {
                "success": True,
                "result": results,
                "agent": "FinancialAgent",
                "operation": "liquidity_analysis",
                "interpretation": f"Liquidity Analysis: Current Ratio is {metrics.get('current_ratio', {}).get('value', 0):.2f}. {metrics.get('current_ratio', {}).get('status', 'Check Data').title()} status."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Liquidity analysis failed: {str(e)}",
                "agent": "FinancialAgent"
            }
    
    def _efficiency_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze efficiency metrics (Asset Turnover, Inventory Turnover)"""
        try:
            financial_cols = self._identify_financial_columns(data)
            results = {}
            
            def get_val(category):
                if category in financial_cols:
                    col = financial_cols[category][0]
                    return data[col].sum()
                return 0.0

            revenue = get_val("revenue")
            assets = get_val("assets")
            cogs = get_val("cost") # Approximation for COGS
            inventory = get_val("inventory") if "inventory" in financial_cols else 0.0
            receivables = get_val("receivables") if "receivables" in financial_cols else 0.0

            metrics = {}
            
            if assets > 0:
                metrics["asset_turnover"] = {
                    "value": float(revenue / assets),
                    "benchmark": 1.0,
                     "description": "Revenue generated per dollar of assets"
                }
            
            if inventory > 0:
                 metrics["inventory_turnover"] = {
                    "value": float(cogs / inventory),
                    "benchmark": 5.0,
                    "description": "Times inventory is sold and replaced"
                }

            if receivables > 0:
                metrics["receivables_turnover"] = {
                    "value": float(revenue / receivables),
                    "benchmark": 10.0,
                     "description": "Efficiency in collecting receivables"
                }

            results["efficiency_metrics"] = metrics

            return {
                "success": True,
                "result": results,
                "agent": "FinancialAgent",
                "operation": "efficiency_analysis",
                "interpretation": "Efficiency analysis complete. Check detailed metrics for turnover ratios."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Efficiency analysis failed: {str(e)}",
                "agent": "FinancialAgent"
            }
    
    def _roi_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze ROI, ROE, and related return metrics"""
        try:
            financial_cols = self._identify_financial_columns(data)
            results = {}
            
            def get_val(category):
                if category in financial_cols:
                    col = financial_cols[category][0]
                    return data[col].sum()
                return 0.0

            revenue = get_val("revenue")
            costs = get_val("cost")
            net_income = revenue - costs if "profit" not in financial_cols else get_val("profit")
            
            equity = get_val("equity")
            assets = get_val("assets")
            investment = get_val("investment") if "investment" in financial_cols else (equity + (get_val("liabilities") or 0))

            metrics = {}

            if investment > 0:
                metrics["roi"] = {
                    "value": float((net_income / investment) * 100),
                    "unit": "%",
                    "description": "Return on Investment"
                }
            
            if equity > 0:
                 metrics["roe"] = {
                    "value": float((net_income / equity) * 100),
                    "unit": "%",
                    "description": "Return on Equity"
                }
            
            if assets > 0:
                metrics["roa"] = {
                    "value": float((net_income / assets) * 100),
                    "unit": "%",
                    "description": "Return on Assets"
                }

            results["return_metrics"] = metrics
            results["components"] = {
                "net_income": float(net_income),
                "total_investment": float(investment),
                "total_equity": float(equity)
            }

            return {
                "success": True,
                "result": results,
                "agent": "FinancialAgent",
                "operation": "roi_analysis",
                "interpretation": f"ROI Analysis: ROI is {metrics.get('roi', {}).get('value', 0):.1f}%. ROE is {metrics.get('roe', {}).get('value', 0):.1f}%."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"ROI analysis failed: {str(e)}",
                "agent": "FinancialAgent"
            }
    
    def _cost_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze cost structure and trends"""
        try:
            financial_cols = self._identify_financial_columns(data)
            results = {}
            
            cost_cols = financial_cols.get("cost", [])
            if not cost_cols:
                # Try to find loose cost matches
                cost_cols = [col for col in data.columns if any(x in col.lower() for x in ['cost', 'expense', 'fee', 'charge'])]
            
            if not cost_cols:
                 return {
                    "success": False,
                    "error": "No cost columns identified for analysis",
                    "agent": "FinancialAgent"
                }

            # 1. Total Cost Breakdown
            total_costs = {}
            grand_total_cost = 0
            for col in cost_cols:
                if pd.api.types.is_numeric_dtype(data[col]):
                    val = data[col].sum()
                    total_costs[col] = float(val)
                    grand_total_cost += val
            
            if grand_total_cost == 0:
                 grand_total_cost = 1 # Avoid div by zero
            
            breakdown = {
                k: {"amount": v, "percentage": float((v / grand_total_cost) * 100)} 
                for k, v in total_costs.items()
            }
            results["cost_breakdown"] = breakdown
            results["total_cost"] = float(grand_total_cost)

            # 2. Cost Trends (if date column exists)
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                data[date_col] = pd.to_datetime(data[date_col])
                
                # Monthly cost trends
                monthly_costs = data.groupby(data[date_col].dt.to_period('M'))[cost_cols].sum()
                
                # Identify rising costs
                trends = {}
                for col in cost_cols:
                    if pd.api.types.is_numeric_dtype(data[col]):
                         # Check if last month > avg
                         recent = monthly_costs[col].iloc[-1]
                         avg = monthly_costs[col].mean()
                         trend = "Increasing" if recent > avg * 1.05 else "Decreasing" if recent < avg * 0.95 else "Stable"
                         trends[col] = trend

                results["cost_trends"] = trends
                results["monthly_costs"] = {k.strftime('%Y-%m'): v for k, v in monthly_costs.sum(axis=1).to_dict().items()}

            return {
                "success": True,
                "result": results,
                "agent": "FinancialAgent",
                "operation": "cost_analysis",
                "interpretation": f"Total Costs analyzed: {grand_total_cost:,.2f}. Major cost driver: {max(breakdown, key=lambda k: breakdown[k]['amount']) if breakdown else 'None'}."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Cost analysis failed: {str(e)}",
                "agent": "FinancialAgent"
            }
    
    def _customer_analysis(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Analyze customer metrics (CLV, Segmentation) if customer data available"""
        try:
            results = {}
            
            # Identify Customer ID column
            cust_cols = [col for col in data.columns if 'customer' in col.lower() or 'client' in col.lower() or 'user_id' in col.lower()]
            
            if not cust_cols:
                 return {
                    "success": False,
                    "error": "No customer identifier column found",
                    "agent": "FinancialAgent"
                }
            
            cust_id = cust_cols[0]
            financial_cols = self._identify_financial_columns(data)
            rev_col = financial_cols.get("revenue", [None])[0]
            if not rev_col:
                # Try finding 'amount' or similar
                rev_col = next((col for col in data.columns if any(x in col.lower() for x in ['amount', 'price', 'value'])), None)

            if not rev_col:
                 return {
                    "success": False,
                    "error": "No revenue/amount column found for customer analysis",
                    "agent": "FinancialAgent"
                }

            # 1. Customer Value (CLV proxy - total revenue per customer)
            customer_value = data.groupby(cust_id)[rev_col].sum().sort_values(ascending=False)
            
            results["customer_metrics"] = {
                "total_customers": int(len(customer_value)),
                "avg_revenue_per_customer": float(customer_value.mean()),
                "median_revenue_per_customer": float(customer_value.median()),
                "top_10_percent_revenue_share": float(customer_value.head(int(len(customer_value)*0.1)).sum() / customer_value.sum() * 100)
            }

            # 2. Segmentation (Top, Medium, Low)
            def segment_customer(val):
                if val >= customer_value.quantile(0.8): return "High Value (Top 20%)"
                if val >= customer_value.quantile(0.5): return "Medium Value"
                return "Low Value"
            
            segments = customer_value.apply(segment_customer).value_counts().to_dict()
            results["customer_segments"] = segments
            
            results["top_customers"] = customer_value.head(5).to_dict()

            return {
                "success": True,
                "result": results,
                "agent": "FinancialAgent",
                "operation": "customer_analysis",
                "interpretation": f"Analyzed {len(customer_value)} customers. Top 10% generate {results['customer_metrics']['top_10_percent_revenue_share']:.1f}% of revenue."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Customer analysis failed: {str(e)}",
                "agent": "FinancialAgent"
            }

    def _financial_forecast(self, data: pd.DataFrame, query: str, **kwargs) -> Dict[str, Any]:
        """Simple linear forecast for financial time series"""
        try:
            results = {}
            date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if not date_cols:
                return {"success": False, "error": "No date column found for forecasting", "agent": "FinancialAgent"}
            
            date_col = date_cols[0]
            data[date_col] = pd.to_datetime(data[date_col])
            
            # Aggregate by month
            df_monthly = data.groupby(data[date_col].dt.to_period('M')).sum(numeric_dtypes=True)
            df_monthly.index = df_monthly.index.to_timestamp()
            
            # Use X as ordinal time (months from start)
            df_monthly['ordinal'] = range(len(df_monthly))
            
            from scipy import stats
            
            forecasts = {}
            
            # Forecast top 3 metric columns (usually Revenue, Cost, Profit) or all if few
            target_cols = [c for c in numeric_cols if c in df_monthly.columns and c != 'ordinal'][:3]
            
            for col in target_cols:
                if len(df_monthly) < 3: # Need points for line
                    continue

                slope, intercept, r_value, p_value, std_err = stats.linregress(df_monthly['ordinal'], df_monthly[col])
                
                # Forecast next 3 months
                next_indices = [len(df_monthly), len(df_monthly)+1, len(df_monthly)+2]
                next_months = [
                    (df_monthly.index[-1] + pd.DateOffset(months=1)).strftime('%Y-%m'),
                    (df_monthly.index[-1] + pd.DateOffset(months=2)).strftime('%Y-%m'),
                    (df_monthly.index[-1] + pd.DateOffset(months=3)).strftime('%Y-%m')
                ]
                
                predictions = [slope * x + intercept for x in next_indices]
                
                forecasts[col] = {
                    "trend_direction": "Positive" if slope > 0 else "Negative",
                    "r_squared": float(r_value**2),
                    "forecast_next_3_periods": dict(zip(next_months, predictions))
                }
            
            results["forecasts"] = forecasts
            
            return {
                "success": True,
                "result": results,
                "agent": "FinancialAgent",
                "operation": "financial_forecast",
                "interpretation": "Generated 3-period linear forecast. Caution: Simple linear projection used."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Financial forecast failed: {str(e)}",
                "agent": "FinancialAgent"
            }
    
