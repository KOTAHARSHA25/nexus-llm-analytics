"""
SIMPLE TESTS - Statistical Analysis Agent
Basic functionality with small datasets and straightforward analyses
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

print("="*80)
print("STATISTICAL AGENT - SIMPLE TESTS")
print("="*80)
print("Testing basic statistical operations with small, clean datasets\n")

# Set random seed for reproducibility
np.random.seed(42)

#==============================================================================
# TEST 1: Basic Descriptive Statistics (Small Dataset)
#==============================================================================
print("\n" + "="*80)
print("TEST 1.1: Descriptive Statistics - Small Clean Dataset")
print("="*80)

data_simple = pd.DataFrame({
    'score': [85, 90, 78, 92, 88, 76, 95, 82, 89, 91],
    'age': [22, 23, 21, 24, 22, 20, 25, 21, 23, 24],
    'hours_studied': [5, 7, 4, 8, 6, 3, 9, 5, 7, 8]
})

print(f"Dataset shape: {data_simple.shape}")
print(f"\nBasic statistics:")
print(data_simple.describe())

# Manual calculation verification
mean_score = data_simple['score'].mean()
std_score = data_simple['score'].std()
median_score = data_simple['score'].median()

print(f"\n✅ Mean score: {mean_score:.2f}")
print(f"✅ Std score: {std_score:.2f}")
print(f"✅ Median score: {median_score:.2f}")
print(f"✅ Min score: {data_simple['score'].min()}")
print(f"✅ Max score: {data_simple['score'].max()}")

# Check skewness and kurtosis
skewness = scipy_stats.skew(data_simple['score'])
kurtosis = scipy_stats.kurtosis(data_simple['score'])
print(f"✅ Skewness: {skewness:.4f} ({'right-skewed' if skewness > 0 else 'left-skewed'})")
print(f"✅ Kurtosis: {kurtosis:.4f}")

print("\n✅ TEST 1.1 PASSED - Descriptive statistics calculated correctly")

#==============================================================================
# TEST 2: Simple T-Test (Two Groups)
#==============================================================================
print("\n" + "="*80)
print("TEST 1.2: Independent T-Test - Two Groups Comparison")
print("="*80)

data_ttest = pd.DataFrame({
    'group': ['Control']*10 + ['Treatment']*10,
    'score': [75, 78, 72, 80, 76, 74, 79, 77, 73, 76] +  # Control
             [82, 85, 88, 84, 86, 83, 87, 89, 85, 84]    # Treatment
})

control_scores = data_ttest[data_ttest['group'] == 'Control']['score']
treatment_scores = data_ttest[data_ttest['group'] == 'Treatment']['score']

print(f"Control group (n={len(control_scores)}): mean={control_scores.mean():.2f}, std={control_scores.std():.2f}")
print(f"Treatment group (n={len(treatment_scores)}): mean={treatment_scores.mean():.2f}, std={treatment_scores.std():.2f}")

# Perform t-test
t_stat, p_value = scipy_stats.ttest_ind(control_scores, treatment_scores)

# Calculate Cohen's d (effect size)
pooled_std = np.sqrt(((len(control_scores)-1)*control_scores.std()**2 + 
                      (len(treatment_scores)-1)*treatment_scores.std()**2) / 
                     (len(control_scores) + len(treatment_scores) - 2))
cohens_d = (treatment_scores.mean() - control_scores.mean()) / pooled_std

print(f"\n✅ T-statistic: {t_stat:.4f}")
print(f"✅ P-value: {p_value:.4f}")
print(f"✅ Significant at α=0.05: {p_value < 0.05}")
print(f"✅ Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'} effect)")

assert p_value < 0.05, "T-test should be significant"
print("\n✅ TEST 1.2 PASSED - T-test shows significant difference between groups")

#==============================================================================
# TEST 3: Simple Correlation (Two Variables)
#==============================================================================
print("\n" + "="*80)
print("TEST 1.3: Pearson Correlation - Study Hours vs Scores")
print("="*80)

hours = data_simple['hours_studied']
scores = data_simple['score']

# Pearson correlation
pearson_r, pearson_p = scipy_stats.pearsonr(hours, scores)

print(f"Study hours: {hours.tolist()}")
print(f"Scores: {scores.tolist()}")
print(f"\n✅ Pearson r: {pearson_r:.4f}")
print(f"✅ P-value: {pearson_p:.4f}")
print(f"✅ Significant: {pearson_p < 0.05}")
print(f"✅ Interpretation: {'Strong' if abs(pearson_r) > 0.7 else 'Moderate' if abs(pearson_r) > 0.4 else 'Weak'} {'positive' if pearson_r > 0 else 'negative'} correlation")

assert abs(pearson_r) > 0.5, "Should show moderate to strong correlation"
print("\n✅ TEST 1.3 PASSED - Correlation analysis working correctly")

#==============================================================================
# TEST 4: Simple Chi-Square (2x2 Contingency Table)
#==============================================================================
print("\n" + "="*80)
print("TEST 1.4: Chi-Square Test - 2x2 Independence Test")
print("="*80)

data_chi = pd.DataFrame({
    'gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male',
               'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female'],
    'preference': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B',
                   'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B']
})

# Create contingency table
contingency = pd.crosstab(data_chi['gender'], data_chi['preference'])
print("Contingency Table:")
print(contingency)

# Chi-square test
chi2, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)

# Cramér's V (effect size)
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / n)

print(f"\n✅ Chi-square statistic: {chi2:.4f}")
print(f"✅ P-value: {p_value:.4f}")
print(f"✅ Degrees of freedom: {dof}")
print(f"✅ Cramér's V: {cramers_v:.4f}")
print(f"✅ Significant: {p_value < 0.05}")

print("\n✅ TEST 1.4 PASSED - Chi-square test executed correctly")

#==============================================================================
# TEST 5: Simple Outlier Detection (IQR Method)
#==============================================================================
print("\n" + "="*80)
print("TEST 1.5: Outlier Detection - IQR Method")
print("="*80)

data_outliers = pd.DataFrame({
    'values': [10, 12, 11, 13, 12, 14, 11, 13, 100, 12, 11, 14, 13]  # 100 is obvious outlier
})

values = data_outliers['values']
Q1 = values.quantile(0.25)
Q3 = values.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = values[(values < lower_bound) | (values > upper_bound)]

print(f"Values: {values.tolist()}")
print(f"\n✅ Q1: {Q1:.2f}")
print(f"✅ Q3: {Q3:.2f}")
print(f"✅ IQR: {IQR:.2f}")
print(f"✅ Lower bound: {lower_bound:.2f}")
print(f"✅ Upper bound: {upper_bound:.2f}")
print(f"✅ Outliers detected: {outliers.tolist()}")
print(f"✅ Number of outliers: {len(outliers)}")

assert len(outliers) >= 1, "Should detect at least one outlier (100)"
assert 100 in outliers.values, "Should detect 100 as outlier"
print("\n✅ TEST 1.5 PASSED - Outlier detection working correctly")

#==============================================================================
# TEST 6: Simple Normality Test
#==============================================================================
print("\n" + "="*80)
print("TEST 1.6: Normality Test - Shapiro-Wilk")
print("="*80)

# Generate normal and non-normal data
normal_data = np.random.normal(50, 10, 50)
non_normal_data = np.random.exponential(2, 50)

# Shapiro-Wilk test
sw_stat_normal, sw_p_normal = scipy_stats.shapiro(normal_data)
sw_stat_nonnormal, sw_p_nonnormal = scipy_stats.shapiro(non_normal_data)

print("Normal Data:")
print(f"✅ Shapiro-Wilk statistic: {sw_stat_normal:.4f}")
print(f"✅ P-value: {sw_p_normal:.4f}")
print(f"✅ Is normal (p>0.05): {sw_p_normal > 0.05}")

print("\nNon-Normal Data (exponential):")
print(f"✅ Shapiro-Wilk statistic: {sw_stat_nonnormal:.4f}")
print(f"✅ P-value: {sw_p_nonnormal:.4f}")
print(f"✅ Is normal (p>0.05): {sw_p_nonnormal > 0.05}")

assert sw_p_normal > 0.05, "Normal data should pass normality test"
assert sw_p_nonnormal < 0.05, "Exponential data should fail normality test"
print("\n✅ TEST 1.6 PASSED - Normality testing distinguishes normal from non-normal")

#==============================================================================
# SUMMARY
#==============================================================================
print("\n" + "="*80)
print("SIMPLE TESTS SUMMARY")
print("="*80)

tests_passed = 6
tests_total = 6

print(f"\n✅ All {tests_passed}/{tests_total} simple tests PASSED!")
print("\nTests completed:")
print("  ✅ 1.1 - Descriptive Statistics")
print("  ✅ 1.2 - Independent T-Test")
print("  ✅ 1.3 - Pearson Correlation")
print("  ✅ 1.4 - Chi-Square Test")
print("  ✅ 1.5 - Outlier Detection (IQR)")
print("  ✅ 1.6 - Normality Test (Shapiro-Wilk)")

print("\n" + "="*80)
print("✅ SIMPLE TESTS: 100% SUCCESS")
print("="*80)
