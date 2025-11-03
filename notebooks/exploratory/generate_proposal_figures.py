"""
CMSE 492 Project Proposal - Figure Generation
Vehicle Fuel Economy Prediction Using Machine Learning

Author: James Restaneo
Email: restaneo@msu.edu
Date: November 2, 2025
GitHub: https://github.com/restaneo/cmse492_project

This script generates all 5 figures exactly as referenced in the proposal PDF.
Dataset: vehicles_2024.csv (2,500 vehicles, 2020-2024)
Expected baseline: R²=0.027, RMSE=23.51 MPG
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set random seed for reproducibility (matches proposal)
np.random.seed(42)

print("="*70)
print("GENERATING FIGURES FOR PROJECT PROPOSAL")
print("Vehicle Fuel Economy Prediction Using Machine Learning")
print("="*70)

# ==============================================================================
# DATA LOADING
# ==============================================================================
print("\nLoading dataset...")
df = pd.read_csv('vehicles_2024.csv')

print(f"Dataset loaded: {len(df)} vehicles")
print(f"Years covered: {df['year'].min()}-{df['year'].max()}")
print(f"Number of features: {len(df.columns)}")

# ==============================================================================
# FIGURE 1: Target Distribution (Combined MPG)
# Proposal Reference: "Figure 1 shows combined MPG distribution 
# (mean=25.3, median=23, std=7.8) with right skew"
# ==============================================================================
print("\n" + "-"*70)
print("Creating Figure 1: Target Distribution")
print("-"*70)

combined_mpg = df['comb08']

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(combined_mpg, bins=50, edgecolor='black', alpha=0.7, color='#4ECDC4')
ax.axvline(np.mean(combined_mpg), color='red', linestyle='--', linewidth=2, 
           label=f'Mean = {np.mean(combined_mpg):.1f} MPG')
ax.axvline(np.median(combined_mpg), color='orange', linestyle='--', linewidth=2, 
           label=f'Median = {np.median(combined_mpg):.1f} MPG')

ax.set_xlabel('Combined MPG', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Combined MPG (EPA 2020-2024 Vehicles)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: 01_target_distribution.png")
print(f"  Mean: {np.mean(combined_mpg):.1f} MPG")
print(f"  Median: {np.median(combined_mpg):.1f} MPG")
print(f"  Std: {np.std(combined_mpg):.1f} MPG")

# ==============================================================================
# FIGURE 2: Engine Characteristics
# Proposal Reference: "Figure 2 reveals displacement and cylinder count 
# relationships with fuel economy"
# ==============================================================================
print("\n" + "-"*70)
print("Creating Figure 2: Engine Characteristics")
print("-"*70)

displacement = df['displ']
cylinders = df['cylinders']
mpg = df['comb08']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Displacement vs MPG
axes[0].scatter(displacement, mpg, alpha=0.4, s=30, edgecolors='black', linewidth=0.3)
axes[0].set_xlabel('Engine Displacement (L)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Combined MPG', fontsize=12, fontweight='bold')
axes[0].set_title('Engine Displacement vs Fuel Economy', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Right: Cylinders vs MPG
cylinder_values = sorted(df['cylinders'].unique())
boxplot_data = [mpg[cylinders == cyl].values for cyl in cylinder_values]

bp = axes[1].boxplot(boxplot_data, labels=cylinder_values, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')

axes[1].set_xlabel('Number of Cylinders', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Combined MPG', fontsize=12, fontweight='bold')
axes[1].set_title('Fuel Economy by Cylinder Count', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('02_engine_characteristics.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: 02_engine_characteristics.png")
print(f"  Displacement range: {displacement.min():.1f}-{displacement.max():.1f} L")
print(f"  Cylinder options: {cylinder_values}")

# ==============================================================================
# FIGURE 3: Make and Fuel Type Distribution  
# Proposal Reference: "Figure 3 displays top manufacturers and fuel type distribution"
# ==============================================================================
print("\n" + "-"*70)
print("Creating Figure 3: Make and Fuel Type Distribution")
print("-"*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Top 10 makes
make_counts = df['make'].value_counts().head(10)

axes[0].barh(range(len(make_counts)), make_counts.values, color='steelblue', edgecolor='black')
axes[0].set_yticks(range(len(make_counts)))
axes[0].set_yticklabels(make_counts.index)
axes[0].set_xlabel('Number of Vehicles', fontsize=12, fontweight='bold')
axes[0].set_title('Top 10 Vehicle Makes in Dataset', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Right: Fuel type distribution
fuel_counts = df['fuelType1'].value_counts()

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9', '#A29BFE']
axes[1].bar(range(len(fuel_counts)), fuel_counts.values, 
            color=colors[:len(fuel_counts)], edgecolor='black')
axes[1].set_xticks(range(len(fuel_counts)))
axes[1].set_xticklabels(fuel_counts.index, rotation=45, ha='right')
axes[1].set_ylabel('Number of Vehicles', fontsize=12, fontweight='bold')
axes[1].set_title('Distribution by Fuel Type', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_make_and_fuel_type.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: 03_make_and_fuel_type.png")
print(f"  Top make: {make_counts.index[0]} ({make_counts.values[0]} vehicles)")
print(f"  Fuel types: {list(fuel_counts.index)}")

# ==============================================================================
# FIGURE 4: Correlation Heatmap
# Proposal Reference: "Figure 4 shows strong correlations: displacement (-0.82), 
# cylinders (-0.76), CO₂ (-0.98) with MPG"
# ==============================================================================
print("\n" + "-"*70)
print("Creating Figure 4: Correlation Heatmap")
print("-"*70)

# Select numerical features
numerical_features = ['comb08', 'city08', 'highway08', 'displ', 
                      'cylinders', 'co2TailpipeGpm', 'fuelCost08']

df_numerical = df[numerical_features].copy()

# Rename for display
display_names = {
    'comb08': 'Combined\nMPG',
    'city08': 'City\nMPG', 
    'highway08': 'Highway\nMPG',
    'displ': 'Displacement\n(L)',
    'cylinders': 'Cylinders',
    'co2TailpipeGpm': 'CO2\nEmissions',
    'fuelCost08': 'Annual\nFuel Cost'
}
df_numerical.columns = [display_names[col] for col in df_numerical.columns]

# Calculate correlation
correlation_matrix = df_numerical.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, square=True, linewidths=1,
            cbar_kws={'label': 'Correlation'})

ax.set_title('Correlation Matrix of Numerical Features', 
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('04_correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: 04_correlation_heatmap.png")
print(f"  Combined MPG vs Displacement: {correlation_matrix.iloc[0, 3]:.2f}")
print(f"  Combined MPG vs Cylinders: {correlation_matrix.iloc[0, 4]:.2f}")
print(f"  Combined MPG vs CO2: {correlation_matrix.iloc[0, 5]:.2f}")

# ==============================================================================
# FIGURE 5: Baseline Model Performance
# Proposal Reference: "Figure 5 presents baseline model performance: 
# R²=0.027, RMSE=23.51 MPG"
# ==============================================================================
print("\n" + "-"*70)
print("Creating Figure 5: Baseline Model Performance")
print("-"*70)

# Prepare features (simple linear regression baseline)
X = df[['year', 'displ', 'cylinders']].copy()
y = df['comb08'].copy()

# Remove NaN
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

print(f"Training baseline model on {len(X)} samples...")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple linear regression
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# Predictions
y_train_pred = baseline_model.predict(X_train)
y_test_pred = baseline_model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nBaseline Model Results:")
print(f"  Train R²:   {train_r2:.4f}")
print(f"  Test R²:    {test_r2:.4f}  (Proposal: 0.027)")
print(f"  Train RMSE: {train_rmse:.2f} MPG")
print(f"  Test RMSE:  {test_rmse:.2f} MPG  (Proposal: 23.51)")
print(f"  Test MAE:   {test_mae:.2f} MPG")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Actual vs Predicted
axes[0].scatter(y_test, y_test_pred, alpha=0.5, s=40, edgecolors='black', 
                linewidth=0.3, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')

textstr = f'R² = {test_r2:.4f}\nRMSE = {test_rmse:.2f} MPG'
axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

axes[0].set_xlabel('Actual Combined MPG', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Combined MPG', fontsize=12, fontweight='bold')
axes[0].set_title('Actual vs Predicted MPG', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Right: Residuals
residuals = y_test - y_test_pred
axes[1].scatter(y_test_pred, residuals, alpha=0.5, s=40, edgecolors='black', 
                linewidth=0.3, color='coral')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')

axes[1].set_xlabel('Predicted Combined MPG', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_baseline_model_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved: 05_baseline_model_performance.png")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("="*70)
print("\nGenerated files (ready for LaTeX proposal):")
print("  1. 01_target_distribution.png")
print("  2. 02_engine_characteristics.png")
print("  3. 03_make_and_fuel_type.png")
print("  4. 04_correlation_heatmap.png")
print("  5. 05_baseline_model_performance.png")
print("\nDataset Information:")
print(f"  File: vehicles_2024.csv")
print(f"  Samples: {len(df)}")
print(f"  Years: {df['year'].min()}-{df['year'].max()}")
print(f"  Target: Combined MPG (range: {combined_mpg.min():.1f}-{combined_mpg.max():.1f})")
print("\nBaseline Model Performance:")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.2f} MPG")
print(f"  Features: year, displacement, cylinders")
print("\nThese figures match the specifications in:")
print("  Restaneo_James_CMSE492_ProjectProposal.pdf")
print("="*70)
