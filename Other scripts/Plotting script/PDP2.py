# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler
import itertools

# ==========================================
# 1. Environment & Data Setup
# ==========================================
csv_path = r'/home/dong/RadicalPolarity/Training/Final_Training/Standardized/xyz_rdkitdescriptors.csv'
model_path = r"/home/dong/RadicalPolarity/Training/Final_Training/Standardized/RF/def2QZVP/NBO/rf_model.pkl"

data = pd.read_csv(csv_path)

# Ensure the exact 16 descriptors used in training
names = ['lumo','Charge', 'AN', 'EN', 'Density', 'DP', 'homo', 'BV','ANN','ENN','disp']

x_raw = data[names].values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_raw)

# Load the trained Random Forest model
model = joblib.load(model_path)

# ==========================================
# 2. 2D PDP Configuration
# ==========================================
# The 4 descriptors you want to analyze interactively
target_features = ['EN', 'ENN', 'BV']

# Generate all unique pairs (4 choose 2 = 6 pairs)
pairs = list(itertools.combinations(target_features, 2))

labels_dict = {
    'EN': 'EN',
    'ENN': 'ENN',
    'BV': 'BV',
}

# SCI Publication Plotting Style Setup
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 14,             
    'axes.linewidth': 1.5,
    'pdf.fonttype': 42,          
# === Add the following parameters to control font size ===
    'axes.labelsize': 22,        # Font size for x and y axis titles (e.g., 'disp', 'lumo')
    'xtick.labelsize': 25,       # Font size for X-axis tick numbers
    'ytick.labelsize': 25,       # Font size for Y-axis tick numbers
    'legend.fontsize': 18        # (Optional) Font size for legend
})

# ==========================================
# 3. Calculate and Plot 2D PDPs
# ==========================================
print("Starting 2D Partial Dependence Plots generation...")

for f1, f2 in pairs:
    idx1 = names.index(f1)
    idx2 = names.index(f2)
    
    # Calculate 2D PDP (Pass a tuple of the two feature indices)
    pdp = partial_dependence(model, x_scaled, features=[(idx1, idx2)], grid_resolution=50)
    
    # Extract values
    grid1_scaled = pdp["grid_values"][0] 
    grid2_scaled = pdp["grid_values"][1]
    z_values = pdp["average"][0]  # Shape will be (len(grid1), len(grid2))
    
    # Inverse transform to physical units
    grid1_phys = (grid1_scaled * scaler.scale_[idx1]) + scaler.mean_[idx1]
    grid2_phys = (grid2_scaled * scaler.scale_[idx2]) + scaler.mean_[idx2]
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    # Create meshgrid for contour plot
    X, Y = np.meshgrid(grid1_phys, grid2_phys)
    Z = z_values.T  # Transpose is required for contourf to match X and Y dimensions
    
    # Generate filled contour plot (Heatmap)
    # Using 'viridis' colormap: Yellow = High Delta G (Bad), Dark Purple = Low Delta G (Good)
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.95)
    
    # Add a colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(r'Partial Dependence', fontweight='bold')
    
    # Formatting axes
    ax.set_xlabel(labels_dict.get(f1, f1), fontweight='bold')
    ax.set_ylabel(labels_dict.get(f2, f2), fontweight='bold')
    
    plt.tight_layout()
    
    filename = f'PDP_2D_{f1}_vs_{f2}'
    plt.savefig(f'{filename}.png', dpi=400)
    plt.savefig(f'{filename}.pdf')
    plt.close()
    
    print(f"Successfully generated: {filename}.png / .pdf")

print("\nAll 6 2D plots generated perfectly! Check your folder.")