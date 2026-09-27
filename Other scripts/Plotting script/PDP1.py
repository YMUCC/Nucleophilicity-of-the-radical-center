# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Environment & Data Setup
# ==========================================
csv_path = r'/home/dong/RadicalPolarity/Training/Final_Training/Standardized/xyz_rdkitdescriptors.csv'
model_path = r"/home/dong/RadicalPolarity/Training/Final_Training/Standardized/RF/def2QZVP/NBO/rf_model.pkl"

data = pd.read_csv(csv_path)

# Exact 16 descriptors from your training script
names = ['lumo','Charge', 'AN', 'EN', 'Density', 'DP', 'homo', 'BV','ANN','ENN','disp']

# Replicate the StandardScaler used during RF training
x_raw = data[names].values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_raw)

# Load the trained model
model = joblib.load(model_path)

# ==========================================
# 2. Define Targets & Labels
# ==========================================
target_features = ['EN', 'ENN', 'BV']

labels_dict = {
     'EN': 'EN',
    'ENN': 'ENN',
    'BV': 'BV',  
}

# SCI Publication Plotting Style Setup - Force white background style
plt.style.use('default') # Ensure initial style is default white
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 14,             
    'axes.linewidth': 1.5,
    'axes.facecolor': 'white',  # Force axes background to white
    'figure.facecolor': 'white', # Force figure background to white
    'pdf.fonttype': 42           
})


# ==========================================
# 3. Calculate PDP & Plot with Physical Conversion
# ==========================================
print("Starting Final PDP generation with White Background...")

for col in target_features:
    col_idx = names.index(col)
    
    # Calculate PDP
    pdp = partial_dependence(model, x_scaled, features=[col_idx], grid_resolution=100)
    
    x_axis_scaled = pdp["grid_values"][0] 
    y_axis = pdp["average"][0]
    
    # Inverse transform to restore original CSV physical units
    x_axis_phys = (x_axis_scaled * scaler.scale_[col_idx]) + scaler.mean_[col_idx]
    
    # Hartree to eV for HOMO-LUMO
    if col == 'HOMO-LUMO':
        x_axis_phys = x_axis_phys * 27.2114

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor('white') # Double-check single axes background
    
    unique_phys = np.sort(data[col].unique())
    is_discrete = len(unique_phys) <= 8
    
    if is_discrete:
        # For discrete features
        ax.step(x_axis_phys, y_axis, where='post', color='#2c3e50', lw=2.5)
        actual_y = [y_axis[np.abs(x_axis_phys - v).argmin()] for v in unique_phys]
        ax.scatter(unique_phys, actual_y, color='#d35400', s=80, zorder=5, edgecolor='black')
        ax.set_xticks(unique_phys)
    else:
        # Continuous features
        ax.plot(x_axis_phys, y_axis, color='#2c3e50', lw=2.5, alpha=0.9)

    # Formatting axes
    # Set font size for axis titles (Labels)
    ax.set_xlabel(labels_dict.get(col, col), fontweight='bold', fontsize=22)
    ax.set_ylabel(r'Partial Dependence', fontweight='bold', fontsize=22)
    
    # If you also need to adjust the size of tick numbers (Ticks), you can add this line:
    ax.tick_params(axis='both', which='major', labelsize=25)
    
    # Aesthetics - Remove grid and keep borders clear
    sns.despine() # Remove top and right spines
    ax.grid(False) # ?? Core modification: Turn off all grid lines to achieve a pure white background
    
    plt.tight_layout()
    plt.savefig(f'PDP_White_{col}.png', dpi=400, facecolor='white')
    plt.savefig(f'PDP_White_{col}.pdf', facecolor='white')
    plt.close()
    
    print(f"Generated: PDP_White_{col}.png (Pure White Style)")

print("\nAll plots with pure white background are ready!")