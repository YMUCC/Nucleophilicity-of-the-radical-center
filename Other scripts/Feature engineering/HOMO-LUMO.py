# -*- coding: utf-8 -*-
import os
from morfeus import XTB, read_xyz

# 1. Define the folder path containing the xyz files
folder_path = r'/home/dong/RadicalPolarity/PCM-Def2TZVP/Solvent-M06-2X/Metal-Model-test/EE'

# 2. Prepare a list to store the results (optional, for easy export later)
results = []

print(f"{'File Name':<20} | {'HOMO (eV)':<10} | {'LUMO (eV)':<10}")
print("-" * 45)

# 3. Iterate through the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xyz'):
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # Read and calculate
            elements, coordinates = read_xyz(file_path)
            xtb = XTB(elements, coordinates)
            
            lumo = xtb.get_lumo()
            homo = xtb.get_homo()
            
            # Print the results for the current file
            print(f"{file_name:<20} | {homo:<10.4f} | {lumo:<10.4f}")
            
            # Save the data
            results.append([file_name, homo, lumo])
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")