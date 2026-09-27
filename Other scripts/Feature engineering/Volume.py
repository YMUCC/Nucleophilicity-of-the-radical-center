# -*- coding: utf-8 -*-
import os
import csv
from morfeus import read_xyz, BuriedVolume

# =====================================================================
# Fully Automatic Buried Volume (%Vbur) Extraction Script (V2)
# Features: 1. Automatically identifies transition metals.
#           2. If no transition metal is found, automatically sets the 
#              [1st atom] as the radical center.
# =====================================================================

# 1. List of transition metals
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
}

root_dir = os.getcwd()
xyz_files = [f for f in os.listdir(root_dir) if f.endswith('.xyz') and os.path.isfile(os.path.join(root_dir, f))]

print(f"?? Found {len(xyz_files)} .xyz files. Starting fully automatic calculation...\n")

extracted_data = []

for file in xyz_files:
    try:
        clean_filename = file.replace('.xyz', '')
        elements, coordinates = read_xyz(file)
        
        if len(elements) == 0:
            print(f"  -> [Warning] {file} is empty, skipping.")
            continue

        center_index = None
        determination_method = ""
        
        # Strategy A: Automatically find transition metals
        for i, el in enumerate(elements):
            if el in TRANSITION_METALS:
                center_index = i + 1  # morfeus indexing starts from 1
                determination_method = "Auto (Metal)"
                break
                
        # Strategy B: If no transition metal is found, default to the 1st atom
        if center_index is None:
            center_index = 1
            determination_method = "Default (1st atom)"

        # Get center atom symbol
        center_symbol = elements[center_index - 1]
        center_label = f"{center_symbol}{center_index}"

        # Calculate buried volume
        bv = BuriedVolume(elements, coordinates, center_index)
        percent_vbur = round(bv.fraction_buried_volume * 100, 3)
        
        print(f"  -> [Success] {file:<20} | Center: {center_label:<4} | Rule: {determination_method:<18} | %Vbur: {percent_vbur}%")
        
        extracted_data.append({
            "Filename": clean_filename,
            "Center_Atom": center_label,
            "Method": determination_method,
            "Percent_Vbur": percent_vbur
        })
        
    except Exception as e:
        print(f"  -> [Error] {file} processing failed: {str(e)}")

# ==========================================
# Export Data
# ==========================================
if extracted_data:
    csv_filename = "Auto_Vbur_Descriptors.csv"
    output_path = os.path.join(root_dir, csv_filename)
    
    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["Filename", "Center_Atom", "Method", "Percent_Vbur"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(extracted_data)
        
    print(f"\n? Extraction complete! Data has been saved to: {csv_filename}")
else:
    print("\n?? Failed to extract any data.")