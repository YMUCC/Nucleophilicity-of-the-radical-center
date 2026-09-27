import os
import glob
from morfeus import Dispersion, read_geometry

# 1. Define folder path
folder_path = r"/home/dong/RadicalPolarity/PCM-Def2TZVP/Solvent-M06-2X/Metal-Model-test/EE"

# 2. Get all .xyz files
xyz_files = glob.glob(os.path.join(folder_path, "*.xyz"))

print(f"Found {len(xyz_files)} .xyz files. Starting calculation...\n")

# 3. Loop through files
for file_path in xyz_files:
    file_name = os.path.basename(file_path)
    print(f"======== Processing: {file_name} ========")
    
    try:
        elements, coordinates = read_geometry(file_path)
        disp = Dispersion(elements, coordinates)
        disp.print_report()
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        
    print("\n")