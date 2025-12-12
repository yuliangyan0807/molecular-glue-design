import os
import requests
import time
from tqdm import tqdm

# ================= Configuration Section =================
# Modify this to your root directory path from the screenshot
ROOT_DIR = "data/TernaryDB/MGD_Train" 

# To avoid being blocked by RCSB, set a small request interval (seconds)
SLEEP_TIME = 0.5 
# ===========================================

def download_ligand_sdf(pdb_id, ligand_id, save_path):
    """
    Try to download the SDF file of the Ligand.
    Prioritize downloading bound state (Bound), if failed then download ideal state (Ideal).
    """
    
    # Strategy 1: Download bound state SDF (Bound State)
    # This SDF's coordinates are aligned with PDB, most suitable for Deep Learning Ground Truth
    url_bound = f"https://models.rcsb.org/v1/{pdb_id}/ligand?label_comp_id={ligand_id}&encoding=sdf"
    
    # Strategy 2: Download ideal state SDF (Ideal State)
    # Only contains chemical topology information, coordinates are usually energy-minimized, does not contain pocket position information
    url_ideal = f"https://files.rcsb.org/ligand/{ligand_id}.sdf"

    try:
        # Try 1: Bound State
        response = requests.get(url_bound, timeout=10)
        if response.status_code == 200 and len(response.text) > 0:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            return "BOUND (Aligned with PDB)"
        
        # Try 2: Ideal State (if Bound fails)
        print(f"   -> Bound SDF not found for {ligand_id} in {pdb_id}, trying Ideal SDF...")
        response = requests.get(url_ideal, timeout=10)
        if response.status_code == 200 and len(response.text) > 0:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            return "IDEAL (Topology only)"
            
    except Exception as e:
        return f"ERROR: {str(e)}"

    return "FAILED (404 Not Found)"

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory '{ROOT_DIR}' not found.")
        return

    count_success = 0
    count_total = 0

    print(f"Starting batch download in: {ROOT_DIR}...\n")

    for folder_name in tqdm(os.listdir(ROOT_DIR)):
        folder_path = os.path.join(ROOT_DIR, folder_name)

        # Ensure it's a folder and follows naming convention (contains underscore)
        if os.path.isdir(folder_path) and "_" in folder_name:
            
            # Parse folder name: e.g. 1A2Y_A_C_PO4
            parts = folder_name.split("_")
            
            # Simple defensive programming to ensure sufficient parts after splitting
            if len(parts) >= 4:
                pdb_id = parts[0]      # 1A2Y
                ligand_id = parts[-1]  # PO4
                
                # Define save path, to avoid overwriting your original ligand.pdb, we save as ligand_rcsb.sdf
                save_path = os.path.join(folder_path, "ligand_rcsb.sdf")
                
                # Check if already exists to avoid duplicate downloads
                if os.path.exists(save_path):
                    print(f"[SKIP] {folder_name}: ligand_rcsb.sdf already exists.")
                    continue

                print(f"[PROC] Processing {folder_name} (PDB: {pdb_id}, Lig: {ligand_id})...")
                
                status = download_ligand_sdf(pdb_id, ligand_id, save_path)
                
                print(f"       Result: {status}")
                
                if "BOUND" in status or "IDEAL" in status:
                    count_success += 1
                
                count_total += 1
                time.sleep(SLEEP_TIME)
            else:
                print(f"[WARN] Skipping {folder_name}: Naming format unexpected.")

    print(f"\n--- Download Complete ---")
    print(f"Total processed: {count_total}")
    print(f"Successfully downloaded: {count_success}")

if __name__ == "__main__":
    main()