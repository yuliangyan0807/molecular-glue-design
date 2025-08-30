from Bio.PDB import PDBParser
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import torch


def get_pocket_and_mask(p1_coords: torch.Tensor, p2_coords: torch.Tensor, cutoff=8.):
    """
    get pocket coords and mask of p1 and p2.
    The pocket coords is the coords of p1.
    Return:
        pocket_coords: (N, 3), float
        mask_p1: (N, 1), bool
        mask_p2: (M, 1), bool
    """
    # get distance matrix
    dist_mat = torch.cdist(p1_coords, p2_coords)
    # get mask
    mask_p1 = dist_mat.min(dim=1).values < cutoff
    mask_p2 = dist_mat.min(dim=0).values < cutoff
    # get pocket coords
    pocket_coords = p1_coords[mask_p1]
    return pocket_coords, mask_p1, mask_p2


def get_interface_atoms_from_pdb(pdb1, pdb2, cutoff=5.0):
    """
    Get interface atoms using the DeepTernary-style pocket detection.
    """
    parser = PDBParser(QUIET=True)
    struct1 = parser.get_structure("prot1", pdb1)
    struct2 = parser.get_structure("prot2", pdb2)

    # Extract coordinates and residue information
    coords1 = []
    residue_ids1 = []
    for atom in struct1.get_atoms():
        coords1.append(atom.coord)
        residue_ids1.append(atom.get_parent().get_id()[1])  # Get residue number
    
    coords2 = []
    residue_ids2 = []
    for atom in struct2.get_atoms():
        coords2.append(atom.coord)
        residue_ids2.append(atom.get_parent().get_id()[1])  # Get residue number
    
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    residue_ids1 = np.array(residue_ids1)
    residue_ids2 = np.array(residue_ids2)
    
    if len(coords1) == 0 or len(coords2) == 0:
        return np.array([]), np.array([], dtype=bool), np.array([], dtype=bool), set(), set()

    # Convert to torch tensors
    p1_coords = torch.tensor(coords1, dtype=torch.float32)
    p2_coords = torch.tensor(coords2, dtype=torch.float32)
    
    # Use the DeepTernary pocket detection function
    pocket_coords, mask_p1, mask_p2 = get_pocket_and_mask(p1_coords, p2_coords, cutoff=cutoff)
    
    # Get unique residue IDs for pocket residues
    pocket_residues_p1 = set(residue_ids1[mask_p1.numpy()])
    pocket_residues_p2 = set(residue_ids2[mask_p2.numpy()])
    
    # Convert back to numpy
    pocket_coords_np = pocket_coords.numpy()
    mask_p1_np = mask_p1.numpy()
    mask_p2_np = mask_p2.numpy()
    
    return pocket_coords_np, mask_p1_np, mask_p2_np, pocket_residues_p1, pocket_residues_p2


def get_pocket_centroid(coords):
    """
    Compute the centroid (average coordinate) of the interface pocket.
    """
    if len(coords) == 0:
        return None
    return np.mean(coords, axis=0)


def process_complex(complex_dir):
    """
    Process a single complex directory using DeepTernary-style pocket detection.
    """
    complex_path = Path(complex_dir)
    pdb1_path = complex_path / "protein1.pdb"
    pdb2_path = complex_path / "protein2.pdb"
    
    # Check if both protein files exist
    if not pdb1_path.exists() or not pdb2_path.exists():
        return None
    
    try:
        # Get interface atoms using DeepTernary method
        pocket_coords, mask_p1, mask_p2, pocket_residues_p1, pocket_residues_p2 = get_interface_atoms_from_pdb(
            str(pdb1_path), str(pdb2_path), cutoff=5.0
        )
        
        # Compute centroid
        centroid = get_pocket_centroid(pocket_coords)
        
        return {
            'complex_name': complex_path.name,
            'interface_atoms': len(pocket_coords),
            'centroid': centroid,
            'pocket_coords': pocket_coords,
            'mask_p1': mask_p1,
            'mask_p2': mask_p2,
            'pocket_residues_p1': pocket_residues_p1,
            'pocket_residues_p2': pocket_residues_p2,
            'num_pocket_residues_p1': len(pocket_residues_p1),
            'num_pocket_residues_p2': len(pocket_residues_p2)
        }
    except Exception as e:
        return None


def traverse_mgd_train(mgd_train_dir, max_complexes=None):
    """
    Traverse all complexes in MGD_Train directory and process them.
    """
    mgd_train_path = Path(mgd_train_dir)
    
    if not mgd_train_path.exists():
        print(f"Error: MGD_Train directory not found at {mgd_train_dir}")
        return
    
    # Get all complex directories
    complex_dirs = [d for d in mgd_train_path.iterdir() if d.is_dir()]
    
    if max_complexes:
        complex_dirs = complex_dirs[:max_complexes]
    
    print(f"Found {len(complex_dirs)} complexes to process")
    
    results = []
    successful = 0
    failed = 0
    
    # Process each complex
    for complex_dir in tqdm(complex_dirs, desc="Processing complexes"):
        result = process_complex(complex_dir)
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful} complexes")
    print(f"Failed to process: {failed} complexes")
    
    return results


def analyze_results(results):
    """
    Analyze the results from processing all complexes.
    """
    if not results:
        print("No results to analyze")
        return
    
    # Basic statistics
    interface_atom_counts = [r['interface_atoms'] for r in results]
    valid_centroids = [r['centroid'] for r in results if r['centroid'] is not None]
    residue_counts_p1 = [r['num_pocket_residues_p1'] for r in results]
    residue_counts_p2 = [r['num_pocket_residues_p2'] for r in results]
    
    # Check for complexes with no interface
    no_interface_complexes = [r for r in results if r['interface_atoms'] == 0]
    
    print(f"\nAnalysis Results:")
    print(f"Total complexes processed: {len(results)}")
    print(f"Complexes with valid centroids: {len(valid_centroids)}")
    print(f"Complexes with NO interface found: {len(no_interface_complexes)}")
    
    if interface_atom_counts:
        print(f"Interface atoms - Min: {min(interface_atom_counts)}, Max: {max(interface_atom_counts)}, Mean: {np.mean(interface_atom_counts):.2f}")
    
    if residue_counts_p1:
        print(f"P1 Pocket residues - Min: {min(residue_counts_p1)}, Max: {max(residue_counts_p1)}, Mean: {np.mean(residue_counts_p1):.2f}")
    
    if residue_counts_p2:
        print(f"P2 Pocket residues - Min: {min(residue_counts_p2)}, Max: {max(residue_counts_p2)}, Mean: {np.mean(residue_counts_p2):.2f}")
    
    # Print complexes with no interface
    if no_interface_complexes:
        print(f"\n⚠️  Complexes with NO interface found ({len(no_interface_complexes)}):")
        for i, complex_info in enumerate(no_interface_complexes, 1):
            print(f"  {i:2d}. {complex_info['complex_name']}")
    
    # Show detailed results for each complex
    print(f"\nDetailed results for each complex:")
    print(f"{'Complex Name':<20} {'Interface Atoms':<15} {'P1 Residues':<12} {'P2 Residues':<12} {'Centroid':<30}")
    print("-" * 100)
    
    for result in results:
        complex_name = result['complex_name'][:19]  # Truncate if too long
        interface_atoms = result['interface_atoms']
        p1_residues = result['num_pocket_residues_p1']
        p2_residues = result['num_pocket_residues_p2']
        
        if result['centroid'] is not None:
            centroid_str = f"({result['centroid'][0]:.1f}, {result['centroid'][1]:.1f}, {result['centroid'][2]:.1f})"
        else:
            centroid_str = "None"
        
        # Highlight complexes with no interface
        if interface_atoms == 0:
            print(f"{complex_name:<20} {interface_atoms:<15} {p1_residues:<12} {p2_residues:<12} {centroid_str:<30} ⚠️ NO INTERFACE")
        else:
            print(f"{complex_name:<20} {interface_atoms:<15} {p1_residues:<12} {p2_residues:<12} {centroid_str:<30}")


def save_results(results, output_file="pocket_analysis_results.txt"):
    """
    Save results to a text file with all interface atom coordinates.
    """
    with open(output_file, 'w') as f:
        f.write("Complex Name\tInterface Atoms\tP1 Pocket Residues\tP2 Pocket Residues\tCentroid X\tCentroid Y\tCentroid Z\tInterface Coordinates\tHas Interface\n")
        for result in results:
            centroid_str = "\t".join([f"{coord:.3f}" for coord in result['centroid']]) if result['centroid'] is not None else "None\tNone\tNone"
            
            # Save all interface coordinates as a string
            if len(result['pocket_coords']) > 0:
                coords_str = ";".join([f"{coord[0]:.3f},{coord[1]:.3f},{coord[2]:.3f}" for coord in result['pocket_coords']])
            else:
                coords_str = "None"
            
            has_interface = "YES" if result['interface_atoms'] > 0 else "NO"
            
            f.write(f"{result['complex_name']}\t{result['interface_atoms']}\t{result['num_pocket_residues_p1']}\t{result['num_pocket_residues_p2']}\t{centroid_str}\t{coords_str}\t{has_interface}\n")
    
    print(f"Results saved to {output_file}")


def save_detailed_results(results, output_dir="pocket_results"):
    """
    Save detailed results including all interface coordinates to separate files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Complex Name\tInterface Atoms\tP1 Pocket Residues\tP2 Pocket Residues\tCentroid X\tCentroid Y\tCentroid Z\tHas Interface\n")
        for result in results:
            centroid_str = "\t".join([f"{coord:.3f}" for coord in result['centroid']]) if result['centroid'] is not None else "None\tNone\tNone"
            has_interface = "YES" if result['interface_atoms'] > 0 else "NO"
            f.write(f"{result['complex_name']}\t{result['interface_atoms']}\t{result['num_pocket_residues_p1']}\t{result['num_pocket_residues_p2']}\t{centroid_str}\t{has_interface}\n")
    
    # Save list of complexes with no interface
    no_interface_file = os.path.join(output_dir, "no_interface_complexes.txt")
    no_interface_complexes = [r for r in results if r['interface_atoms'] == 0]
    with open(no_interface_file, 'w') as f:
        f.write(f"# Complexes with NO interface found: {len(no_interface_complexes)}\n")
        f.write("# Complex Name\n")
        for complex_info in no_interface_complexes:
            f.write(f"{complex_info['complex_name']}\n")
    
    # Save detailed coordinates for each complex
    for result in results:
        complex_name = result['complex_name']
        coords_file = os.path.join(output_dir, f"{complex_name}_interface_coords.txt")
        
        with open(coords_file, 'w') as f:
            f.write(f"# Interface coordinates for {complex_name}\n")
            f.write(f"# Total interface atoms: {result['interface_atoms']}\n")
            f.write(f"# P1 pocket residues: {result['num_pocket_residues_p1']}\n")
            f.write(f"# P2 pocket residues: {result['num_pocket_residues_p2']}\n")
            if result['centroid'] is not None:
                f.write(f"# Centroid: {result['centroid'][0]:.3f}, {result['centroid'][1]:.3f}, {result['centroid'][2]:.3f}\n")
            f.write(f"# P1 pocket residue IDs: {sorted(result['pocket_residues_p1'])}\n")
            f.write(f"# P2 pocket residue IDs: {sorted(result['pocket_residues_p2'])}\n")
            f.write(f"# Has interface: {'YES' if result['interface_atoms'] > 0 else 'NO'}\n")
            f.write("# X\tY\tZ\n")
            
            for coord in result['pocket_coords']:
                f.write(f"{coord[0]:.3f}\t{coord[1]:.3f}\t{coord[2]:.3f}\n")
    
    print(f"Detailed results saved to {output_dir}/")
    print(f"Summary file: {summary_file}")
    print(f"No interface complexes list: {no_interface_file}")
    print(f"Individual coordinate files: {len(results)} files created")


if __name__ == "__main__":
    # Define the MGD_Train directory path
    mgd_train_dir = "./data/TernaryDB/MGD_Train"
    
    # Process first 50 complexes for testing
    results = traverse_mgd_train(mgd_train_dir, max_complexes=None)
    
    if results:
        # Analyze results
        analyze_results(results)
        
        # Save results with all interface coordinates
        save_results(results)
        save_detailed_results(results)
    else:
        print("No complexes were successfully processed.")
