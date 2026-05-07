import os
import sys
import random
import numpy as np
import torch
import rdkit
from tqdm import tqdm
from rdkit import Chem
from copy import deepcopy
from configs.config import DATASET_ARGS, DictToObject
from pocket import get_elilipsoid_for_interface, get_interface_from_graphs
from random import sample
import json
import warnings
warnings.filterwarnings('ignore')

from datasets import Dataset, load_from_disk
from utils.rigid_utils import parse_pdb, get_torsion_angle, parse_pdb_ligand
from utils.constants import BBHeavyAtom
from utils.training_utils import collate_fn

from DeepTernary.deepternary.models.geometry_utils import random_rotation_translation

def read_ligand_pdb(pdb_path, sanitize=True, remove_hs=True, return_symbols=False):
    """
    Read a ligand PDB file and return atomic numbers and coordinates.

    Args:
        pdb_path (str): Path to the ligand PDB file.
        sanitize (bool): Whether to sanitize the RDKit molecule.
        remove_hs (bool): Whether to remove hydrogens after loading.
        return_symbols (bool): If True, also return list of atomic symbols.

    Returns:
        If return_symbols is False:
            (Z, coords)
        If return_symbols is True:
            (Z, coords, symbols)

        Where:
            - Z: np.ndarray of shape [N], dtype=int32, atomic numbers
            - coords: np.ndarray of shape [N, 3], dtype=float32, coordinates in Angstrom
            - symbols: list[str] of length N, atomic symbols (e.g., 'C', 'N')
    """

    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Enable proximity bonding to infer bonds when CONECT records are missing (common in ligand PDBs)
    mol = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False, proximityBonding=True)
    if mol is None:
        raise ValueError(f"Failed to load molecule from PDB: {pdb_path}")

    # Try sanitize if requested, but be tolerant to valence issues in cofactors
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Fallback: keep as-is to at least extract coordinates/elements
            pass

    if mol.GetNumConformers() == 0:
        raise ValueError(f"No conformer/coordinates found in PDB: {pdb_path}")

    # if remove_hs:
    #     mol = Chem.RemoveHs(mol)

    # conf = mol.GetConformer()
    # positions = conf.GetPositions()

    # atoms = list(mol.GetAtoms())
    # Z = np.array([a.GetAtomicNum() for a in atoms], dtype=np.int32)
    # coords = np.asarray(positions, dtype=np.float32)
    if remove_hs:
        mol = Chem.RemoveHs(mol, sanitize=False)
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    atoms = list(mol.GetAtoms())
    Z = np.array([a.GetAtomicNum() for a in atoms], dtype=np.int32)
    coords = np.asarray(positions, dtype=np.float32)

    if coords.shape[0] != Z.shape[0]:
        raise ValueError(
            f"Number of atoms ({Z.shape[0]}) does not match number of coordinates ({coords.shape[0]})."
        )

    if return_symbols:
        symbols = [a.GetSymbol() for a in atoms]
        return Z, coords, symbols

    return Z, coords

def construct_interface_modeling_dataset(
    data_dir: str,
):
    complexes = []
    for entry in sorted(os.listdir(data_dir)):
        complex_dir = os.path.join(data_dir, entry)
        if os.path.isdir(complex_dir):
            complexes.append(complex_dir)

    print(f"Number of complexes: {len(complexes)}")

    dataset = []

    for complex_dir in tqdm(complexes):
    
        # Extract complex name from directory path
        name = os.path.basename(complex_dir)

        p1_path = os.path.join(complex_dir, 'protein1.pdb')
        p2_path = os.path.join(complex_dir, 'protein2.pdb')

        # Build protein structures.
        p1 = parse_pdb(p1_path)[0]
        p2 = parse_pdb(p2_path)[0]

        p1_ca_coords = p1['pos_heavyatom'][:, BBHeavyAtom.CA, :]
        p2_ca_coords = p2['pos_heavyatom'][:, BBHeavyAtom.CA, :]

        p1_residue = p1['aa']
        p2_residue = p2['aa']
        
        # Extract interface coordinates.
        _, p1_interface_mask, p2_interface_mask, p1_interface_residues, p2_interface_residues = get_interface_from_graphs(
            p1_ca_coords, 
            p2_ca_coords, 
            cutoff=8.0 # 8.0 Å for C-alpha atoms cutoff distance.
        )
        
        # Check if interface has any residues - skip if none
        if len(p1_interface_residues) == 0 or len(p2_interface_residues) == 0:
            print(f"⚠️  Complex {name}: No interface residues found, skipping...")
            continue

        # Move proteins to centroid.
        p1_centroid = torch.sum(p1_ca_coords, dim=0) / len(p1_ca_coords)
        p2_centroid = torch.sum(p2_ca_coords, dim=0) / len(p2_ca_coords)
        p1_ca_coords = p1_ca_coords - p1_centroid
        p2_ca_coords = p2_ca_coords - p2_centroid

        # Build separate interfaces for p1 and p2.
        p1_interface_coords = p1_ca_coords[p1_interface_mask]
        p2_interface_coords = p2_ca_coords[p2_interface_mask]

        # Compute per-interface ellipsoids
        i1_mu, i1_sigma = get_elilipsoid_for_interface(p1_interface_coords)
        i2_mu, i2_sigma = get_elilipsoid_for_interface(p2_interface_coords)
        
        data = {
            'name': name,
            'p1_residue': p1_residue,
            'p2_residue': p2_residue,
            'p1_coords': p1_ca_coords,
            'p2_coords': p2_ca_coords,
            'p1_interface_mask': p1_interface_mask,
            'p2_interface_mask': p2_interface_mask,
            'p1_interface_coords': p1_interface_coords,
            'i1_mu': i1_mu,
            'i1_sigma': i1_sigma,
            'p2_interface_coords': p2_interface_coords,
            'i2_mu': i2_mu,
            'i2_sigma': i2_sigma,
            'p1_interface_residues': p1_interface_residues,
            'p2_interface_residues': p2_interface_residues,
        }
        # print(data)
    
        dataset.append(data)
    
    dataset = Dataset.from_list(dataset)
    print(f"Number of complexes for interface modeling: {len(dataset)}")
    return dataset

def construct_flow_matching_dataset(
    data_dir: str,
    ligand_center: bool = True,
):
    complexes = []
    for entry in sorted(os.listdir(data_dir)):
        complex_dir = os.path.join(data_dir, entry)
        if os.path.isdir(complex_dir):
            complexes.append(complex_dir)

    dataset = []

    interface_modeling_dataset = load_from_disk("./data/Moloctite/interface_modeling_dataset_1208")
    complexes_with_interface = interface_modeling_dataset['name']
    print(f"Number of complexes with interface: {len(complexes_with_interface)}")

    for complex_dir in tqdm(complexes):

        # Mark if the complex has interface.
        name = os.path.basename(complex_dir)

        interface_flag = name in complexes_with_interface

        p1_path = os.path.join(complex_dir, 'protein1.pdb')
        p2_path = os.path.join(complex_dir, 'protein2.pdb')
        # lig_path = os.path.join(complex_dir, 'ligand.pdb')
        lig_path = os.path.join(complex_dir, 'ligand_rcsb.sdf')

        # Build protein structures.
        p1 = parse_pdb(p1_path)[0]
        p2 = parse_pdb(p2_path)[0]

        p1_ca_coords = p1['pos_heavyatom'][:, BBHeavyAtom.CA, :]
        p2_ca_coords = p2['pos_heavyatom'][:, BBHeavyAtom.CA, :]

        
        # Extract interface coordinates.
        # if interface_flag:
        #     _, p1_interface_mask, p2_interface_mask, p1_interface_residues, p2_interface_residues = get_interface_from_graphs(
        #         p1_ca_coords, 
        #         p2_ca_coords, 
        #         cutoff=8.0 # 8.0 Å for C-alpha atoms cutoff distance.
        #     )

        try: 
            lig = parse_pdb_ligand(lig_path, heavy_only=True, mode='full')
        except Exception as e:
            print(f"Error: {e}")
            print(f"Ligand path: {lig_path} does not exist.")
            continue
        # Encoded ligand atom type with hybridization and aromaticity.
        lig_full_element = lig['ligand_atom_feature_full']
        lig_coords = lig['pos']
        # lig_coords_gt = deepcopy(lig_coords)
        # print(center)
        if ligand_center:
            # Find the center of the ligand.
            center = np.sum(lig_coords, axis=0) / len(lig_coords)
            lig_coords = lig_coords - center

            center = torch.from_numpy(center).to(p1['pos_heavyatom'].dtype)

            # Move p1 to the center of the ligand.
            p1['pos_heavyatom'] = p1['pos_heavyatom'] - center[None, None, :]
            # Calculate the torsion angles after translation.
            p1['torsion_angle'], p1['torsion_angle_mask'] = get_torsion_angle(p1['pos_heavyatom'], p1['aa'])
            
            # Move p2 to its own center (centroid)
            p2_ca_coords = p2['pos_heavyatom'][:, BBHeavyAtom.CA, :]  # [L, 3]
            # p2_ca_coords_ligand_center = p2_ca_coords - center[None, :].numpy()
            # if interface_flag:
            #     p2_interface_coords = p2_ca_coords_ligand_center[p2_interface_mask]
            #     p2_mu, p2_sigma = get_elilipsoid_for_interface(p2_interface_coords)
            # else:
            #     p2_interface_coords = p2_ca_coords_ligand_center[random.choice(range(len(p2_ca_coords_ligand_center)), 20)]
            #     p2_mu, p2_sigma = get_elilipsoid_for_interface(p2_interface_coords)
            
            p2_center = torch.mean(p2_ca_coords, dim=0)  # [3]
            p2['pos_heavyatom'] = p2['pos_heavyatom'] - p2_center[None, None, :]

            
            # Apply random rotation only (no translation)
            rot_T, _ = random_rotation_translation(translation_distance=0.0)
            # rot_T: [3, 3] rotation matrix
            
            # Apply rotation to all heavy atoms
            # pos_heavyatom shape: [L, A, 3]
            L, A = p2['pos_heavyatom'].shape[:2]
            p2_coords_flat = p2['pos_heavyatom'].reshape(L * A, 3)  # [L*A, 3]
            
            # Apply rotation: X_rotated = R @ X_centered
            p2_coords_flat_transformed = (rot_T @ p2_coords_flat.T).T
            p2['pos_heavyatom'] = p2_coords_flat_transformed.reshape(L, A, 3)
            
            # Calculate the torsion angles after transformation
            p2['torsion_angle'], p2['torsion_angle_mask'] = get_torsion_angle(p2['pos_heavyatom'], p2['aa'])
            
            # Calculate inverse transformation to restore p2 to correct position in complex
            # Current: p2_rotated = R @ (p2_orig - p2_center)
            # Target: p2_target = p2_orig - center (correct position relative to ligand center)
            # So: p2_target = R^T @ p2_rotated + (p2_center - center)
            R = rot_T  # [3, 3]
            R_inv = R.T  # [3, 3]
            t_inv = p2_center - center  # [3]
            
            # Convert to numpy for storage
            R_inv_np = R_inv.numpy().astype(np.float32)  # [3, 3]
            t_inv_np = t_inv.numpy().astype(np.float32)  # [3]
        else:
            # Move the ligand to the center of the protein 1, and move the protein 2 to the center of itself with random rotation.
            p1_ca_coords = p1['pos_heavyatom'][:, BBHeavyAtom.CA, :]
            center = torch.sum(p1_ca_coords, dim=0) / len(p1_ca_coords)

            # 1. Update Ligand coordinates (center relative to P1 center)
            lig_coords = lig_coords - center[None, :].numpy()

            # 2. Update Protein 1 coordinates (center at origin)
            p1['pos_heavyatom'] = p1['pos_heavyatom'] - center[None, None, :]
            # Recalculate torsion angles for P1 after translation (consistency with if block)
            p1['torsion_angle'], p1['torsion_angle_mask'] = get_torsion_angle(p1['pos_heavyatom'], p1['aa'])

            # 3. Handle Protein 2 (Center at self-origin + Random Rotation)
            # Calculate P2's original geometric center
            p2_ca_coords = p2['pos_heavyatom'][:, BBHeavyAtom.CA, :]
            p2_center = torch.mean(p2_ca_coords, dim=0)

            # Generate random rotation (translation_distance=0 because we center it manually)
            rot_T, _ = random_rotation_translation(translation_distance=0.0)
            
            # Flatten P2 coordinates for matrix multiplication
            L, A = p2['pos_heavyatom'].shape[:2]
            p2_coords_flat = p2['pos_heavyatom'].reshape(L * A, 3)

            # Transform P2: (X - P2_center) * R^T (equivalent to R @ vec)
            # Move P2 to its own center (0,0,0) then rotate
            p2_coords_flat_centered = p2_coords_flat - p2_center
            p2_coords_flat_transformed = (rot_T @ p2_coords_flat_centered.T).T
            
            # Update P2 coordinates
            p2['pos_heavyatom'] = p2_coords_flat_transformed.reshape(L, A, 3)
            # Recalculate torsion angles for P2
            p2['torsion_angle'], p2['torsion_angle_mask'] = get_torsion_angle(p2['pos_heavyatom'], p2['aa'])

            # 4. Calculate the Inverse Transformation (Ground Truth Label)
            # We need (R_inv, t_inv) to map the 'Randomized P2' back to 'GT P2 relative to Centered P1'.
            # Target (GT relative): P2_orig - P1_center
            # Current (Input): R @ (P2_orig - P2_center)
            #
            # Derivation:
            # R_inv @ Current = P2_orig - P2_center
            # R_inv @ Current + (P2_center - P1_center) = P2_orig - P1_center (Target)
            #
            # So: R_inv = R^T, t_inv = P2_center - P1_center
            
            R_inv = rot_T.T
            t_inv = p2_center - center

            # Convert to numpy for storage
            R_inv_np = R_inv.numpy().astype(np.float32)
            t_inv_np = t_inv.numpy().astype(np.float32)

        data = {
            'name': name,
            'R_inv': R_inv_np,  # Inverse rotation matrix to restore p2
            't_inv': t_inv_np,  # Inverse translation vector to restore p2
            'p1': p1,  # p1 structure dict
            'p2': p2,  # p2 structure dict (after transformation)
            'lig_seq': np.asarray(lig_full_element, dtype=np.int32),
            'lig_coords': np.asarray(lig_coords, dtype=np.float32), # moved ligand coordinates
            'interface_flag': interface_flag,
        }

        dataset.append(data)

    dataset = Dataset.from_list(dataset)
    print(f"Number of complexes for flow matching: {len(dataset)}")

    return dataset

def filter_dataset_by_protein_length(dataset_path, output_path, min_length=50, max_length=700):
    """
    Filter dataset to keep only samples where both proteins have length between min_length and max_length
    
    Args:
        dataset_path: Path to the input dataset
        output_path: Path to save the filtered dataset
        min_length: Minimum protein length to keep (default: 50)
        max_length: Maximum protein length to keep (default: 500)
    """
    from datasets import load_from_disk
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Filtering criteria: {min_length} <= protein_length <= {max_length}")
    
    # Filter dataset
    filtered_data = []
    removed_count = 0
    removed_too_short = 0
    removed_too_long = 0
    
    for i, data in enumerate(tqdm(dataset, desc="Filtering dataset")):
        # Get protein lengths
        p1_length = len(data['p1']['aa']) if isinstance(data['p1']['aa'], (list, np.ndarray)) else data['p1']['aa'].shape[0]
        p2_length = len(data['p2']['aa']) if isinstance(data['p2']['aa'], (list, np.ndarray)) else data['p2']['aa'].shape[0]
        
        # Check if both proteins are within the length range
        p1_valid = min_length <= p1_length <= max_length
        p2_valid = min_length <= p2_length <= max_length
        
        if p1_valid and p2_valid:
            filtered_data.append(data)
        else:
            removed_count += 1
            if not p1_valid:
                if p1_length < min_length:
                    removed_too_short += 1
                else:
                    removed_too_long += 1
            if not p2_valid:
                if p2_length < min_length:
                    removed_too_short += 1
                else:
                    removed_too_long += 1
    
    print(f"\nFiltering complete:")
    print(f"  Original size: {len(dataset)}")
    print(f"  Filtered size: {len(filtered_data)}")
    print(f"  Total removed: {removed_count} samples")
    print(f"    - Too short (< {min_length}): {removed_too_short} samples")
    print(f"    - Too long (> {max_length}): {removed_too_long} samples")
    print(f"  Kept: {len(filtered_data)} samples ({(len(filtered_data)/len(dataset)*100):.2f}%)")
    
    # Create new dataset
    filtered_dataset = Dataset.from_list(filtered_data)
    
    # Save filtered dataset
    print(f"\nSaving filtered dataset to {output_path}...")
    filtered_dataset = filtered_dataset.save_to_disk(output_path)
    print(f"✓ Filtered dataset saved successfully!")
    
    return filtered_dataset

def check_ligand(data_dir):
    complexes = []
    for entry in sorted(os.listdir(data_dir)):
        complex_dir = os.path.join(data_dir, entry)
        if os.path.isdir(complex_dir):
            complexes.append(complex_dir)

    for complex_dir in tqdm(complexes):
        name = os.path.basename(complex_dir).split('_')[1]
        lig_path = os.path.join(complex_dir, 'ligand_rcsb.sdf')
        
        if not os.path.exists(lig_path):
            print(f"Warning: No ligand file found: {lig_path}")
            continue

        try:
            ligand = parse_pdb_ligand(lig_path, heavy_only=True, mode='full')
            print(ligand['element'])
            print(ligand['aromatic_list'])
            print(ligand['ligand_atom_feature_full'])
            print(ligand['pos'])
            print(f"Number of atoms: {len(ligand['pos'])}")
            print(f"Number of atoms: {len(ligand['ligand_atom_feature_full'])}")
            print(f"##################################")
        except (rdkit.Chem.rdchem.AtomValenceException, ValueError) as e:
            print(f"Error: {e}")
            print(f"Ligand path: {lig_path}")
            continue

def check_translation(dataset):
    trans = []
    for data in tqdm(dataset):
        tran = data['t_inv']
        trans.append(tran)
    
    print(f"mean of translation: {np.mean(trans, axis=0)}")
    print(f"std of translation: {np.std(trans, axis=0)}")

def check_center(dataset):
    val = []
    for data in tqdm(dataset):
        t_inv = data['t_inv']
        # Convert to tensor if it's a list (HuggingFace datasets serializes tensors as lists)
        if isinstance(t_inv, list):
            t_inv = torch.tensor(t_inv)
        
        p1 = data['p1']
        p2 = data['p2']
        
        # Convert pos_heavyatom to tensor if it's a list
        p1_pos_heavyatom = p1['pos_heavyatom']
        if isinstance(p1_pos_heavyatom, list):
            p1_pos_heavyatom = torch.tensor(p1_pos_heavyatom)
        
        p2_pos_heavyatom = p2['pos_heavyatom']
        if isinstance(p2_pos_heavyatom, list):
            p2_pos_heavyatom = torch.tensor(p2_pos_heavyatom)
        
        p1_ca_coords = p1_pos_heavyatom[:, BBHeavyAtom.CA, :]
        p2_ca_coords = p2_pos_heavyatom[:, BBHeavyAtom.CA, :]
        p1_center = torch.mean(p1_ca_coords, axis=0)
        p2_center = torch.mean(p2_ca_coords, axis=0)
        print(f"t_inv: {t_inv}")
        print(f"p1 center: {p1_center}")
        print(f"p2 center: {p2_center}")
        
        lig_coords = data['lig_coords']
        if isinstance(lig_coords, list):
            lig_coords = torch.tensor(lig_coords)
        lig_center = torch.mean(lig_coords, axis=0)
        print(f"lig center: {lig_center}")
        print(f"p2_center_original: {lig_center + t_inv}")
        print(f"p2_center + p1_center: {p2_center + p1_center}")
        val.append(p2_center + p1_center - t_inv)
        print(f"####################################")

    print(f"mean of val: {np.mean(val, axis=0)}")
    print(f"std of val: {np.std(val, axis=0)}")

if __name__ == "__main__":
    # Construct interface modeling dataset.
    dataset =construct_interface_modeling_dataset(
        data_dir="./data/TernaryDB/MGD_test"
    )
    dataset = dataset.save_to_disk("interface_modeling_dataset_eval")

    # dataset = load_from_disk("interface_modeling_dataset_v2")
    # print(dataset)
    # print(len(dataset))

    # Construct flow matching dataset.
    # dataset = construct_flow_matching_dataset(
    #     data_dir="./data/TernaryDB/MGD_test", ligand_center=True
    # )
    # dataset = dataset.save_to_disk("TernaryDataset_test")
    
    # Filter dataset to keep only proteins with 50 <= length <= 500
    # filter_dataset_by_protein_length(
    #     dataset_path="./data/Moloctite/TernaryDataset",
    #     output_path="./data/Moloctite/TernaryDataset_filtered",
    #     min_length=5,
    #     max_length=700
    # )

    # Randomly print 10 samples and save to .log
    # dataset = load_from_disk("flow_matching_dataset_v1")
    # num_samples = min(10, len(dataset))
    # random_indices = sample(range(len(dataset)), num_samples)
    # samples = [dataset[i] for i in random_indices]

    # with open("random_samples.log", "w", encoding="utf-8") as f:
    #     for i, sample_item in enumerate(samples):
    #         f.write(f"Sample {i+1}:\n")
    #         # Use json.dumps for pretty printing, handle numpy types
    #         def default(o):
    #             if hasattr(o, 'tolist'):
    #                 return o.tolist()
    #             return str(o)
    #         f.write(json.dumps(sample_item, indent=2, default=default, ensure_ascii=False))
    #         f.write("\n\n")
    # print(f"Randomly printed {num_samples} samples and saved to random_samples.log")

    # Check ligand.
    # check_ligand(data_dir="./data/TernaryDB/MGD_Train")

    # dataset = load_from_disk("./data/Moloctite/TernaryDataset_filtered")
    # check_center(dataset)

    pass