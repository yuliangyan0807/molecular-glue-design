import os
import random
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from copy import deepcopy
from configs.config import DATASET_ARGS, DictToObject
from pocket import get_elilipsoid_for_interface, get_interface_from_graphs

from datasets import Dataset, load_from_disk

from DeepTernary.deepternary.models.process_mols import (
    distance_featurizer,
    get_geometry_graph_ring,
    get_lig_graph_revised,
    get_rdkit_coords_v2,
    get_rec_graph,
    get_receptor_inference,
    lig_atom_featurizer,
    read_molecule,
    rigid_transform_Kabsch_3D
)
from DeepTernary.deepternary.models.ternary_pdb import get_pocket_and_mask
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

    mol = Chem.MolFromPDBFile(pdb_path, sanitize=sanitize, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to load molecule from PDB: {pdb_path}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"No conformer/coordinates found in PDB: {pdb_path}")

    if remove_hs:
        mol = Chem.RemoveHs(mol)

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
    
    ds_cfg = DictToObject(DATASET_ARGS)

    dataset = []

    for complex_dir in tqdm(complexes):
    
        # Extract complex name from directory path
        name = os.path.basename(complex_dir)

        p1_path = os.path.join(complex_dir, 'protein1.pdb')
        p2_path = os.path.join(complex_dir, 'protein2.pdb')

        recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p1_path)
        p1_graph = get_rec_graph(
            recs, recs_coords, c_alpha_coords, n_coords, c_coords,
            use_rec_atoms=ds_cfg.use_rec_atoms,
            rec_radius=ds_cfg.rec_graph_radius,
            surface_max_neighbors=ds_cfg.surface_max_neighbors,
            surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
            surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
            c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,
        )
        protein1_coords_gt = deepcopy(p1_graph.ndata['x'])

        recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p2_path)
        p2_graph = get_rec_graph(
            recs, recs_coords, c_alpha_coords, n_coords, c_coords,
            use_rec_atoms=ds_cfg.use_rec_atoms,
            rec_radius=ds_cfg.rec_graph_radius,
            surface_max_neighbors=ds_cfg.surface_max_neighbors,
            surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
            surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
            c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,
        )
        protein2_coords_gt = deepcopy(p2_graph.ndata['x'])

        # Get the protein1 and protein2's residue sequence.
        p1_residue = p1_graph.ndata['feat'][:, 0]
        p2_residue = p2_graph.ndata['feat'][:, 0]
        
        # Extract interface coordinates.
        interface_coords, p1_interface_mask, p2_interface_mask, p1_interface_residues, p2_interface_residues = get_interface_from_graphs(
            protein1_coords_gt, 
            protein2_coords_gt, 
            cutoff=8.0
        )
        
        # Check if interface has any residues - skip if none
        if len(p1_interface_residues) == 0 and len(p2_interface_residues) == 0:
            print(f"⚠️  Complex {name}: No interface residues found, skipping...")
            continue

        
        # Store original interface coordinates before any transformations
        interface_coords_original = deepcopy(interface_coords)
        # Flag if only one side has interface (based on original detection)
        single_sided = (len(p1_interface_residues) == 0) != (len(p2_interface_residues) == 0)

        # Decide randomly whether to move p1 or p2 (50% chance each)
        move_p1 = random.random() < 0.5

        if move_p1:
            # random move p1
            protein1_rot_T, protein1_rot_b = random_rotation_translation(translation_distance=5)
            protein1_coord_to_move = p1_graph.ndata['x']
            protein1_mean_to_remove = protein1_coord_to_move.mean(dim=0, keepdims=True)
            p1_graph.ndata['x'] = (protein1_rot_T @ (protein1_coord_to_move - protein1_mean_to_remove).T).T + protein1_rot_b
        else:
            # random move p2
            protein2_rot_T, protein2_rot_b = random_rotation_translation(translation_distance=5)
            protein2_coord_to_move = p2_graph.ndata['x']
            protein2_mean_to_remove = protein2_coord_to_move.mean(dim=0, keepdims=True)
            p2_graph.ndata['x'] = (protein2_rot_T @ (protein2_coord_to_move - protein2_mean_to_remove).T).T + protein2_rot_b

        # Get the protein1 and protein2's C-alpha coordinates.
        p1_coords = p1_graph.ndata['x']
        p2_coords = p2_graph.ndata['x']

        # Build separate interfaces for p1 and p2 (moved protein's interface moves accordingly)
        p1_interface_coords = p1_coords[p1_interface_mask]
        p2_interface_coords = p2_coords[p2_interface_mask]

        # Fallback: make both sides non-empty if possible
        p1_interface_coords = p1_interface_coords if len(p1_interface_coords) > 0 else p2_interface_coords
        p2_interface_coords = p2_interface_coords if len(p2_interface_coords) > 0 else p1_interface_coords

        # Compute per-interface ellipsoids
        i1_mu, i1_sigma = get_elilipsoid_for_interface(p1_interface_coords)
        i2_mu, i2_sigma = get_elilipsoid_for_interface(p2_interface_coords)
        
        data = {
            'name': name,
            'p1_residue': p1_residue,
            'p2_residue': p2_residue,
            'p1_coords_gt': protein1_coords_gt,
            'p2_coords_gt': protein2_coords_gt,
            'p1_coords': p1_coords,
            'p2_coords': p2_coords,
            'p1_interface_mask': p1_interface_mask,
            'p2_interface_mask': p2_interface_mask,
            'interface_coords_original': interface_coords_original,
            'p1_interface_coords': p1_interface_coords,
            'i1_mu': i1_mu,
            'i1_sigma': i1_sigma,
            'p2_interface_coords': p2_interface_coords,
            'i2_mu': i2_mu,
            'i2_sigma': i2_sigma,
            'p1_interface_residues': p1_interface_residues,
            'p2_interface_residues': p2_interface_residues,
            'single_sided': single_sided,
        }
        # print(data)
    
        dataset.append(data)
    
    dataset = Dataset.from_list(dataset)

    return dataset

def construct_flow_matching_dataset(
    data_dir: str,
):
    complexes = []
    for entry in sorted(os.listdir(data_dir)):
        complex_dir = os.path.join(data_dir, entry)
        if os.path.isdir(complex_dir):
            complexes.append(complex_dir)
    
    ds_cfg = DictToObject(DATASET_ARGS)

    dataset = []

    interface_modeling_dataset = load_from_disk("interface_modeling_dataset_v2")
    complexes_with_interface = interface_modeling_dataset['name']
    print(f"Number of complexes with interface: {len(complexes_with_interface)}")

    for complex_dir in tqdm(complexes):

        # Mark if the complex has interface.
        name = os.path.basename(complex_dir)
        if name not in complexes_with_interface:
            interface_flag = False
        else:
            interface_flag = True

        p1_path = os.path.join(complex_dir, 'protein1.pdb')
        p2_path = os.path.join(complex_dir, 'protein2.pdb')
        lig_path = os.path.join(complex_dir, 'ligand.pdb')

        lig_Z, lig_coords = read_ligand_pdb(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)

        # Build receptor graphs
        recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p1_path)
        p1_graph = get_rec_graph(
            recs, recs_coords, c_alpha_coords, n_coords, c_coords,
            use_rec_atoms=ds_cfg.use_rec_atoms,
            rec_radius=ds_cfg.rec_graph_radius,
            surface_max_neighbors=ds_cfg.surface_max_neighbors,
            surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
            surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
            c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,
        )
        protein1_coords_gt = deepcopy(p1_graph.ndata['x'])

        recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p2_path)
        p2_graph = get_rec_graph(
            recs, recs_coords, c_alpha_coords, n_coords, c_coords,
            use_rec_atoms=ds_cfg.use_rec_atoms,
            rec_radius=ds_cfg.rec_graph_radius,
            surface_max_neighbors=ds_cfg.surface_max_neighbors,
            surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
            surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
            c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors,
        )
        protein2_coords_gt = deepcopy(p2_graph.ndata['x'])

        # Get the protein1 and protein2's residue sequence.
        p1_residue = p1_graph.ndata['feat'][:, 0]
        p2_residue = p2_graph.ndata['feat'][:, 0]

        # Randomly move one protein (p1 or p2) as in interface dataset
        move_p1 = random.random() < 0.5
        if move_p1:
            rot_T, rot_b = random_rotation_translation(translation_distance=5)
            coords_to_move = p1_graph.ndata['x']
            mean_to_remove = coords_to_move.mean(dim=0, keepdims=True)
            p1_graph.ndata['x'] = (rot_T @ (coords_to_move - mean_to_remove).T).T + rot_b
        else:
            rot_T, rot_b = random_rotation_translation(translation_distance=5)
            coords_to_move = p2_graph.ndata['x']
            mean_to_remove = coords_to_move.mean(dim=0, keepdims=True)
            p2_graph.ndata['x'] = (rot_T @ (coords_to_move - mean_to_remove).T).T + rot_b

        # Compute inverse transform that maps moved protein back to original position
        # X_moved = R @ (X_orig - mean) + b  =>  X_orig = R^T @ X_moved + (mean - R^T @ b)
        R = rot_T  # [3,3]
        t = rot_b  # [3]
        R_inv = R.T
        t_inv = (mean_to_remove.squeeze(0) - R_inv @ t.squeeze(0))

        # Collect current (possibly moved) coords
        p1_coords = p1_graph.ndata['x']
        p2_coords = p2_graph.ndata['x']

        data = {
            'name': name,
            'R_inv': R_inv,
            't_inv': t_inv,
            'p1_residue': p1_residue,
            'p2_residue': p2_residue,
            'p1_coords_gt': protein1_coords_gt,
            'p2_coords_gt': protein2_coords_gt,
            'p1_coords': p1_coords,
            'p2_coords': p2_coords,
            'lig_seq': np.asarray(lig_Z, dtype=np.int32),
            'lig_coords': np.asarray(lig_coords, dtype=np.float32),
            'interface_flag': interface_flag,
        }

        dataset.append(data)

    dataset = Dataset.from_list(dataset)
    print(f"Number of complexes for flow matching: {len(dataset)}")

    return dataset

if __name__ == "__main__":
    # Construct interface modeling dataset.
    # dataset =construct_interface_modeling_dataset(
    #     data_dir="./data/TernaryDB/MGD_Train"
    # )
    # dataset = dataset.save_to_disk("interface_modeling_dataset_v3")

    # dataset = load_from_disk("interface_modeling_dataset_v2")
    # print(dataset)
    # print(len(dataset))

    # Construct flow matching dataset.
    dataset = construct_flow_matching_dataset(
        data_dir="./data/TernaryDB/MGD_Train"
    )
    dataset = dataset.save_to_disk("flow_matching_dataset_v1")