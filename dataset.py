import os
import torch
from copy import deepcopy
from functools import partial
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdMolAlign import GetAlignmentTransform
from mmengine.config import Config
from tqdm import tqdm

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

# Simple class to convert dict to object for dot notation access
class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

# Config for molecular glue design.
DATASET_ARGS = {
    ### triplet preprocess dataset args start
    'min_lig_atoms': 3,
    'min_pocket_atoms': 3,
    'preprocess_sub_path': 'pdb2311_merge',
    'freeze': 'protein1',
    'random_flip_proteins': True,
    ### triplet preprocess dataset args end
    "geometry_regularization_ring": True,
    "use_rdkit_coords": True,
    "bsp_proteins": False,
    "dataset_size": None,
    "translation_distance": 5.0,
    "n_jobs": 20,
    "chain_radius": 11,
    "rec_graph_radius": 30,
    "c_alpha_max_neighbors": 10,
    "lig_graph_radius": 5,
    "lig_max_neighbors": None,
    "pocket_cutoff": 6,     # pocket cutoff between alpha C of protein and the ligand
    'pocket_cutoff_p12': 10,    # pocket cutoff between alpha C of proteins
    "pocket_mode": "match_atoms_to_lig",
    "remove_h": True,
    "only_polar_hydrogens": False,
    "use_rec_atoms": False,
    "surface_max_neighbors": 5,
    "surface_graph_cutoff": 5,
    "surface_mesh_cutoff": 2,
    "subgraph_augmentation": False,
    "min_shell_thickness": 3,
    "rec_subgraph": False,
    "subgraph_radius": 10,
    "subgraph_max_neigbor": 8,
    "subgraph_cutoff": 4
}


@torch.no_grad()
def _build_data_dict_for_complex_dir(complex_dir):
    """
    Build a single data dict (as used by predict_one_unbound) from a complex directory
    that contains ligand.pdb, protein1.pdb, protein2.pdb.

    Returns: dict matching keys prepared for model(**data, mode='predict').
    """
    ds_cfg = DictToObject(DATASET_ARGS)
    
    # Extract complex name from directory path
    name = os.path.basename(complex_dir)

    lig_path = os.path.join(complex_dir, 'ligand.pdb')
    p1_path = os.path.join(complex_dir, 'protein1.pdb')
    p2_path = os.path.join(complex_dir, 'protein2.pdb')

    lig = read_molecule(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)

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

    lig, lig_graph = get_lig_graph_revised(
        lig,
        name=name,
        max_neighbors=ds_cfg.lig_max_neighbors,
        ideal_path=None,
        use_random_coords=False,
        seed=0,
        use_rdkit_coords=True,
        radius=ds_cfg.lig_graph_radius,
    )
    geometry_graph = get_geometry_graph_ring(lig)

    lig_coords_gt = deepcopy(lig_graph.ndata['x'])
    p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = get_pocket_and_mask(
        lig_coords_gt, p1_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff
    )
    p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = get_pocket_and_mask(
        lig_coords_gt, p2_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff
    )
    assert torch.allclose(p1lig_lig_pocket_coords, lig_coords_gt[p1lig_lig_pocket_mask])

    p1lig_lig_pocket_coords = lig_coords_gt[p1lig_lig_pocket_mask]
    p1lig_p1_pocket_coords = deepcopy(p1lig_lig_pocket_coords)

    data = dict(
        lig_graph=lig_graph,
        rec_graph=p1_graph,
        rec2_graph=p2_graph,
        geometry_graph=geometry_graph,
        complex_name=[name],
        rec2_coords=[protein2_coords_gt],
        p1lig_p1_pocket_mask=[p1lig_p1_pocket_mask],
        p1lig_p1_pocket_coords=[p1lig_p1_pocket_coords],
        p1lig_lig_pocket_mask=[p1lig_lig_pocket_mask],
        p1lig_lig_pocket_coords=[p1lig_lig_pocket_coords],
        p2lig_p2_pocket_mask=[p2lig_p2_pocket_mask],
        p2lig_p2_pocket_coords=[p2lig_lig_pocket_coords_origin],
        p2lig_lig_pocket_mask=[p2lig_lig_pocket_mask],
        p2lig_lig_pocket_coords=[p2lig_lig_pocket_coords_origin],
    )

    return data

def iter_data_dicts_from_ternarydb(base_dir: str = "./data/TernaryDB/pdbs"):
    """
    Iterate all complex subdirectories under base_dir and yield (name, data_dict).
    Only directories containing ligand.pdb, protein1.pdb, protein2.pdb are considered.
    """
    for entry in tqdm(sorted(os.listdir(base_dir))):
        complex_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(complex_dir):
            continue
        lig_path = os.path.join(complex_dir, 'ligand.pdb')
        p1_path = os.path.join(complex_dir, 'protein1.pdb')
        p2_path = os.path.join(complex_dir, 'protein2.pdb')
        if not (os.path.exists(lig_path) and os.path.exists(p1_path) and os.path.exists(p2_path)):
            continue
        try:
            data = _build_data_dict_for_complex_dir(complex_dir)
            yield entry, data
        except Exception as e:
            print(f"Skip {entry}: {e}")


def build_all_data_dicts_from_ternarydb(base_dir: str = "./data/TernaryDB/pdbs"):
    """Return a list of (name, data_dict) for all valid complexes under base_dir."""
    return list(iter_data_dicts_from_ternarydb(base_dir))


if __name__ == '__main__':
    base_dir = "./data/TernaryDB/pdbs"
    dataset = build_all_data_dicts_from_ternarydb(base_dir)
    print(dataset[0:2])
    print(f"Built dataset for {len(dataset)} complexes under {base_dir}")