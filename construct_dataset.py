import os
import random
from copy import deepcopy
from configs.config import DATASET_ARGS, DictToObject
from pocket import get_elilipsoid_for_interface, get_interface_from_graphs

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


def construct_interface_modeling_dataset(
    data_dir: str,
):
    complexes = []
    for entry in sorted(os.listdir(data_dir)):
        complex_dir = os.path.join(data_dir, entry)
        if os.path.isdir(complex_dir):
            complexes.append(complex_dir)
    
    # print(complexes)

    for complex_dir in complexes:
        ds_cfg = DictToObject(DATASET_ARGS)
    
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

        # print(p1_graph.ndata['x'].shape)
        # print(p2_graph.ndata['x'].shape)
        # print(protein1_coords_gt.shape)
        # print(protein2_coords_gt.shape)
        # print(p1_graph.ndata['x'])
        # print(p2_graph.ndata['x'])
        
        # Extract interface coordinates.
        interface_coords, p1_interface_mask, p2_interface_mask, p1_interface_residues, p2_interface_residues = get_interface_from_graphs(
            protein1_coords_gt, 
            protein2_coords_gt, 
            cutoff=8.0
        )
        
        # Check if interface has any residues - skip if none
        if len(p1_interface_residues) == 0 and len(p2_interface_residues) == 0:
            print(f"⚠️  Complex {name}: No interface residues found, skipping...")

        
        print(interface_coords.shape)
        print(p1_interface_mask.shape)
        print(p2_interface_mask.shape)
        print(p1_interface_residues.shape)
        print(p2_interface_residues.shape)
        print(interface_coords)
        print(p1_interface_mask)
        print(p2_interface_mask)
        print(p1_interface_residues)
        print(p2_interface_residues)
        
        # Store original interface coordinates before any transformations
        interface_coords_gt = deepcopy(interface_coords)

        # Get ellipsoid parameters for interface
        mean, covariance = get_elilipsoid_for_interface(interface_coords)
        print(mean)
        print(covariance)
        break

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

        data = dict(
            rec_graph=p1_graph,
            rec2_graph=p2_graph,
            complex_name=[name],
            protein1_coords_gt=[protein1_coords_gt],
            protein2_coords_gt=[protein2_coords_gt],
            # Interface information
            interface_coords=[interface_coords],
            interface_coords_gt=[interface_coords_gt],
            p1_interface_mask=[p1_interface_mask],
            p2_interface_mask=[p2_interface_mask],
            p1_interface_residues=[p1_interface_residues],
            p2_interface_residues=[p2_interface_residues],
        )

        return data


if __name__ == "__main__":
    construct_interface_modeling_dataset(
        data_dir="./data/TernaryDB/MGD_Train"
    )