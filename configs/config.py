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