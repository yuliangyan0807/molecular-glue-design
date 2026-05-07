import math
import numpy as np
from sqlalchemy import true
import torch
from torch_scatter import scatter
import torch.nn.functional as F
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.constants import (AA, max_num_heavyatoms, max_num_hydrogens,
                        restype_to_heavyatom_names, 
                        restype_to_hydrogen_names,
                        BBHeavyAtom)
import utils.constants as constants

def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center    # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center    # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)    # (N, L, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat

def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, L, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p

def _get_torsion(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign( (torch.cross(v1, v2, dim=-1) * v0).sum(-1) )
    dihed = sgn*torch.acos( (n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))
    return dihed

def get_chi_angles(restype, pos14):
    chi_angles = torch.full([4], fill_value=float("inf")).to(pos14)
    base_atom_names = constants.chi_angles_atoms[restype]
    for i, four_atom_names in enumerate(base_atom_names):
        atom_indices = [constants.restype_atom14_name_to_index[restype][a] for a in four_atom_names]
        p = torch.stack([pos14[i] for i in atom_indices])
        # if torch.eq(p, 99999).any():
        #     continue
        torsion = _get_torsion(*torch.unbind(p, dim=0))
        chi_angles[i] = torsion
    return chi_angles


def get_psi_angle(pos14: torch.Tensor) -> torch.Tensor:
    return _get_torsion(pos14[0], pos14[1], pos14[2], pos14[3]).reshape([1]) # af style psi, N,CA,C,O


def get_torsion_angle(pos14: torch.Tensor, aa: torch.LongTensor):
    torsion, torsion_mask = [], []
    for i in range(pos14.shape[0]):
        if aa[i] < constants.AA.UNK: # 0-19
            chi = get_chi_angles(aa[i].item(), pos14[i])
            psi = get_psi_angle(pos14[i])
            torsion_this = torch.cat([psi, chi], dim=0)
            torsion_mask_this = torsion_this.isfinite()
        else:
            torsion_this = torch.full([5], 0.)
            torsion_mask_this = torch.full([5], False)
        torsion.append(torsion_this.nan_to_num(posinf=0.))
        torsion_mask.append(torsion_mask_this)
    
    torsion = torch.stack(torsion) % (2 * math.pi)
    torsion_mask = torch.stack(torsion_mask).bool()

    return torsion, torsion_mask

def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    bfactor_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.float)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True
            bfactor_heavyatom[idx] = res[atom_name].get_bfactor()
    return pos_heavyatom, mask_heavyatom, bfactor_heavyatom

def get_consecutive_flag(chain_nb, res_nb, mask):
    """
    Args:
        chain_nb, res_nb
    Returns:
        consec: A flag tensor indicating whether residue-i is connected to residue-(i+1), 
                BoolTensor, (B, L-1)[b, i].
    """
    d_res_nb = (res_nb[:, 1:] - res_nb[:, :-1]).abs()   # (B, L-1)
    same_chain = (chain_nb[:, 1:] == chain_nb[:, :-1])
    consec = torch.logical_and(d_res_nb == 1, same_chain)
    consec = torch.logical_and(consec, mask[:, :-1])
    return consec


def get_terminus_flag(chain_nb, res_nb, mask):
    consec = get_consecutive_flag(chain_nb, res_nb, mask)
    N_term_flag = F.pad(torch.logical_not(consec), pad=(1, 0), value=1)
    C_term_flag = F.pad(torch.logical_not(consec), pad=(0, 1), value=1)
    return N_term_flag, C_term_flag

def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign( (torch.cross(v1, v2, dim=-1) * v0).sum(-1) )
    dihed = sgn*torch.acos( (n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999) )
    dihed = torch.nan_to_num(dihed)
    return dihed

def get_backbone_dihedral_angles(pos_atoms, chain_nb, res_nb, mask):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
        chain_nb:   (N, L).
        res_nb:     (N, L).
        mask:       (N, L).
    Returns:
        bb_dihedral:    Omega, Phi, and Psi angles in radian, (N, L, 3).
        mask_bb_dihed:  Masks of dihedral angles, (N, L, 3).
    """
    pos_N  = pos_atoms[:, :, BBHeavyAtom.N]   # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C  = pos_atoms[:, :, BBHeavyAtom.C]

    N_term_flag, C_term_flag = get_terminus_flag(chain_nb, res_nb, mask)  # (N, L)
    omega_mask = torch.logical_not(N_term_flag)
    phi_mask = torch.logical_not(N_term_flag)
    psi_mask = torch.logical_not(C_term_flag)

    # N-termini don't have omega and phi
    omega = F.pad(
        dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]), 
        pad=(1, 0), value=0,
    )
    phi = F.pad(
        dihedral_from_four_points(pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:], pos_C[:, 1:]),
        pad=(1, 0), value=0,
    )

    # C-termini don't have psi
    psi = F.pad(
        dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]),
        pad=(0, 1), value=0,
    )

    mask_bb_dihed = torch.stack([omega_mask, phi_mask, psi_mask], dim=-1)
    bb_dihedral = torch.stack([omega, phi, psi], dim=-1) * mask_bb_dihed
    return bb_dihedral, mask_bb_dihed

def pairwise_dihedrals(pos_atoms):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
    Returns:
        Inter-residue Phi and Psi angles, (N, L, L, 2).
    """
    N, L = pos_atoms.shape[:2]
    pos_N  = pos_atoms[:, :, BBHeavyAtom.N]   # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C  = pos_atoms[:, :, BBHeavyAtom.C]

    ir_phi = dihedral_from_four_points(
        pos_C[:,:,None].expand(N, L, L, 3), 
        pos_N[:,None,:].expand(N, L, L, 3), 
        pos_CA[:,None,:].expand(N, L, L, 3), 
        pos_C[:,None,:].expand(N, L, L, 3)
    )
    ir_psi = dihedral_from_four_points(
        pos_N[:,:,None].expand(N, L, L, 3), 
        pos_CA[:,:,None].expand(N, L, L, 3), 
        pos_C[:,:,None].expand(N, L, L, 3), 
        pos_N[:,None,:].expand(N, L, L, 3)
    )
    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)
    return ir_dihed

def get_ligand_atom_features(rdmol):

    num_atoms = rdmol.GetNumAtoms()

    atomic_number = []
    aromatic = []
    hybrid = []
    degree = []
    for atom_idx in range(num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        HYBRID_TYPES = {t: i for i, t in enumerate(HybridizationType.names.values())}
        hybrid.append(HYBRID_TYPES[hybridization])
        degree.append(atom.GetDegree())
    
    node_type = torch.tensor(atomic_number, dtype=torch.long)

    row, col = [], []
    for bond in rdmol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    hs = (node_type == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=num_atoms).numpy()
    # need to change ATOM_FEATS accordingly
    feat_mat = np.array([atomic_number, aromatic, degree, num_hs, hybrid]).transpose()
    return feat_mat

def get_index(atom_num, hybridization, is_aromatic):

    return constants.MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]

def parse_pdb_ligand(path, heavy_only=True, mode='full'):

    assert mode in ['basic', 'add_aromatic', 'full']

    # mol = Chem.MolFromPDBFile(path, sanitize=False)
    mol = Chem.MolFromMolFile(path)
    if mol is None:
        raise ValueError(f"Failed to load molecule from file: {path}. The file may be corrupted or in an unsupported format.")
    Chem.SanitizeMol(mol)
    if heavy_only:
        mol = Chem.RemoveHs(mol)
    
    feat_mat = get_ligand_atom_features(mol)

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in mol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    pos = mol.GetConformer().GetPositions()

    element = []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atomic_number = atom.GetAtomicNum()
        element.append(atomic_number)
        x, y, z = pos[atom_idx]
        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight
    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)
    element = np.array(element, dtype=np.int32)
    pos = np.array(pos, dtype=np.float32)

    row, col, edge_type = [], [], []
    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]
    edge_index = np.array([row, col], dtype=np.int32)
    edge_type = np.array(edge_type, dtype=np.int32)
    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    element_list = element
    hybridization_list = hybridization

    aromatic_list = [v[1] for v in feat_mat]  # feat_mat column 1 is aromatic (0 or 1)

    try:
        x = [get_index(e, h, a) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
    except Exception as e:
        print(e)
        print(element_list)
        print(hybridization_list)
        print(aromatic_list)
        raise e
    x = torch.tensor(x)
    ligand_atom_feature_full = x

    data = {
        'element': element,
        'pos': pos,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'aromatic_list': aromatic_list,
        'center_of_mass': center_of_mass,
        'ligand_atom_feature_full': ligand_atom_feature_full
    }

    return data

def parse_pdb(path, model_id=0, unknown_threshold=1.0):
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)

def parse_biopython_structure(entity, unknown_threshold=1.0):
    chains = Selection.unfold_entities(entity, 'C')
    chains.sort(key=lambda c: c.get_id())
    data = EasyDict({
        'chain_id': [], 'chain_nb': [],
        'resseq': [], 'icode': [], 'res_nb': [],
        'aa': [],
        'pos_heavyatom': [], 'mask_heavyatom': [],
        # 'pos_hydrogen': [], 'mask_hydrogen': [],
        # 'bfactor_heavyatom': [],
    })
    tensor_types = {
        'chain_nb': torch.LongTensor,
        'resseq': torch.LongTensor,
        'res_nb': torch.LongTensor,
        'aa': torch.LongTensor,
        'pos_heavyatom': torch.stack,
        'mask_heavyatom': torch.stack,
        # 'bfactor_heavyatom': torch.stack,
        # 'pos_hydrogen': torch.stack,
        # 'mask_hydrogen': torch.stack,
    }

    count_aa, count_unk = 0, 0

    for i, chain in enumerate(chains):
        seq_this = 0   # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode
        for _, res in enumerate(residues):
            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            # Chain info
            data.chain_id.append(chain.get_id())
            data.chain_nb.append(i)

            # Residue types
            data.aa.append(restype) # Will be automatically cast to torch.long

            # Heavy atoms
            pos_heavyatom, mask_heavyatom, bfactor_heavyatom = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)
            # data.bfactor_heavyatom.append(bfactor_heavyatom)

            # Hydrogen atoms
            # pos_hydrogen, mask_hydrogen = _get_residue_hydrogen_info(res)
            # data.pos_hydrogen.append(pos_hydrogen)
            # data.mask_hydrogen.append(mask_hydrogen)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

    if len(data.aa) == 0:
        return None, None

    if (count_unk / count_aa) >= unknown_threshold:
        return None, None

    seq_map = {}
    for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
        seq_map[(chain_id, resseq, icode)] = i

    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])
    
    # # ignore UNKNOWN residues and nobackbone residues, true for used residue
    # seq_mask = data['aa'] != AA.UNK
    # bb_mask = data['mask_heavyatom'][:, BBHeavyAtom.CA] & data['mask_heavyatom'][:, BBHeavyAtom.C] & data['mask_heavyatom'][:, BBHeavyAtom.N]
    # data['res_mask'] = seq_mask & bb_mask

    return data, seq_map

def get_conditioned_coords(
    p_mask,
    attn,
    topk_k,
):  
    B, _ = p_mask.shape
    device = p_mask.device
    p_valid_counts = p_mask.sum(dim=1)  # (B,)

    p_needs_topk = p_valid_counts > topk_k  # (B,)
    # Initialize with residue mask (for samples that don't need topk)
    i_mask = p_mask.clone()
    
    if p_needs_topk.any():
        # Get top-K indices for all samples
        _, topk_indices = torch.topk(attn, k=topk_k, dim=-1, largest=True)  # (B, K)
        
        # For samples that need topk, reset mask and set topk positions
        # Create batch indices: (B, K)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, topk_k)
        
        # Reset mask for samples that need topk
        i_mask[p_needs_topk] = False
        
        # Set topk positions to True using index_put_ for vectorized assignment
        # Flatten indices for samples that need topk
        need_topk_mask = p_needs_topk.unsqueeze(1).expand(B, topk_k)  # (B, K)
        batch_idx = batch_indices[need_topk_mask]  # (num_needs_topk * K,)
        topk_idx = topk_indices[need_topk_mask]  # (num_needs_topk * K,)
        i_mask[(batch_idx, topk_idx)] = True

    return i_mask

def get_rigid_transform(coords_src, coords_tgt):
    """
    Calculates the optimal rigid transformation (Rotation R, Translation t)
    to align coords_src to coords_tgt using the Kabsch algorithm.
    
    Minimizes RMSD: || (coords_src @ R.T + t) - coords_tgt ||^2
    
    Args:
        coords_src: (N, 3) tensor, points to be moved (e.g., protein 1 interface).
        coords_tgt: (N, 3) tensor, target points (e.g., protein 2 interface).
        
    Returns:
        R: (3, 3) rotation matrix.
        t: (1, 3) translation vector.
    """
    
    # 1. Input Validation
    # Ensure inputs are 2D tensors (N, 3)
    assert coords_src.dim() == 2 and coords_tgt.dim() == 2, "Inputs must be (N, 3)"
    assert coords_src.shape == coords_tgt.shape, \
        f"Shape mismatch: {coords_src.shape} vs {coords_tgt.shape}. Kabsch requires 1-to-1 point correspondence."

    # 2. Compute Centroids
    # Calculate the center of mass for both point sets
    centroid_src = torch.mean(coords_src, dim=0, keepdim=True) # (1, 3)
    centroid_tgt = torch.mean(coords_tgt, dim=0, keepdim=True) # (1, 3)

    # 3. Center the points (Remove Translation)
    # Shift points so that their centroids are at the origin
    src_centered = coords_src - centroid_src # (N, 3)
    tgt_centered = coords_tgt - centroid_tgt # (N, 3)

    # 4. Compute Covariance Matrix H
    # H = P^T * Q
    # Shape: (3, N) @ (N, 3) -> (3, 3)
    H = torch.matmul(src_centered.transpose(0, 1), tgt_centered)

    # 5. Singular Value Decomposition (SVD)
    # Decompose H into U, S, Vh (where Vh is V transpose)
    # H = U @ S @ Vh
    U, S, Vh = torch.linalg.svd(H)
    
    # Note: torch.linalg.svd returns Vh (V transpose). 
    # To get V, we need Vh.T
    V = Vh.transpose(0, 1)

    # 6. Compute Rotation Matrix R
    # R = V @ U^T
    # This gives the optimal rotation that aligns the principal axes
    R = torch.matmul(V, U.transpose(0, 1))

    # 7. Correction for Reflection
    # Check if the determinant is -1 (which implies a reflection/mirror image)
    # We want a proper rotation (det = 1), so we flip the last column of V if needed.
    if torch.linalg.det(R) < 0:
        # Create a correction matrix like diag(1, 1, -1)
        # We can implement this by flipping the last row of Vh (or last col of V)
        # Re-compute R with the corrected V
        Vh_corrected = Vh.clone()
        Vh_corrected[2, :] *= -1 # Flip the 3rd eigenvector direction
        V_corrected = Vh_corrected.transpose(0, 1)
        R = torch.matmul(V_corrected, U.transpose(0, 1))

    # 8. Compute Translation Vector t
    # The optimal translation moves the rotated source centroid to the target centroid
    # t = centroid_tgt - (R @ centroid_src.T).T
    t = centroid_tgt - torch.matmul(centroid_src, R.transpose(0, 1))

    return R, t

def rotate_and_translate(points: torch.Tensor, rot:torch.Tensor, trans:torch.Tensor):
    assert points.shape[1] == 3
    assert rot.shape == (3, 3)
    assert trans.shape == (1, 3)
    return (rot @ points.T).T + trans

def kabsch_align(Y1, Y2, mask=None):
    """
    Align Y2 -> Y1 using differentiable Kabsch.

    Args:
        Y1: (B, K, 3) target (e.g., receptor)
        Y2: (B, K, 3) source (e.g., ligand)
        mask: (B, K) optional mask for valid points

    Returns:
        R: (B, 3, 3)
        t: (B, 3)
        Y2_aligned: (B, K, 3)
    """

    B, K, _ = Y1.shape

    if mask is not None:
        mask = mask.unsqueeze(-1)  # (B, K, 1)
        Y1_mean = (Y1 * mask).sum(dim=1) / mask.sum(dim=1)
        Y2_mean = (Y2 * mask).sum(dim=1) / mask.sum(dim=1)
    else:
        Y1_mean = Y1.mean(dim=1)
        Y2_mean = Y2.mean(dim=1)

    # Center
    Y1_c = Y1 - Y1_mean.unsqueeze(1)
    Y2_c = Y2 - Y2_mean.unsqueeze(1)

    if mask is not None:
        Y1_c = Y1_c * mask
        Y2_c = Y2_c * mask

    # Covariance: source^T @ target
    A = torch.matmul(Y2_c.transpose(1, 2), Y1_c)  # (B,3,3)

    # Batch SVD
    U, S, Vt = torch.linalg.svd(A)

    # Reflection correction
    det = torch.det(torch.matmul(Vt.transpose(1,2), U.transpose(1,2)))
    corr = torch.eye(3, device=Y1.device).unsqueeze(0).repeat(B,1,1)
    corr[:, -1, -1] = det

    # Rotation
    R = torch.matmul(torch.matmul(Vt.transpose(1,2), corr), U.transpose(1,2))

    # Translation
    t = Y1_mean - torch.matmul(R, Y2_mean.unsqueeze(-1)).squeeze(-1)

    # Apply transform
    # Y2_aligned = torch.matmul(Y2, R.transpose(1,2)) + t.unsqueeze(1)

    return R, t

# def correct_ligand(prediction, rdkit_coords, lig):
#     """Correct model predict ligand atom coords.
#     Args:
#         prediction (torch.Tensor): Predicted ligand atom coords
#         rdkit_coords (np.ndarray): Ligand atom coords from rdkit
#         lig_keypts (torch.Tensor): Ligand keypoint coords
#         rec_keypts (torch.Tensor): Receptor keypoint coords
#         name (str): Complex name
#     """

#     lig_rdkit = deepcopy(lig)
#     conf = lig_rdkit.GetConformer()
#     for i in range(lig_rdkit.GetNumAtoms()):
#         x, y, z = rdkit_coords[i]
#         conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

#     lig_rdkit = RemoveHs(lig_rdkit)

#     lig = RemoveHs(lig)
#     lig_equibind = deepcopy(lig)
#     conf = lig_equibind.GetConformer()
#     for i in range(lig_equibind.GetNumAtoms()):
#         x, y, z = prediction[i]
#         conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

#     coords_pred = lig_equibind.GetConformer().GetPositions()

#     Z_pt_cloud = coords_pred
#     rotable_bonds = get_torsions([lig_rdkit])
#     new_dihedrals = np.zeros(len(rotable_bonds))
#     for idx, r in enumerate(rotable_bonds):
#         new_dihedrals[idx] = get_dihedral_vonMises(lig_rdkit, lig_rdkit.GetConformer(), r,
#                                                    Z_pt_cloud)
#     optimized_mol = apply_changes(lig_rdkit, new_dihedrals, rotable_bonds)

#     coords_pred_optimized = optimized_mol.GetConformer().GetPositions()
#     try:
#         R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_pred.T)
#     except Exception as e:
#         print(e)
#         return prediction.numpy()
#     coords_pred_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()

#     return coords_pred_optimized