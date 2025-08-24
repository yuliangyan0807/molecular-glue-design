import os
from typing import Iterator, List, Tuple, Dict
from Bio import PDB

BASE_DIR = "./data/TernaryDB/pdbs"
PDBBIND_DIR = "./data/PDBBind/P-P"


def iter_complex_dirs(base_dir: str) -> Iterator[str]:
    for entry in sorted(os.listdir(base_dir)):
        complex_dir = os.path.join(base_dir, entry)
        if os.path.isdir(complex_dir):
            yield complex_dir


def iter_pdbbind_files(base_dir: str) -> Iterator[Tuple[str, str]]:
    """Iterate over PDBBind complex files, yielding (file_path, complex_name)."""
    for entry in sorted(os.listdir(base_dir)):
        if entry.endswith('_complex.pdb'):
            file_path = os.path.join(base_dir, entry)
            if os.path.isfile(file_path):
                complex_name = entry.replace('_complex.pdb', '')
                yield file_path, complex_name


def read_pdbbind_complex(file_path: str) -> List[str]:
    """Read a PDBBind complex file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Complex file not found: {file_path}")
    with open(file_path, "r") as f:
        return f.readlines()


def find_ter_index(lines: List[str]) -> int:
    """Find the index of the first TER record; return -1 if not found."""
    for i, line in enumerate(lines):
        if line.startswith("TER"):
            return i
    return -1


def split_pdbbind_complex(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Split complex into protein1 (before TER) and protein2 (after TER)."""
    ter_idx = find_ter_index(lines)
    
    # protein1: ATOM records before TER
    prot1_raw: List[str] = []
    for i, line in enumerate(lines):
        if ter_idx != -1 and i >= ter_idx:
            break
        if line.startswith("ATOM"):
            prot1_raw.append(line)
    
    # protein2: ATOM records after TER
    prot2_raw: List[str] = []
    if ter_idx != -1:
        for i in range(ter_idx + 1, len(lines)):
            line = lines[i]
            if line.startswith("ATOM"):
                prot2_raw.append(line)
    
    return prot1_raw, prot2_raw


def process_pdbbind_complex(file_path: str, output_dir: str) -> None:
    """Process a single PDBBind complex file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read complex file
    lines = read_pdbbind_complex(file_path)
    
    # Split into proteins
    prot1_raw, prot2_raw = split_pdbbind_complex(lines)
    
    # Renumber atom serials
    prot1 = renumber_serials(prot1_raw)
    prot2 = renumber_serials(prot2_raw)
    
    # Write output files
    with open(os.path.join(output_dir, "protein1.pdb"), "w") as f:
        f.writelines(prot1)
    with open(os.path.join(output_dir, "protein2.pdb"), "w") as f:
        f.writelines(prot2)
    
    # Move the original complex file to the output directory
    import shutil
    complex_filename = os.path.basename(file_path)
    dest_path = os.path.join(output_dir, complex_filename)
    shutil.move(file_path, dest_path)


def process_pdbbind_dataset() -> None:
    """Process all PDBBind complex files."""
    processed = 0
    failures = 0
    
    # Get all complex files before processing (since they will be moved)
    complex_files = list(iter_pdbbind_files(PDBBIND_DIR))
    
    for file_path, complex_name in complex_files:
        try:
            # Create output directory for this complex
            output_dir = os.path.join(PDBBIND_DIR, complex_name)
            process_pdbbind_complex(file_path, output_dir)
            processed += 1
            print(f"OK: {complex_name}")
        except Exception as e:
            print(f"FAIL: {complex_name} ({e})")
            failures += 1
    
    print(f"PDBBind processing done. processed={processed}, failures={failures}")


def read_gt_complex(complex_dir: str) -> List[str]:
    gt_path = os.path.join(complex_dir, "gt_complex.pdb")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"gt_complex.pdb not found in {complex_dir}")
    with open(gt_path, "r") as f:
        return f.readlines()

def find_indices(lines: List[str]) -> Tuple[int, int, int]:
    """Return (first_hetatm_idx, last_hetatm_idx, first_ter_idx); -1 if not found."""
    first_het = -1
    last_het = -1
    first_ter = -1
    for i, line in enumerate(lines):
        rec = line[0:6].strip()
        if rec == "HETATM":
            if first_het == -1:
                first_het = i
            last_het = i
        elif rec == "TER" and first_ter == -1:
            first_ter = i
    return first_het, last_het, first_ter

def renumber_serials(lines: List[str]) -> List[str]:
    """Renumber atom serial field (cols 7-11) starting from 1 for ATOM/HETATM lines."""
    out: List[str] = []
    serial = 1
    for line in lines:
        if len(line) < 6:
            continue
        rec = line[0:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        # Replace atom serial (columns 7-11)
        prefix = line[:6]
        suffix = line[11:]
        new_line = f"{prefix}{serial:5d}{suffix}"
        out.append(new_line if new_line.endswith("\n") else new_line + "\n")
        serial += 1
    return out

def split_and_write(complex_dir: str, lines: List[str]) -> None:
    first_het, last_het, first_ter = find_indices(lines)

    # protein1: ATOM before first HETATM
    prot1_raw: List[str] = []
    if first_het == -1:
        # no ligand; take all leading ATOMs as protein1 until first TER or end
        for i, line in enumerate(lines):
            rec = line[0:6].strip()
            if rec == "ATOM":
                prot1_raw.append(line)
            elif rec == "HETATM":
                break
    else:
        for i in range(first_het):
            line = lines[i]
            if line[0:6].strip() == "ATOM":
                prot1_raw.append(line)

    # ligand: from first HETATM to last HETATM (only HETATM records)
    lig_raw: List[str] = []
    if first_het != -1:
        for i in range(first_het, last_het + 1):
            line = lines[i]
            if line[0:6].strip() == "HETATM":
                lig_raw.append(line)

    # protein2: ATOM after first TER to the last ATOM
    prot2_raw: List[str] = []
    if first_ter != -1:
        for i in range(first_ter + 1, len(lines)):
            line = lines[i]
            if line[0:6].strip() == "ATOM":
                prot2_raw.append(line)

    # Renumber and write
    prot1 = renumber_serials(prot1_raw)
    lig = renumber_serials(lig_raw)
    prot2 = renumber_serials(prot2_raw)

    with open(os.path.join(complex_dir, "protein1.pdb"), "w") as f:
        f.writelines(prot1)
    with open(os.path.join(complex_dir, "ligand.pdb"), "w") as f:
        f.writelines(lig)
    with open(os.path.join(complex_dir, "protein2.pdb"), "w") as f:
        f.writelines(prot2)

def parse_seqres_and_ssbond(file_path: str) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """Parse SEQRES and SSBOND records from PDB file."""
    seqres_data = {}  # chain_id -> sequence
    ssbond_pairs = []  # list of (chain1, chain2) pairs
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('SEQRES'):
                # SEQRES format: SEQRES serial chain_id num_residues residues...
                parts = line.split()
                if len(parts) >= 4:
                    chain_id = parts[2]
                    residues = parts[4:]
                    if chain_id not in seqres_data:
                        seqres_data[chain_id] = []
                    seqres_data[chain_id].extend(residues)
            
            elif line.startswith('SSBOND'):
                # SSBOND format: SSBOND serial CYS chain1 res1 CYS chain2 res2 ...
                parts = line.split()
                if len(parts) >= 6:
                    chain1 = parts[3]
                    chain2 = parts[6]
                    if chain1 != chain2:  # Only inter-chain bonds
                        ssbond_pairs.append((chain1, chain2))
    
    return seqres_data, ssbond_pairs

def group_chains_by_sequence_and_bonds(seqres_data: Dict[str, List[str]], 
                                     ssbond_pairs: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """Group chains into protein1 and protein2 based on sequence similarity and SSBOND connections."""
    if not seqres_data:
        return [], []
    
    # Convert sequences to strings for comparison
    chain_sequences = {chain: ' '.join(seq) for chain, seq in seqres_data.items()}
    
    # Find unique sequences and their lengths
    unique_sequences = {}
    for chain, seq in chain_sequences.items():
        if seq not in unique_sequences:
            unique_sequences[seq] = []
        unique_sequences[seq].append(chain)
    
    # Strategy 1: If we have exactly 2 unique sequences, use them
    if len(unique_sequences) == 2:
        seq_list = list(unique_sequences.values())
        # Put the longer sequence as protein1, shorter as protein2
        if len(seq_list[0][0]) > len(seq_list[1][0]):
            protein1_chains = seq_list[0]
            protein2_chains = seq_list[1]
        else:
            protein1_chains = seq_list[1]
            protein2_chains = seq_list[0]
    
    # Strategy 2: If we have more than 2 unique sequences, use a more sophisticated approach
    else:
        # Sort chains by sequence length (descending)
        chains_by_length = sorted(chain_sequences.keys(), 
                                key=lambda x: len(chain_sequences[x]), reverse=True)
        
        # Start with the longest chain as protein1
        protein1_chains = [chains_by_length[0]]
        protein2_chains = []
        
        # Use SSBOND connections to determine grouping
        for chain1, chain2 in ssbond_pairs:
            if chain1 in protein1_chains and chain2 not in protein1_chains and chain2 not in protein2_chains:
                protein2_chains.append(chain2)
            elif chain2 in protein1_chains and chain1 not in protein1_chains and chain1 not in protein2_chains:
                protein2_chains.append(chain1)
        
        # Group remaining chains based on sequence similarity
        for chain in chains_by_length[1:]:
            if chain not in protein1_chains and chain not in protein2_chains:
                # Check if this chain has similar sequence to protein1
                if chain_sequences[chain] == chain_sequences[protein1_chains[0]]:
                    protein1_chains.append(chain)
                else:
                    # If no protein2 chains yet, or if this chain is similar to existing protein2 chains
                    if not protein2_chains or any(chain_sequences[chain] == chain_sequences[p2] for p2 in protein2_chains):
                        protein2_chains.append(chain)
                    else:
                        # If this is a new sequence type, add to protein2 (assuming it's part of the second protein)
                        protein2_chains.append(chain)
    
    return protein1_chains, protein2_chains

def process_pdbbind():
    """Process PDBBind complex files using Bio.PDB to split by chains based on SEQRES and SSBOND."""
    parser = PDB.PDBParser(QUIET=True)
    pdbio = PDB.PDBIO()
    processed = 0
    failures = 0
    
    for complex_dir in iter_complex_dirs(PDBBIND_DIR):
        try:
            complex_name = os.path.basename(complex_dir)
            complex_file = os.path.join(complex_dir, f"{complex_name}_complex.pdb")
            
            if not os.path.exists(complex_file):
                print(f"SKIP: {complex_name} (complex file not found)")
                continue
            
            # Parse SEQRES and SSBOND information
            seqres_data, ssbond_pairs = parse_seqres_and_ssbond(complex_file)
            
            if not seqres_data:
                print(f"SKIP: {complex_name} (no SEQRES data found)")
                continue
            
            # Group chains based on sequence and bonds
            protein1_chains, protein2_chains = group_chains_by_sequence_and_bonds(seqres_data, ssbond_pairs)
            
            if not protein1_chains or not protein2_chains:
                print(f"SKIP: {complex_name} (could not determine protein groups)")
                continue
            
            # Parse the complex structure
            structure = parser.get_structure(complex_name, complex_file)
            model = structure[0]
            
            # Create output directory
            os.makedirs(complex_dir, exist_ok=True)
            
            # Write protein1.pdb
            protein1_structure = PDB.Structure.Structure(f"{complex_name}_protein1")
            protein1_model = PDB.Model.Model(0)
            for chain_id in protein1_chains:
                if chain_id in model:
                    protein1_model.add(model[chain_id])
            protein1_structure.add(protein1_model)
            
            with open(os.path.join(complex_dir, "protein1.pdb"), "w") as f:
                pdbio.set_structure(protein1_structure)
                pdbio.save(f)
            
            # Write protein2.pdb
            protein2_structure = PDB.Structure.Structure(f"{complex_name}_protein2")
            protein2_model = PDB.Model.Model(0)
            for chain_id in protein2_chains:
                if chain_id in model:
                    protein2_model.add(model[chain_id])
            protein2_structure.add(protein2_model)
            
            with open(os.path.join(complex_dir, "protein2.pdb"), "w") as f:
                pdbio.set_structure(protein2_structure)
                pdbio.save(f)
            
            processed += 1
            print(f"OK: {complex_name} (protein1: {protein1_chains}, protein2: {protein2_chains})")
            
        except Exception as e:
            print(f"FAIL: {complex_name} ({e})")
            failures += 1
    
    print(f"PDBBind processing done. processed={processed}, failures={failures}")

def main() -> None:
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "pdbbind":
        # Process PDBBind dataset
        process_pdbbind()
    else:
        # Process TernaryDB dataset (default)
        processed = 0
        failures = 0
        for complex_dir in iter_complex_dirs(BASE_DIR):
            try:
                lines = read_gt_complex(complex_dir)
                split_and_write(complex_dir, lines)
                processed += 1
                print(f"OK: {complex_dir}")
            except Exception as e:
                print(f"FAIL: {complex_dir} ({e})")
                failures += 1
        print(f"TernaryDB processing done. processed={processed}, failures={failures}")

if __name__ == "__main__":
    main()