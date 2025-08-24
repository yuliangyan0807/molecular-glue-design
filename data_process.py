import os
from typing import Iterator, List, Tuple

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

def main() -> None:
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "pdbbind":
        # Process PDBBind dataset
        process_pdbbind_dataset()
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