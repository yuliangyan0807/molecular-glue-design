import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def get_complex_pdb_path() -> Path | None:
    """Return a PDB file path under the given complex directory.
    Prefers 'gt_complex.pdb' if present; otherwise picks the first *.pdb.
    """
    complex_dir = Path("./data/TernaryDB/pdbs/1NSG_A_B_RAD")
    if not complex_dir.exists() or not complex_dir.is_dir():
        print(f"Complex directory not found: {complex_dir}")
        return None

    preferred = complex_dir / "gt_complex.pdb"
    if preferred.exists():
        return preferred

    for pdb in sorted(complex_dir.glob("*.pdb")):
        return pdb

    print(f"No PDB files found under: {complex_dir}")
    return None


def visualize_with_pymol_cli(pdb_path: Path) -> None:
    """Visualize using the PyMOL CLI (no cmd API) and save a PNG."""
    pymol_bin = shutil.which("pymol")
    if pymol_bin is None:
        print(
            "PyMOL executable not found in PATH. Please install PyMOL and ensure 'pymol' is available.\n"
            "- conda install -c conda-forge pymol-open-source  (adds 'pymol' binary)\n"
            "- or install official PyMOL and add to PATH"
        )
        return

    complex_name = pdb_path.parent.name
    output_png = f"{complex_name}_visualization.png"
    overview_png = f"{complex_name}_overview.png"
    closeup_png = f"{complex_name}_ligand_closeup.png"

    # Infer chains and ligand from PDB content (robust to folder naming)
    chain_order: list[str] = []
    ligand_counts: dict[str, int] = {}
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    if len(line) >= 22:
                        chain_id = line[21].strip() or "A"
                        if chain_id not in chain_order:
                            chain_order.append(chain_id)
                elif line.startswith("HETATM"):
                    if len(line) >= 21:
                        resn = line[17:20].strip().upper()
                        if resn not in {"HOH", "WAT", "DOD"}:
                            ligand_counts[resn] = ligand_counts.get(resn, 0) + 1
    except Exception:
        pass

    chain1 = chain_order[0] if len(chain_order) >= 1 else "A"
    chain2 = chain_order[1] if len(chain_order) >= 2 else (chain_order[0] if chain_order else "B")
    ligand_resn = None
    if ligand_counts:
        ligand_resn = max(ligand_counts.items(), key=lambda kv: kv[1])[0]

    ligand_sel = (
        f"{complex_name} and resn {ligand_resn}"
        if ligand_resn and ligand_resn != "HOH"
        else f"({complex_name} and hetatm and not resn HOH and not solvent) or ({complex_name} and organic)"
    )

    # Prepare a temporary .pml script with visualization commands
    pml_commands = f"""
load {pdb_path}, {complex_name}

hide everything

# Enforce bond guessing for HET groups lacking CONECT
set connect_mode, 1
rebuild

# Background and render quality
bg_color white
set ray_opaque_background, on
set antialias, 2
set orthoscopic, on
set two_sided_lighting, on
set ambient, 0.5
set specular, 0.3
set ray_shadows, 0
set ignore_zero_occupancy, on

# Smooth, nice-looking ribbons
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set cartoon_oval_width, 0.35
set cartoon_helix_radius, 0.9
set cartoon_sampling, 14
set cartoon_transparency, 0.0
set cartoon_highlight_color, grey70
set cartoon_gap_cutoff, 4.0

# Define pastel colors
set_color pastel_salmon, [255, 160, 160]
set_color pastel_teal, [100, 200, 190]
set_color ligand_yellow, [245, 185, 0]
set_color pocket_blue, [70, 140, 255]

# Build a separate ligand object before disabling original
select lig_sel, {ligand_sel}
create {complex_name}_lig, lig_sel

# Improve ligand bonding/appearance
set valence, 1, {complex_name}_lig

# Split chains for robust coloring, then disable the original to avoid duplicates
split_chains {complex_name}
disable {complex_name}

# Show protein cartoons from all chain objects
show cartoon, all and polymer.protein
color grey70, all and polymer.protein

# Primary chain colors (best-effort; no error if chain absent)
color pastel_salmon, {complex_name}_{chain1}
color pastel_teal, {complex_name}_{chain2}

# Ligand sticks
show sticks, {complex_name}_lig
set stick_radius, 0.23, {complex_name}_lig
color ligand_yellow, {complex_name}_lig

# Pocket residues around ligand
select pocket, byres (({complex_name}_lig) around 4.0) and polymer.protein
show sticks, pocket
set stick_radius, 0.15, pocket
color pocket_blue, pocket

# Optional H-bonds/contacts (won't error if none)
dist hbonds, {complex_name}_lig, pocket, mode=2, cutoff=3.5, angle=45
hide labels, hbonds
color lightblue, hbonds

# -------- Overview (fits entire complex) --------
center all
orient all
zoom all, 6, 0, 1
png {overview_png}, ray=1, width=2200, height=1200

# -------- Ligand closeup --------
center {complex_name}_lig
zoom {complex_name}_lig, 10
png {closeup_png}, ray=1, width=1600, height=1100

quit
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pml", delete=False) as tmp_pml:
        tmp_pml.write(pml_commands)
        tmp_pml_path = Path(tmp_pml.name)

    try:
        # Run PyMOL headless (-cq) with the generated .pml script
        result = subprocess.run(
            [pymol_bin, "-cq", str(tmp_pml_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print("PyMOL exited with errors:")
            print(result.stderr)
        else:
            print(f"Image saved: {output_png}")
    finally:
        # Clean up the temporary script
        try:
            tmp_pml_path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    pdb_path = get_complex_pdb_path()
    if pdb_path is None:
        return

    print(f"Visualizing complex from: {pdb_path}")
    visualize_with_pymol_cli(pdb_path)


if __name__ == "__main__":
    main()