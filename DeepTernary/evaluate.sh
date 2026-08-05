#!/usr/bin/env bash
# Batch MGD inference over all complexes under MGD_test (one subfolder = one complex).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MGD_ROOT="/home/yuliangyan/Code/Trust-App-AI-Lab/molecular_glue_design/data/TernaryDB/MGD_test"
RESULTS_ROOT="${SCRIPT_DIR}/results/MGD_test"

DEVICE="${DEVICE:-cpu}"

shopt -s nullglob
entries=("${MGD_ROOT}"/*/)
shopt -u nullglob

if [[ ${#entries[@]} -eq 0 ]]; then
  echo "No subdirectories under: ${MGD_ROOT}" >&2
  exit 1
fi

echo "Found ${#entries[@]} entries under ${MGD_ROOT}"
ok=0
skip=0
fail=0

for dir in "${entries[@]}"; do
  [[ -d "${dir}" ]] || continue
  name="$(basename "${dir}")"

  lig="${dir}/ligand.pdb"
  if [[ ! -f "${lig}" ]]; then
    lig="${dir}/ligand.sdf"
  fi
  p1="${dir}/protein1.pdb"
  p2="${dir}/protein2.pdb"
  gt="${dir}/gt_complex.pdb"

  if [[ ! -f "${p1}" || ! -f "${p2}" || ! -f "${gt}" ]]; then
    echo "SKIP ${name}: missing protein1.pdb, protein2.pdb, and/or gt_complex.pdb"
    ((skip++)) || true
    continue
  fi
  if [[ ! -f "${lig}" ]]; then
    echo "SKIP ${name}: no ligand.pdb or ligand.sdf"
    ((skip++)) || true
    continue
  fi

  outdir="${RESULTS_ROOT}/${name}"
  mkdir -p "${outdir}"

  echo "=== [${ok}+${skip}+${fail}+1/${#entries[@]}] ${name} ==="
  if python predict.py \
    --device "${DEVICE}" \
    --task MGD \
    --name "${name}" \
    --lig "${lig}" \
    --p1 "${p1}" \
    --p2 "${p2}" \
    --complex_gt_path "${gt}" \
    --outdir "${outdir}"; then
    ((ok++)) || true
  else
    echo "FAILED: ${name}" >&2
    ((fail++)) || true
  fi
done

echo "Done. ok=${ok} skip=${skip} fail=${fail} results under ${RESULTS_ROOT}"
