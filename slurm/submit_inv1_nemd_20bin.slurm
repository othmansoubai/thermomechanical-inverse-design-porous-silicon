#!/bin/bash
# ============================================================
# INV1 - kappa re-derivation with the CORRECTED 20-bin temperature profile
# ------------------------------------------------------------
# Identical to submit_inv1_nemd.slurm in every respect except the input deck:
# in.nemd_inv1 adds "units box" to the chunk/atom binning, giving 20 bins along
# z instead of the 4 produced by the original input. Same geometry, same
# potential, same durations, same three seeds.
#
# Purpose: test whether the coarse binning biases the reported conductivity.
# If kappa comes back within seed scatter of 2.36 +/- 0.18 W/m.K, the
# four-bin limitation discussed in the manuscript does not affect the result.
#
# Output is labelled INV1_AR2.5_s37_20bin so the original files are untouched.
#
# HOW TO SUBMIT (from ~/phase3_mech):
#   sbatch --array=1 submit_inv1_nemd_20bin.slurm     # one seed first
#   # check logs/lammps_nemd_INV1_AR2.5_s37_20bin_seed12345.log:
#   #   ">>> ACTUAL POROSITY: 19.26 %" and a 20-row temperature profile
#   sbatch --array=2-3 submit_inv1_nemd_20bin.slurm   # then the rest
# ============================================================
#SBATCH --job-name=INV1_20bin
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=08:00:00
#SBATCH --partition=mediumq
#SBATCH --output=logs/inv1_20bin_%a.out
#SBATCH --error=logs/inv1_20bin_%a.err

module purge
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos

WORKDIR=$HOME/phase3_mech
cd $WORKDIR
mkdir -p logs results

# Corrected input: same as the Phase 2 deck plus "units box" on the binning.
NEMD_IN=$WORKDIR/in.nemd_inv1

# --- INV1 candidate: fixed geometry, same 3 seeds as the original run ---
LABEL="INV1_AR2.5_s37_20bin"
SEEDS=(12345 45678 78901)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID-1))]}

# Pore bounds (lattice units) - 5x2 footprint, Pore-B offset +3 (37.5% stagger)
PAXLO=0; PAXHI=5; PAYLO=0; PAYHI=2
PBXLO=3; PBXHI=8; PBYLO=3; PBYHI=5

echo "================================================"
echo "INV1 NEMD 20-bin  task=${SLURM_ARRAY_TASK_ID}  seed=${SEED}"
echo "input: ${NEMD_IN}"
echo "Pore A (x,y): [${PAXLO},${PAXHI}] x [${PAYLO},${PAYHI}]"
echo "Pore B (x,y): [${PBXLO},${PBXHI}] x [${PBYLO},${PBYHI}]"
echo "expected phi_actual ~ 19.26 %"
echo "original 4-bin result: kappa = 2.36 +/- 0.18 W/m.K"
echo "================================================"

mpirun -np ${SLURM_NTASKS} --mca pml ob1 --mca btl ^vader,openib lmp \
    -var seed   ${SEED}  \
    -var label  ${LABEL} \
    -var pAxlo  ${PAXLO} -var pAxhi ${PAXHI} -var pAylo ${PAYLO} -var pAyhi ${PAYHI} \
    -var pBxlo  ${PBXLO} -var pBxhi ${PBXHI} -var pBylo ${PBYLO} -var pByhi ${PBYHI} \
    -in ${NEMD_IN} \
    -log logs/lammps_nemd_${LABEL}_seed${SEED}.log

mv temp_profile_${LABEL}_seed${SEED}.dat results/ 2>/dev/null || true
mv e_transfer_${LABEL}_seed${SEED}.dat    results/ 2>/dev/null || true

echo "INV1 20-bin task ${SLURM_ARRAY_TASK_ID} complete: seed=${SEED}"
