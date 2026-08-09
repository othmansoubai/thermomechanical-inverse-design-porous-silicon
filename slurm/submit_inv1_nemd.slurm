#!/bin/bash
# ============================================================
# INV1 inverse-design candidate - kappa (NEMD Müller-Plathe) validation
# ------------------------------------------------------------
# Geometry INV1_AR2.5_s37  (C_d3 stagger 37.5% + AR=2.5 footprint):
#   Pore-A x[0,5] y[0,2]   Pore-B x[3,8] y[3,5]  (offset +3)
#   atom-counted phi = 19.26 %  (run prints ACTUAL POROSITY to confirm)
#   GP target: kappa 2.4-2.7 W/m.K,  E ~ 70-71 GPa
#
# Uses the Phase 2 NEMD input (lives in ~/phase2_ml, referenced by abs path).
# Same 3 seeds as the tensile job for matched error bars.
#
# HOW TO SUBMIT (from ~/phase3_mech):
#   sbatch --array=1 submit_inv1_nemd.slurm        # one seed first
#   # confirm logs/lammps_nemd_INV1_AR2.5_s37_seed12345.log shows
#   #   ">>> ACTUAL POROSITY: 19.26 %"  and a clean temp profile, then:
#   sbatch --array=2-3 submit_inv1_nemd.slurm
# ============================================================
#SBATCH --job-name=INV1_nemd
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=08:00:00
#SBATCH --partition=mediumq
#SBATCH --output=logs/inv1_nemd_%a.out
#SBATCH --error=logs/inv1_nemd_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=o.soubai@example.com

module purge
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos

WORKDIR=$HOME/phase3_mech
cd $WORKDIR
mkdir -p logs results

# NEMD input lives in phase2_ml; Si.sw is read from CWD (phase3_mech has it).
# If you prefer a local copy:  cp ~/phase2_ml/in.nemd_porous_parametric ~/phase3_mech/
NEMD_IN=$HOME/phase2_ml/in.nemd_porous_parametric

# --- INV1 candidate: fixed geometry, 3 seeds ---
LABEL="INV1_AR2.5_s37"
SEEDS=(12345 45678 78901)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID-1))]}

# Pore bounds (lattice units) - 5x2 footprint, Pore-B offset +3 (37.5% stagger)
PAXLO=0; PAXHI=5; PAYLO=0; PAYHI=2
PBXLO=3; PBXHI=8; PBYLO=3; PBYHI=5

echo "================================================"
echo "INV1 NEMD  task=${SLURM_ARRAY_TASK_ID}  seed=${SEED}"
echo "Pore A (x,y): [${PAXLO},${PAXHI}] x [${PAYLO},${PAYHI}]"
echo "Pore B (x,y): [${PBXLO},${PBXHI}] x [${PBYLO},${PBYHI}]"
echo "expected phi_actual ~ 19.26 %"
echo "================================================"

mpirun -np ${SLURM_NTASKS} --mca pml ob1 --mca btl ^vader,openib lmp \
    -var seed   ${SEED}  \
    -var label  ${LABEL} \
    -var pAxlo  ${PAXLO} -var pAxhi ${PAXHI} -var pAylo ${PAYLO} -var pAyhi ${PAYHI} \
    -var pBxlo  ${PBXLO} -var pBxhi ${PBXHI} -var pBylo ${PBYLO} -var pByhi ${PBYHI} \
    -in ${NEMD_IN} \
    -log logs/lammps_nemd_${LABEL}_seed${SEED}.log

# NEMD outputs (written to CWD) -> results/
mv temp_profile_${LABEL}_seed${SEED}.dat results/ 2>/dev/null || true
mv e_transfer_${LABEL}_seed${SEED}.dat    results/ 2>/dev/null || true

echo "INV1 NEMD task ${SLURM_ARRAY_TASK_ID} complete: seed=${SEED}"
