#!/bin/bash
# ============================================================
# INV1 inverse-design candidate - E (uniaxial tensile) validation
# ------------------------------------------------------------
# Geometry INV1_AR2.5_s37  (C_d3 stagger 37.5% + AR=2.5 footprint):
#   Pore-A x[0,5] y[0,2]   Pore-B x[3,8] y[3,5]  (offset +3)
#   atom-counted phi = 19.26 %  (run prints POROSITY_PCT to confirm)
#   GP target: kappa 2.4-2.7 W/m.K,  E ~ 70-71 GPa
#
# Same workflow as submit_phase3.slurm: 3 seeds, in.uniaxial_tensile_parametric.
#
# HOW TO SUBMIT (from ~/phase3_mech):
#   # smoke test one seed first:
#   sbatch --array=1 submit_inv1_tensile.slurm
#   # check results/stress_strain_INV1_AR2.5_s37_seed12345.dat, then the rest:
#   sbatch --array=2-3 submit_inv1_tensile.slurm
# ============================================================
#SBATCH --job-name=INV1_tens
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --partition=mediumq
#SBATCH --output=logs/inv1_tens_%a.out
#SBATCH --error=logs/inv1_tens_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=o.soubai@example.com

module purge
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos

WORKDIR=$HOME/phase3_mech
cd $WORKDIR
mkdir -p logs results

# Si.sw must be in WORKDIR (same as Phase 3).

# --- INV1 candidate: fixed geometry, 3 seeds for error bars ---
LABEL="INV1_AR2.5_s37"
SEEDS=(12345 45678 78901)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID-1))]}

# Pore bounds (lattice units) - 5x2 footprint, Pore-B offset +3 (37.5% stagger)
PAXLO=0; PAXHI=5; PAYLO=0; PAYHI=2
PBXLO=3; PBXHI=8; PBYLO=3; PBYHI=5

echo "================================================"
echo "INV1 tensile  task=${SLURM_ARRAY_TASK_ID}  seed=${SEED}"
echo "Pore A (x,y): [${PAXLO},${PAXHI}] x [${PAYLO},${PAYHI}]"
echo "Pore B (x,y): [${PBXLO},${PBXHI}] x [${PBYLO},${PBYHI}]"
echo "expected phi_actual ~ 19.26 %"
echo "================================================"

mpirun -np ${SLURM_NTASKS} --mca pml ob1 --mca btl ^vader,openib lmp \
    -var seed       ${SEED}  \
    -var label      ${LABEL} \
    -var pAxlo      ${PAXLO} \
    -var pAxhi      ${PAXHI} \
    -var pAylo      ${PAYLO} \
    -var pAyhi      ${PAYHI} \
    -var pBxlo      ${PBXLO} \
    -var pBxhi      ${PBXHI} \
    -var pBylo      ${PBYLO} \
    -var pByhi      ${PBYHI} \
    -var erate      1e-4 \
    -var max_strain 0.10 \
    -in in.uniaxial_tensile_parametric \
    -log logs/lammps_${LABEL}_seed${SEED}.log

mv stress_strain_${LABEL}_seed${SEED}.dat results/ 2>/dev/null || true
mv final_${LABEL}_seed${SEED}.data        results/ 2>/dev/null || true

echo "INV1 tensile task ${SLURM_ARRAY_TASK_ID} complete: seed=${SEED}"
