#!/bin/bash
# ============================================================
# SLURM Job Array - Paper 3 Phase 1 (51 jobs) - UPDATED
# Uniaxial Tensile MD on 17 porous Si geometries x 3 seeds.
#
# Changes from v1:
#   - --array=1-51 (was 1-54; dropped P1_low_50pct, 3 jobs removed)
#   - --time=06:00:00 (was 04:00:00; deformation MD takes longer than NEMD)
#   - --partition=mediumq (was shortq; shortq capped at 2h)
#
# HOW TO SUBMIT:
#   cd ~/phase3_mech
#   # Smoke test first (bulk_si only, 3 jobs):
#   sbatch --array=1-3 submit_phase3.slurm
#   # Then check results/ for stress_strain_bulk_si_*.dat
#   # If clean, launch the rest:
#   PHASE3_ID=$(sbatch --array=4-51 submit_phase3.slurm | awk '{print $4}')
#   sbatch --dependency=afterok:$PHASE3_ID run_analysis_phase3.slurm
# ============================================================
#SBATCH --job-name=Si_phase3
#SBATCH --array=1-51
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --partition=mediumq
#SBATCH --output=logs/phase3_%a.out
#SBATCH --error=logs/phase3_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=o.soubai@example.com

# --- Environment --------------------------------------------
module purge
module load LAMMPS/2Aug2023_update2-foss-2023a-kokkos

WORKDIR=$HOME/phase3_mech
cd $WORKDIR
mkdir -p logs results

# Si.sw must be in WORKDIR (one-time copy from Phase 2):
#   cp ~/phase2_ml/Si.sw ~/phase3_mech/

# --- Read row for this array task ---------------------------
# CSV columns: job_id,label,group,seed,pAxlo,pAxhi,pAylo,pAyhi,pBxlo,pBxhi,pBylo,pByhi,phi_actual
CSV="$WORKDIR/jobs_phase3.csv"

ROW=$(grep -v '^#' $CSV | awk -F',' -v id=${SLURM_ARRAY_TASK_ID} 'NR>1 && NF>1 {n++; if(n==id) print}')

LABEL=$(echo $ROW | cut -d',' -f2)
GROUP=$(echo $ROW | cut -d',' -f3)
SEED=$(echo  $ROW | cut -d',' -f4)
PAXLO=$(echo $ROW | cut -d',' -f5)
PAXHI=$(echo $ROW | cut -d',' -f6)
PAYLO=$(echo $ROW | cut -d',' -f7)
PAYHI=$(echo $ROW | cut -d',' -f8)
PBXLO=$(echo $ROW | cut -d',' -f9)
PBXHI=$(echo $ROW | cut -d',' -f10)
PBYLO=$(echo $ROW | cut -d',' -f11)
PBYHI=$(echo $ROW | cut -d',' -f12)
PHI=$(echo   $ROW | cut -d',' -f13)

echo "================================================"
echo "SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID}"
echo "Label / group / seed: ${LABEL} / ${GROUP} / ${SEED}"
echo "Pore A (x,y) lat. u.: [${PAXLO},${PAXHI}] x [${PAYLO},${PAYHI}]"
echo "Pore B (x,y) lat. u.: [${PBXLO},${PBXHI}] x [${PBYLO},${PBYHI}]"
echo "phi_actual          : ${PHI} %"
echo "================================================"

# --- Run LAMMPS ---------------------------------------------
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

# --- Move outputs to results/ -------------------------------
mv stress_strain_${LABEL}_seed${SEED}.dat results/ 2>/dev/null || true
mv final_${LABEL}_seed${SEED}.data        results/ 2>/dev/null || true

echo "Job ${SLURM_ARRAY_TASK_ID} complete: ${LABEL} seed=${SEED}"
