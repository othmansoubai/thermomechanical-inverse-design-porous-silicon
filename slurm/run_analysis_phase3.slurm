#!/bin/bash
# ============================================================
# Phase 3 Analysis Job - runs analyze_elastic.py
#
# HOW TO SUBMIT (after the main array):
#   PHASE3_ID=$(sbatch submit_phase3.slurm | awk '{print $4}')
#   sbatch --dependency=afterok:$PHASE3_ID run_analysis_phase3.slurm
# ============================================================
#SBATCH --job-name=Si_p3_anal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=shortq
#SBATCH --output=logs/phase3_analysis.out
#SBATCH --error=logs/phase3_analysis.err

WORKDIR=$HOME/phase3_mech
cd $WORKDIR

# Reuse the Phase 2 venv - it already has numpy<2.0 + pandas
module purge
module load GCCcore/12.3.0 Python/3.11.3-GCCcore-12.3.0
source $HOME/phase2_ml/analysis_env/bin/activate

python3 analyze_elastic.py \
    --results       results \
    --jobs          jobs_phase3.csv \
    --out-perseed   elastic_perseed.csv \
    --out-aggregated elastic_dataset.csv

deactivate
echo "Analysis complete."
