#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --account=pr_325_general
#SBATCH --output=/scratch/sx2490/Logs/softs/%j.out
#SBATCH --error=/scratch/sx2490/Logs/softs/%j.err
#SBATCH --job-name=patchtst_etth1

mkdir -p /scratch/sx2490/Logs/softs

# Print job information
echo "================ JOB INFORMATION ================"
echo "Running on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "===========================================" 

# Using singularity container for model training
singularity exec --nv \
    --overlay /scratch/sx2490/pytorch-example/my_pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; \
    cd /scratch/sx2490/SOFTS_exp/SOFTS_embsim/ && \
    echo 'Starting training...' && \
    bash SOFTS_ETTh1.sh || { echo 'Training script failed'; exit 1; }"

echo "================ JOB COMPLETED ================"
echo "End time: $(date)"
echo "===========================================" 

# Extract and save results to a summary file
OUTFILE="/scratch/sx2490/Logs/softs/${SLURM_JOB_ID}.out"
SUMMARY_FILE="/scratch/sx2490/SOFTS_exp/SOFTS_embsim/${SLURM_JOB_ID}.txt"

echo "Extracting results to summary file: ${SUMMARY_FILE}"
echo "JOB ID: ${SLURM_JOB_ID}" > "${SUMMARY_FILE}"
echo "Run completed at: $(date)" >> "${SUMMARY_FILE}"
echo "===========================================" >> "${SUMMARY_FILE}"

# Extract settings and results
grep -E "^.*long_term_forecast.*|^mse:" "${OUTFILE}" >> "${SUMMARY_FILE}"

echo "Summary extraction completed"

# Check job status
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Reason 