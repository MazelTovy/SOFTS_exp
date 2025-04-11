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
#SBATCH --job-name=softs_mask_etth1

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
    cd /scratch/sx2490/SOFTS_exp/SOFTS_mask/ && \
    echo 'Starting training...' && \
    bash SOFTS_ETTh1.sh || { echo 'Training script failed'; exit 1; }"

echo "================ JOB COMPLETED ================"
echo "End time: $(date)"
echo "===========================================" 

# Log file path
OUTFILE="/scratch/sx2490/Logs/softs/${SLURM_JOB_ID}.out"
SUMMARY_FILE="/scratch/sx2490/SOFTS_exp/SOFTS_mask/${SLURM_JOB_ID}_summary.txt"

# Create clean tmp file without Namespace lines
TMP_FILE="/scratch/sx2490/Logs/softs/${SLURM_JOB_ID}_cleaned.out"
sed '/^Namespace/d' "${OUTFILE}" > "${TMP_FILE}"

echo "Extracting results to summary file: ${SUMMARY_FILE}"
echo "JOB ID: ${SLURM_JOB_ID}" > "${SUMMARY_FILE}"
echo "Run completed at: $(date)" >> "${SUMMARY_FILE}"
echo "===========================================" >> "${SUMMARY_FILE}"

# Extract and summarize results
{
  echo "============= RESULTS SUMMARY ============="
  echo "Dataset | seq_len | pred_len | Orig Loss | Penalty Loss | % Diff | MSE | MAE"
  echo "---------------------------------------------------------------------"
  
  # Extract dataset, seq_len, pred_len and loss information
  grep -A 2 "===== Training Loss Summary =====" "${TMP_FILE}" | grep -E "Dataset:|Avg Original Loss:|Avg Penalty Loss:|Avg Difference:" | while read -r line; do
    if [[ $line == *"Dataset:"* ]]; then
      dataset=$(echo $line | sed -E 's/.*Dataset: ([^,]+), seq_len: ([^,]+), pred_len: ([^,]+)/\1/')
      seq_len=$(echo $line | sed -E 's/.*Dataset: ([^,]+), seq_len: ([^,]+), pred_len: ([^,]+)/\2/')
      pred_len=$(echo $line | sed -E 's/.*Dataset: ([^,]+), seq_len: ([^,]+), pred_len: ([^,]+)/\3/')
      echo -n "$dataset | $seq_len | $pred_len | "
    elif [[ $line == *"Avg Original Loss:"* ]]; then
      orig_loss=$(echo $line | sed -E 's/.*Avg Original Loss: ([0-9.]+)/\1/')
      echo -n "$orig_loss | "
    elif [[ $line == *"Avg Penalty Loss:"* ]]; then
      penalty_loss=$(echo $line | sed -E 's/.*Avg Penalty Loss: ([0-9.]+)/\1/')
      echo -n "$penalty_loss | "
    elif [[ $line == *"Avg Difference:"* ]]; then
      diff_pct=$(echo $line | sed -E 's/.*Avg Difference: [0-9.]+ \(([0-9.]+)%\)/\1/')
      echo -n "$diff_pct | "
    fi
  done
  
  # Extract MSE and MAE from final test results
  test_mse=$(grep "mse:" "${TMP_FILE}" | tail -1 | sed -E 's/.*mse:([0-9.]+), mae:.*/\1/')
  test_mae=$(grep "mse:" "${TMP_FILE}" | tail -1 | sed -E 's/.*mae:([0-9.]+).*/\1/')
  echo "$test_mse | $test_mae"
  
  echo "---------------------------------------------------------------------"
  echo "Note: All losses are averages across training epochs"
} >> "${SUMMARY_FILE}"

# Add more detailed information
{
  echo -e "\n\n============= DETAILED TRAINING LOGS ============="
  # Extract all epoch summaries
  grep -E "Epoch Losses|Epoch: [0-9]+, Steps: [0-9]+ \|" "${TMP_FILE}" >> "${SUMMARY_FILE}"
  
  echo -e "\n============= MASK COMPARISON DETAILS ============="
  # Extract mask comparison details
  grep -A 6 "Mask Loss Comparison" "${TMP_FILE}" >> "${SUMMARY_FILE}"
  
  echo -e "\n============= FINAL TEST RESULTS ============="
  grep "mse:" "${TMP_FILE}" | tail -5 >> "${SUMMARY_FILE}"
} >> "${SUMMARY_FILE}"

echo "Summary extraction completed. Results saved to ${SUMMARY_FILE}"

# Cleanup temp file
rm "${TMP_FILE}"

# Check job status
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Reason 