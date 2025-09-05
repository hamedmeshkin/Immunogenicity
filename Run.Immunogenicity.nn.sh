#!/bin/bash
#SBATCH --account=CDERID0047
#SBATCH -J Immunogenicity        # Job name
#SBATCH -o logfiles/Immunogenicity_%j.out         # Standard output and error log (%x = job name, %j = job ID)
#SBATCH -e logfiles/Immunogenicity_%j.err         # Error log
#SBATCH --nodes=1                # Request N nodes
#SBATCH --cpus-per-task=12   	 # Use 100 CPU cores per task
#SBATCH --mem=20GB          	 # Adjust memory as needed
#SBATCH --time=10:00:00     	 # Time limit (10 hours)
####SBATCH --exclude=bc002
###SBATCH --constraint=gpu_cc_80
#SBATCH --constraint=gpu_mem_80
###SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Print job details for debugging
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

echo check for gpu: nvidia-smi output:
nvidia-smi
echo

# Get start of job information
START_TIME=`date +%s`

CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


#source /projects01/mikem/Python-3.11/set_env.sh
#export PYTHONPATH=/projects01/dars-comp-bio/anaconda3/lib/python3.11/site-packages/:$PYTHONPATH
#LD_LIBRARY_PATH=/home/mikem/lib:$LD_LIBRARY_PATH

export PATH="/projects01/dars-comp-bio/miniconda3/bin:$PATH"


python3.13 main.py --output $SLURM_ARRAY_TASK_ID --input ver1_1.0% >& logfiles/"$SLURM_JOB_ID".o.txt


# Get end of job information
EXIT_STATUS=$?
END_TIME=`date +%s`

