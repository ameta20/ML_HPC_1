#!/bin/bash
#SBATCH --job-name=job1  # Job name appearing in the queue
#SBATCH --time=0-00:30:00            # <<< SET TIME LIMIT (D-HH:MM:SS) - e.g., 8 hours
#SBATCH --nodes=1                    # Number of nodes for this script (usually 1)
#SBATCH --ntasks=1                   # Number of tasks (usually 1 for a script)
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --partition=batch
#SBATCH --qos=normal                 # Quality of Service
#SBATCH --output=ml_hpc_1_%j.log # Combined output and error log file (%j = Job ID)
#SBATCH --error=ml_hpc_1_%j.log  # Send errors to the same file as output

################################################################################
#                            SETUP ENVIRONMENT                                 #
################################################################################
# --- Load necessary modules ---
module load bio/Seaborn/0.13.2-gfbf-2023b
module load vis/matplotlib/3.8.2-gfbf-2023b
module load lang/SciPy-bundle/2023.11-gfbf-2023b
module load data/scikit-learn/1.4.0-gfbf-2023b



################################################################################
#                            EXECUTE THE SCRIPT                              #
################################################################################

python test_linear_regression.py
# Check the exit status of the script
SCRIPT_EXIT_CODE=$?
if [ $SCRIPT_EXIT_CODE -eq 0 ]; then
    echo "Script finished successfully."
else
    echo "ERROR: Script finished with exit code $SCRIPT_EXIT_CODE."
fi


