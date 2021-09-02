#!/bin/bash
#
#SBATCH --job-name=YeoCorr
#SBATCH --output=job_YeoCorr_output
#SBATCH --mem=1gb
#SBATCH --ntasks=1
#SBATCH --time=0:15:00
#SBATCH --mail-user=aglinska@bc.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-1102



module load matlab/2020b # rm -rf job_YeoCorr.slurm;nano job_YeoCorr.slurm
echo 'YeoCorrSingleSub('$SLURM_ARRAY_TASK_ID')'
matlab -nodesktop -nodisplay -r 'YeoCorrSingleSub('$SLURM_ARRAY_TASK_ID')'
#empty line at the end
