#!/bin/bash
#
#SBATCH --job-name=YeoCorr1k
#SBATCH --output=job_YeoCorr_output1k_%A_%a
#SBATCH --mem=1gb
#SBATCH --ntasks=1
#SBATCH --time=0:15:00
#SBATCH --mail-user=aglinska@bc.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-1000


#rm -rf job_YeoCorr1k.slurm
#nano job_YeoCorr1k.slurm
#sbatch job_YeoCorr1k.slurm

module load matlab/2020b 
echo 'YeoCorrSingleSub('$SLURM_ARRAY_TASK_ID')'
matlab -nodesktop -nodisplay -r 'YeoCorrSingleSub('$SLURM_ARRAY_TASK_ID')'
#empty line at the end

