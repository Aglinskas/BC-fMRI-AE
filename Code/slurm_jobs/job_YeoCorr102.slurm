#!/bin/bash
#
#SBATCH --job-name=YeoCorr102
#SBATCH --output=job_YeoCorr_output102_%A_%a
#SBATCH --mem=1gb
#SBATCH --ntasks=1
#SBATCH --time=0:15:00
#SBATCH --mail-user=aglinska@bc.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-102

#rm -rf job_YeoCorr102.slurm
#nano job_YeoCorr102.slurm
#sbatch job_YeoCorr102.slurm

module load matlab/2020b
echo 'YeoCorrSingleSub('$(($SLURM_ARRAY_TASK_ID+1000))')'
matlab -nodesktop -nodisplay -r 'YeoCorrSingleSub('$(($SLURM_ARRAY_TASK_ID+1000))')'
#empty line at the end


