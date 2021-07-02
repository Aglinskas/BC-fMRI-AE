#!/bin/bash

#SBATCH --job-name=job_AWS_download_ABIDE
#SBATCH --output=job_AWS_download_ABIDE

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
##SBATCH --mem-per-cpu=16gb
##SBATCH --partition=PostP

#SBATCH --mail-user=aglinska@bc.edu
#SBATCH --mail-type=ALL

##SBATCH --array=0-7

#rm -rf job_AWS_download_ABIDE.slurm
#nano job_AWS_download_ABIDE.slurm
#sbatch job_AWS_download_ABIDE.slurm

bash 
cat "filler text" > AWS_job_running.txt

aws s3 sync s3://fcp-indi/data/Projects/ABIDE ./data/ABIDE --no-sign-request

cat "filler text" > AWS_job_finished.txt

#Empty line at the end







