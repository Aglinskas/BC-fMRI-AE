#!/bin/bash
#SBATCH --job-name=job_papermill-train-CVAE
#SBATCH --output=slurm_papermill-train-CVAE_output.txt
#SBATCH --error=slurm_papermill-train-CVAE_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-user=aglinska@bc.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=64gb
#SBATCH --partition=anzellos


cd ~/BC-fMRI-AE/Notebooks

notebook_name='009-train-CVAE-cmats'

now=$(date '+%Y-%m-%d-%H-%M-%S');echo $now
outname=('./papermill/papermill|'$notebook_name'|'$now);echo ${outname}

papermill $notebook_name.ipynb $outname.ipynb --autosave-cell-every 5 --progress-bar




