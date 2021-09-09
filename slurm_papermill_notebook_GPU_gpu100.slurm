#!/bin/bash
#SBATCH --job-name=job_papermill_notebook_GPU_gpu100
#SBATCH --output=output_papermill_notebook_GPU_gpu100
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=aglinska@bc.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=64gb
#SBATCH --partition=gpuv100

#rm -rf slurm_papermill_notebook_GPU_gpu100.slurm; nano slurm_papermill_notebook_GPU_gpu100.slurm
#sbatch slurm_papermill_notebook_GPU_gpu100.slurm

bash
module load tensorflow/2.3.1gpu
cd /data/aglinska/BC-fMRI-AE/Notebooks

notebook_name='01-train-CVAE'

now=$(date '+%Y-%m-%d-%H-%M-%S');echo $now
outname=('./papermill/papermill|'$notebook_name'|'$now);echo ${outname}

papermill $notebook_name.ipynb $outname.ipynb --autosave-cell-every 5 --progress-bar






