#!/bin/bash
#SBATCH --job-name=job_papermill_notebook_GPU_gpu100
#SBATCH --output=output_papermill_notebook_GPU_gpu100
#SBATCH --error=error_papermill_notebook_GPU_gpu100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mail-user=aglinska@bc.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=64gb
#SBATCH --partition=gpuv100


bash
module load tensorflow/2.3.1gpu
cd ~/BC-fMRI-AE/Notebooks

notebook_name='009-train-CVAE-cmats.ipynb'
analysis_name='Sep2-DIM_16_16-beta-1e-6-gamma-1-filt64-v5'

now=$(date '+%Y-%m-%d-%H-%M-%S');echo $now
#outname=('./papermill/papermill|'$notebook_name'|'$now);echo ${outname}

mkdir -p ~/BC-fMRI-AE/Assets/tf_weights/${analysis_name} 

outname=~/BC-fMRI-AE/Assets/tf_weights/${analysis_name}/CVAE_train_$now.ipynb

papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar \
-p analysis_name ${analysis_name} \
-p latent_dim1 16 \
-p latent_dim2 16 \
-p gamma 100 \
-p beta 1e-6 \
-p batch_size 32 \
-p kernel_size 3 \
-p filters 64 \
-p intermediate_dim 128 \
-p nlayers 3 \
-p strides 2 \
-p learning_rate 0.001 \



