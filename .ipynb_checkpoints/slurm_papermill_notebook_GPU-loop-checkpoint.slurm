#!/bin/bash
#SBATCH --job-name=job_papermill_notebook_GPU-sfari-test-new
#SBATCH --output=output_papermill_notebook_GPU-sfari-test-new
#SBATCH --error=error_papermill_notebook_GPU-sfari-test-new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-user=aglinska@bc.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=64gb
#SBATCH --partition=anzellos


## anzellos  gpuv100

bash
module load tensorflow/2.3.1gpu
cd ~/BC-fMRI-AE/Notebooks

notebook_name='0091-train-CVAE-cmats-with-SFARI.ipynb'

partition="az-new"
gpu=0

for run in {1..10}
do
    analysis_name="with-SFARI-$partition-$gpu-run$run"
    
    now=$(date '+%Y-%m-%d-%H-%M-%S');echo $now
    
    mkdir -p ~/BC-fMRI-AE/Assets/tf_weights/${analysis_name} 
    outname=~/BC-fMRI-AE/Assets/tf_weights/${analysis_name}/CVAE_train_$now.ipynb

    papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar \
    -p analysis_name ${analysis_name} \
    -p latent_dim1 64 \
    -p latent_dim2 16 \
    -p gamma 100 \
    -p beta 1e-6 \
    -p batch_size 64 \
    -p kernel_size 3 \
    -p filters 8 \
    -p intermediate_dim 128 \
    -p nlayers 4 \
    -p strides 2 \
    -p learning_rate 0.001 \
    -p gpu_id $gpu
done


