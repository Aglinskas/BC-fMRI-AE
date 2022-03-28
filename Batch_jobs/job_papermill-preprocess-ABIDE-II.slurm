#!/bin/bash
#SBATCH --job-name=papermill-preprocess-ABIDE-II
#SBATCH --output=ABIDE-II-preproc-jobs/output-preprocess-ABIDE-II_%a.txt
#SBATCH --error=ABIDE-II-preproc-jobs/error-preprocess-ABIDE-II_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --array=0-1000
##SBATCH --mail-user=aglinska@bc.edu
##SBATCH --mail-type=ALL

#rm -rf job_papermill-preprocess-ABIDE-I.slurm; nano job_papermill-preprocess-ABIDE-I.slurm
#sbatch job_papermill-preprocess-ABIDE-I.slurm

bash

date

cd ~/BC-fMRI-AE/Notebooks/

subID=$SLURM_ARRAY_TASK_ID

notebook_name=~/BC-fMRI-AE/Notebooks/00-ABIDE-II-compcorr-filter.ipynb
outname=~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb

echo $subID
echo $notebook_name
echo $outname

r=0
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=1
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=2
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=3
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=4
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=5
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=6
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=7
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=8
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=9
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=10
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=11
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar

r=12
papermill $notebook_name ~/BC-fMRI-AE/Code/ABIDE-II-preproc-notebooks/preproc-$subID-$r.ipynb -p s $subID -p r $r --autosave-cell-every 5 --inject-input-path --progress-bar