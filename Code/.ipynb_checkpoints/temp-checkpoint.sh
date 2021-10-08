module load singularity/

data_dir='data/ABIDE2/RawData/ABIDEII-BNI_1'
output_dir='data/ABIDE2/Derivatives/test/'
sub='sub-29006'

echo $data_dir
echo $output_dir
echo $sub

singularity run --cleanenv fmriprep.simg $data_dir output_dir participant --participant-label $sub