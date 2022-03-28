srun --job-name=AidasCPUInteractiveJob --nodes=1 --ntasks-per-node=8 --time=24:00:00 --mem=64gb --pty bash -I

module load matlab/2020b

matlab -nodesktop -nodisplay

addpath(genpath('./CoSMoMVPA'))


folders = dir('./data/ABIDE/Derivatives/cpac_nofilt_global/');
folders = {folders.name}';
folders = folders(5:end);


%sub = 'sub-Caltech0051456';
n = length(folders);
for i = 1:n

	if exist(ofn)==0
		sub = folders{i};

		disp(sprintf('%d/%d | %s',i,n,datestr(datetime)));

		deriv = 'preproc';
		fn_temp = './data/ABIDE/Derivatives/cpac_nofilt_global/%s/ses-1/func/%s_ses-1_%s.nii.gz';
		ofn = sprintf(fn_temp,sub,sub,'YeoCorr');

		epi_fn = cosmo_fmri_dataset(sprintf(fn_temp,sub,sub,'preproc'));
		mask_fn = cosmo_fmri_dataset(sprintf(fn_temp,sub,sub,'mask'));

		rois = cosmo_fmri_dataset('./BC-fMRI-AE/Data/epi_rois.nii.gz','mask',mask_fn);
		epi = cosmo_fmri_dataset(epi_fn,'mask',mask_fn);


		roi_arr = arrayfun(@(r) mean(epi.samples(:,rois.samples==r),2),1:51,'UniformOutput',0);
		roi_arr = cell2mat(roi_arr)';
		%size(corr(epi.samples,roi_arr')')

		cmat = epi;
		cmat.samples = corr(epi.samples,roi_arr')';
		cosmo_map2fmri(cmat,ofn);
	end

end







