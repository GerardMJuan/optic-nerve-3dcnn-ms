%% AUTHOR: Aran Garcia-Vidal


function flipOneEye(RightImage, LeftImage)

% ImageName = ImageName(1 : end - 4);

OrigNifti = nifti(RightImage);
OrigImageToFlip = niftiread(LeftImage);

FlippedImage = flipud(OrigImageToFlip);

niftiwrite(FlippedImage,strcat(LeftImage(1 : end - 4),'_flipped.nii'));

FlippedNifti = nifti(strcat(LeftImage(1 : end - 4),'_flipped.nii'));
FlippedNifti.mat = OrigNifti.mat;
FlippedNifti.mat0 = OrigNifti.mat0;

create(FlippedNifti);

end