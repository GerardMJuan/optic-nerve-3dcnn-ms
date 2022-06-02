function CreateBoxFromOpticNerveNoEye(Path, ImatgeCropejada, LabelDeLaImatgeCorresponent)


label = niftiread(strcat(Path, '/', LabelDeLaImatgeCorresponent));

[i, j, k]=ind2sub(size(label), find(label));

Imatge = niftiread(strcat(Path, '/', ImatgeCropejada));
OrigNifti = nifti(strcat(Path, '/', ImatgeCropejada));

OpticNerve = Imatge(i - 15 : i + 24, j - 15 : j + 15, k-2 : k+10);

niftiwrite(OpticNerve, strcat(Path, '/Eye_',ImatgeCropejada));
EyeNifti = nifti(strcat(Path, '/Eye_',ImatgeCropejada));

EyeNifti.mat = OrigNifti.mat;
EyeNifti.mat0 = OrigNifti.mat0;

create(EyeNifti);


end