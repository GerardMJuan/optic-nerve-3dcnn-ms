%% AUTHOR: Aran Garcia-Vidal

function CreateBoxFromOpticNerveNoEye(Path, ImatgeCropejada, LabelDeLaImatgeCorresponent)

label = niftiread(strcat(Path, '/', LabelDeLaImatgeCorresponent));

[i, j, k]=ind2sub(size(label), find(label));
% create permutation adding and substracting the index around the image. 
indexes = [i j k;
           i+1 j k;
           i-1 j k;
           i j+1 k;
           i j-1 k;
           i+1 j-1 k;
           i-1 j+1 k;
           i+1 j+1 k;
           i-1 j-1 k;]

Imatge = niftiread(strcat(Path, '/', ImatgeCropejada));
OrigNifti = nifti(strcat(Path, '/', ImatgeCropejada));

% create various optic nerves around different points of the image
for x=1:size(indexes,1)
    i = indexes(x,1); 
    j = indexes(x,2);
    k = indexes(x,3);

    OpticNerve = Imatge(i - 15 : i + 24, j - 15 : j + 15, k-2 : k+10);

    niftiwrite(OpticNerve, strcat(Path, '/Eye_', int2str(x), '_', ImatgeCropejada));
    EyeNifti = nifti(strcat(Path, '/Eye_', int2str(x), '_', ImatgeCropejada));

    EyeNifti.mat = OrigNifti.mat;
    EyeNifti.mat0 = OrigNifti.mat0;

    create(EyeNifti);
end