#!/bin/sh


# ------------ Canviar aquesta linia pel projecte en concret ----------------
# Exemple --> Projecte=MS-PATHS
Projecte=FAST_SAT

# -------------Canviar aquesta linia pel FINAL de la imatge a corregir-------
#--------------Hauria de tenir l'aspecte següent-----------------------------
#--------------sub-1-PatId-ses-Id-MPRAGE.nii.gz------------------------------
# Exemple --> Imatge=MPRAGE.nii.gz
# Imatge=t2_tse_fs_cor_384_BILAT_3mm

# -------------Canviar aquesta linia per l'extensió de la imatge a corregir--
#Exemple --> Ext=nii.gz
Ext=nii.gz

RunDir=/mnt/DADES/Gerard/RunNas
if [ ! -d $RunDir ]
then
  mkdir $RunDir
fi

export PYTHONNOUSERSITE=True

# path to the images
NiftisPostProc=

curr_dir=$PWD

LlistaPat=$(ls $NiftisPostProc)
for Pat in $LlistaPat
do

  if [ -e $NiftisPostProc/$Pat/*."$Ext" ]
  then
    gunzip $(ls $NiftisPostProc/$Pat/*."$Ext")
  fi
  
  FastSatImage=$(ls $NiftisPostProc/$Pat/*.nii)

  if [ -f $FastSatImage ] && [ ! -f $NiftisPostProc/$Pat/label_"$Pat"_crop_Left_flipped.nii.gz ]
  then

    mkdir $RunDir/$Pat
    cp $FastSatImage $RunDir/$Pat

    LocalFastSat=$(ls $RunDir/$Pat/.)

    sct_crop_image -i $RunDir/$Pat/$LocalFastSat -xmin 0 -xmax 174 -ymin 0 -ymax -1 -zmin 0 -zmax -1 -o $RunDir/$Pat/n4_"$Pat"_T2FastSat_crop_Right.nii
    sct_crop_image -i $RunDir/$Pat/$LocalFastSat -xmin 175 -xmax -1 -ymin 0 -ymax -1 -zmin 0 -zmax -1 -o $RunDir/$Pat/n4_"$Pat"_T2FastSat_crop_Left.nii

    matlab -nosplash -nodisplay -nodesktop -r "flipOneEye('$RunDir/$Pat/n4_"$Pat"_T2FastSat_crop_Right.nii', '$RunDir/$Pat/n4_"$Pat"_T2FastSat_crop_Left.nii');exit"

    sct_label_utils -i $RunDir/$Pat/n4_"$Pat"_T2FastSat_crop_Right.nii -create-viewer 3 -o $RunDir/$Pat/label_"$Pat"_crop_Right.nii.gz
    sct_label_utils -i $RunDir/$Pat/n4_"$Pat"_T2FastSat_crop_Left_flipped.nii -create-viewer 3 -o $RunDir/$Pat/label_"$Pat"_crop_Left_flipped.nii.gz

    matlab -nosplash -nodisplay -nodesktop -r "cd('$curr_dir'); CreateBoxFromOpticNerveNoEye('$RunDir/$Pat', 'n4_"$Pat"_T2FastSat_crop_Right.nii', 'label_"$Pat"_crop_Right.nii.gz'); CreateBoxFromOpticNerveNoEye('$RunDir/$Pat', 'n4_"$Pat"_T2FastSat_crop_Left_flipped.nii', 'label_"$Pat"_crop_Left_flipped.nii.gz'); exit"

    Moure=$(ls $RunDir/$Pat )
    for Fitxer in $Moure
    do
      cp -r $RunDir/$Pat/$Fitxer $NiftisPostProc/$Pat
    done

    rm -r $RunDir/$Pat

  fi

  echo $Pat is done
done
