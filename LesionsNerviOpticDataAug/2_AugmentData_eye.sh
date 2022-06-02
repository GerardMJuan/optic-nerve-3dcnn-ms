#!/bin/sh
## This script assumes that it has already been run for a set of patients,
# and it just needs to run the modificed
# CreateBoxFromOpticNerveNoEye 
# to create the data augmentation scans.

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

RunDir=/home/extop/GERARD/RunNas
if [ ! -d $RunDir ]
then
  mkdir $RunDir
fi

export PYTHONNOUSERSITE=True

# path to the images
NiftisPostProc=/mnt/Bessel/Gproj/Gerard_DATA/FAT-SAT/PRISMA

curr_dir=$PWD

LlistaPat=$(ls $NiftisPostProc)
for Pat in $LlistaPat
do
  Pat=11747113
  # If it is done 
  if [ -f $NiftisPostProc/$Pat/label_"$Pat"_crop_Left_flipped.nii.gz ]
  then

    # Move to running dir
    mkdir $RunDir/$Pat
    Moure=$(ls $NiftisPostProc/$Pat/*.nii.gz)
    for Fitxer in $Moure
    do
      cp -r $NiftisPostProc/$Pat $RunDir
    done

    # create various optic nerves for each eye. Pass without the .nii extension
    matlab -nosplash -nodisplay -nodesktop -r "cd('$curr_dir'); CreateBoxFromOpticNerveNoEye('$RunDir/$Pat', 'n4_"$Pat"_T2FastSat_crop_Right.nii', 'label_"$Pat"_crop_Right.nii.gz'); CreateBoxFromOpticNerveNoEye('$RunDir/$Pat', 'n4_"$Pat"_T2FastSat_crop_Left_flipped.nii', 'label_"$Pat"_crop_Left_flipped.nii.gz'); settings; exit(0)"

    # Move back to actual directory
    Moure=$(ls $RunDir/$Pat )
    for Fitxer in $Moure
    do
      cp -r $RunDir/$Pat/$Fitxer $NiftisPostProc/$Pat
    done

    rm -r $RunDir/$Pat
    break
  fi
  echo $Pat is done
done
