import numpy as np
import tensorflow as tf
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import os
from os import listdir
from sklearn.model_selection import train_test_split
import sklearn
from keras_preprocessing.image import ImageDataGenerator
from random import random
import matplotlib.pyplot  as plt
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

from util_functions import process_scan

path_scans='/mnt/Bessel/Gproj/Gerard_DATA/FAT-SAT/TRIO'
path_excel='/mnt/Bessel/Gproj/Gerard_DATA/FAT-SAT/new_lesion2.xlsx'

path_scans_test='/mnt/Bessel/Gproj/Gerard_DATA/FAT-SAT/PRISMA'
path_excel_test='/mnt/Bessel/Gproj/Gerard_DATA/FAT-SAT/test_labels.ods'

# type of model
model_name = "SVM" # Either SVM or RF
out_dir = f"/mnt/Bessel/Gproj/Gerard_DATA/FAT-SAT/{model_name}_results"

## BEST PARAMS (from validation)
if model_name == "SVM":
    params = {}
else:
    params = {}
###################
#### read excel ###
###################

#train
info=pd.read_excel(path_excel)
ids_excel=[]
label_right=[]
label_left=[]
for row in info.itertuples():
    ids_excel.append(str(row[1]))
    left=row[2]
    if left=='Y':
        label_left.append(np.float(1))
    else:
        label_left.append(np.float(-1))

    right=row[3]
    if right=='Y':
        label_right.append(np.float(1))
    else:
        label_right.append(np.float(-1))

#test
info=pd.read_excel(path_excel_test)
ids_excel_test=[]
label_right_test=[]
label_left_test=[]
n_lesio=0
n_nolesio=0
for row in info.itertuples():
    ids_excel_test.append(str(row[1]))
    left_test=row[2]
    if left_test=='Y':
        label_left_test.append(np.float(1))
        n_lesio+=1
    else:
        label_left_test.append(np.float(-1))
        n_nolesio+=1

    right_test=row[3]
    if right_test=='Y':
        label_right_test.append(np.float(1))
        n_lesio+=1
    else:
        label_right_test.append(np.float(-1))
        n_nolesio+=1



################
## Load scans ##
################

#train
scans=[]
ids_scans=[]
labels=[]
for infile in tqdm(listdir(path_scans)):
    ids_scans.append(infile+'_l')
    ids_scans.append(infile+'_r')
    scans.append(process_scan(f'{path_scans}/{infile}', 'n4_{}_T2FastSat_crop_Left_flipped.nii'.format(infile)))
    scans.append(process_scan(f'{path_scans}/{infile}', 'n4_{}_T2FastSat_crop_Right.nii'.format(infile)))

    position=ids_excel.index(infile)
    labels.append(label_left[position])
    labels.append(label_right[position])

print('The number of nerves loaded is: ' + str(len(scans)))

#test
x_test=[]
ids_scans_test=[]
y_test=[]
for infile in tqdm(listdir(path_scans_test)):
    ids_scans_test.append(infile+'_l')
    ids_scans_test.append(infile+'_r')
    x_test.append(process_scan(f'{path_scans_test}/{infile}', 'n4_{}_T2FastSat_crop_Left_flipped.nii'.format(infile)))
    x_test.append(process_scan(f'{path_scans_test}/{infile}', 'n4_{}_T2FastSat_crop_Right.nii'.format(infile)))

    position=ids_excel_test.index(infile)
    y_test.append(label_left_test[position])
    y_test.append(label_right_test[position])

print(y_test)

#Loop for performing different permutations and do an average of the results
accuracy=[]
accuracy_bal=[]
tp_permu=[]
tn_permu=[]
fp_permu=[]
fn_permu=[]

recall_permu=[]
precisiontrue_permu=[]
precisionfalse_permu=[]
specificity_permu=[]

iterations=200
for p in tqdm(range(iterations)):

   #SEPARATE BETWEEN POSITIVE AND NEGATIVE
    lesio=0
    nolesio=0
    x_train_lesion=[]
    y_train_lesion=[]
    ids_xtrain_lesion=[]
    ids_xtrain_nolesion=[]
    x_train_nolesion=[]
    y_train_nolesion=[]
    for i in range(len(scans)):
        if labels[i]==1.0:
            lesio+=1
            x_train_lesion.append(scans[i])
            y_train_lesion.append(labels[i])
            ids_xtrain_lesion.append(ids_scans[i])
        else:
            nolesio+=1
            x_train_nolesion.append(scans[i])
            y_train_nolesion.append(labels[i])
            ids_xtrain_nolesion.append(ids_scans[i])

    print('Number of lesion scans in training dataset', lesio)
    print('Number of no-lesion scans in training dataset', nolesio)

    #SELECT A SUBSAMPLING OF NEGATIVE CASES TO MATCH POSITIVE SIZE

    ids_xtrain_nolesion_selection=random.sample(ids_xtrain_nolesion,lesio)
    ids_xtrain_list=list(ids_scans)

    x_train_nolesion_selection=[]
    y_train_nolesion_selection=[]
    for i in range(len(ids_xtrain_nolesion_selection)):
        selection=ids_xtrain_list.index(ids_xtrain_nolesion[i])

        x_train_nolesion=np.asarray(x_train_nolesion)
        x_train_nolesion_selection.append(scans[selection])
        y_train_nolesion=np.asarray(y_train_nolesion)
        y_train_nolesion_selection.append(labels[selection])

    x_train=np.concatenate([x_train_lesion,x_train_nolesion_selection], axis=0)
    y_train=np.concatenate([y_train_lesion,y_train_nolesion_selection], axis=0)
    ids_x_train=np.concatenate([ids_xtrain_lesion,ids_xtrain_nolesion_selection], axis=0)

    
    x_train_svm=[]
    y_train_svm=[]

    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            x_train_svm.append(x_train[i][j].ravel())
            y_train_svm.append(y_train[i])

    x_test_svm=[]
    for i in range(len(x_test)):
        x_test_svm.append(x_test[i][0].ravel())

    #convert to float
    x_train_svm=np.asarray(x_train_svm,dtype=np.float32)
    x_test_svm=np.asarray(x_test_svm,dtype=np.float32)
    y_train=np.asarray(y_train_svm,dtype=np.float32)
    y_test=np.asarray(y_test,dtype=np.float32)

    print(len(x_train_svm))
    print(len(x_test_svm))

    if model_name == "SVM":
        clf = SVC()
        clf.set_params(params) # set the parameters
        clf.fit(x_train_svm, y_train)
        acc=clf.score(x_test_svm,y_test)
        prediction=clf.predict(x_test_svm)
        acc_bal=balanced_accuracy_score(y_test,prediction)
        tn,fp,fn,tp=confusion_matrix(y_test,prediction).ravel()
    else:
        clf = RandomForestClassifier()
        clf.set_params(params) # set the parameters
        clf.fit(x_train_svm, y_train)
        acc=clf.score(x_test_svm,y_test)
        prediction=clf.predict(x_test_svm)
        acc_bal=balanced_accuracy_score(y_test,prediction)
        tn,fp,fn,tp=confusion_matrix(y_test,prediction).ravel()



    #####################
    ## METRICS RESULTS ##
    #####################

    recall_permu.append(tp/(tp+fn))
    precisionfalse_permu.append(tn/(tn+fn))
    precisiontrue_permu.append(tp/(tp+fp))
    specificity_permu.append(tn/(tn+fp))

    print('Accuracy ='+str(acc))
    print('tp=',tp)
    print('tn=',tn)
    print('fp=',fp)
    print('fn=',fn)

    accuracy.append(acc)
    accuracy_bal.append(acc_bal)
    tp_permu.append(tp)
    tn_permu.append(tn)
    fp_permu.append(fp)
    fn_permu.append(fn)





accuracy_av=sum(accuracy)/iterations
acc_av_std=np.std(accuracy)
accuracy_bal_av=sum(accuracy_bal)/iterations
tp_average=sum(tp_permu)/iterations
tp_std=np.std(tp_permu)
tn_average=sum(tn_permu)/iterations
tn_std=np.std(tn_permu)
fp_average=sum(fp_permu)/iterations
fp_std=np.std(fp_permu)
fn_average=sum(fn_permu)/iterations
fn_std=np.std(fn_permu)
recall_av=sum(recall_permu)/iterations
recall_std=np.std(recall_permu)
precisionfalse_av=sum(precisionfalse_permu)/iterations
precisionfalse_std=np.std(precisionfalse_permu)
precisiontrue_av=sum(precisiontrue_permu)/iterations
precisiontrue_std=np.std(precisiontrue_permu)
specificity_av=sum(specificity_permu)/iterations
specificity_std=np.std(specificity_permu)

print('True positives',tp_average)
print('True positives std ', tp_std)
print('True negatives', tn_average)
print('True negatives std ', tn_std)
print('False positives',fp_average)
print('False positives std ', fp_std)
print('False negatives', fn_average)
print('False negatives std ', fn_std)
print('Accuracy average ='+str(accuracy_av))
print('Std of accuracies = ' + str(acc_av_std))
print('Accuracy balanced = ' + str(accuracy_bal_av))
print('Max value accuracy balanced = ' +str(np.max(accuracy_bal)))
print('Min value accuracy balanced = ' + str(np.min(accuracy_bal)))
print('Recall average='+str(recall_av))
print('Recall std='+str(recall_std))
print('Precision false average ='+str(precisionfalse_av))
print('Precision false std = '+str(precisionfalse_std))
print('Precision true average = ' + str(precisiontrue_av))
print('Precision true std = ' + str(precisiontrue_std))
print('Specificity average = ' + str(specificity_av))
print('Specificity std = ' + str(specificity_std))




