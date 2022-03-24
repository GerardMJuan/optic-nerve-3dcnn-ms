import numpy as np
import tensorflow as tf
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import os
from os import listdir
from sklearn.model_selection import train_test_split
import sklearn
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
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

path_scans=''
path_excel=''

###############
## FUNCTIONS ##
###############

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = np.amin(volume) # HARDCODED ( need to change )
    max = np.amax(volume) # HARDCODED ( need to change )
    #volume[volume < min] = min
    #volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def normalize_mean(volume):
    mean = np.average(volume)
    std = np.std(volume)
    volume = (volume - mean) / std
    volume = volume.astype("float32")
    return volume

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    #volume = normalize(volume)
    volume = normalize_mean(volume)
    # Resize width, height and depth
    # volume = resize_volume(volume)
    return volume

SVM_algorithm=True
RF_algorithm=False

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



################
## Load scans ##
################

scans_l=[]
scans_r=[]
ids_scans_l=[]
ids_scans_r=[]
labels_l=[]
labels_r=[]
groups=[]
cont=1
for infile in tqdm(listdir(path_scans)):
    ids_scans_l.append(infile+'_l')
    ids_scans_r.append(infile+'_r')
    scans_l.append(process_scan(path_scans+infile+'/Eye_n4_{}_T2FastSat_crop_Left_flipped.nii'.format(infile)))
    scans_r.append(process_scan(path_scans+infile+'/Eye_n4_{}_T2FastSat_crop_Right.nii'.format(infile)))

    position=ids_excel.index(infile)
    labels_l.append(label_left[position])
    labels_r.append(label_right[position])
    groups.append(cont)
    groups.append(cont)
    cont+=1


#################
# CNN STRUCTURE #
#################

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

    pos2_scans=[]
    pos2_labels=[]
    pos2_ids=[]
    pos_scans=[]
    pos_labels=[]
    pos_ids=[]
    neg_scans=[]
    neg_labels=[]
    neg_ids=[]

    for i in range(len(scans_l)):
        if labels_r[i]==1.0 and labels_l[i]==1.0:
            pos2_scans.append([scans_l[i],scans_r[i]])
            pos2_labels.append([labels_l[i],labels_r[i]])
            pos2_ids.append([ids_scans_l[i],ids_scans_r[i]])

        elif labels_r[i]==1.0 or labels_l[i]==1.0:
            pos_scans.append([scans_l[i],scans_r[i]])
            pos_labels.append([labels_l[i],labels_r[i]])
            pos_ids.append([ids_scans_l[i],ids_scans_r[i]])

        else:
            neg_scans.append([scans_l[i],scans_r[i]])
            neg_labels.append([labels_l[i],labels_r[i]])
            neg_ids.append([ids_scans_l[i],ids_scans_r[i]])

    '''pos2_scans=np.asarray(pos2_scans)
    pos2_labels=np.asarray(pos2_labels)
    pos2_ids=np.asarray(pos2_ids)
    pos_scans=np.asarray(pos_scans)
    pos_labels=np.asarray(pos_labels)
    pos_ids=np.asarray(pos_ids)
    neg_scans=np.asarray(neg_scans)
    neg_labels=np.asarray(neg_labels)
    neg_ids=np.asarray(neg_ids)'''

    x_train=[]
    y_train=[]
    ids_train=[]
    x_val=[]
    y_val=[]
    ids_val=[]

    #divide scanners with a Y and a N eye
    selection=list(range(68))
    selection=random.sample(selection,len(selection))
    for i in range(68):
        if i<53:
            x_train.append(pos_scans[selection[i]])
            y_train.append(pos_labels[selection[i]])
            ids_train.append(pos_ids[selection[i]])
        else:
            x_val.append(pos_scans[selection[i]])
            y_val.append(pos_labels[selection[i]])
            ids_val.append(pos_ids[selection[i]])

    print(np.shape)

    #divide scanners with a Y and a Y eye
    selection2=list(range(2))
    selection2=random.sample(selection2,len(selection2))
    train_selection2=selection2[0]
    val_selection2=selection2[-1]

    
    x_train.append(pos2_scans[train_selection2])
    y_train.append(pos2_labels[train_selection2])
    ids_train.append(pos2_ids[train_selection2])

    x_val.append(pos2_scans[val_selection2])
    y_val.append(pos2_labels[val_selection2])
    ids_val.append(pos2_ids[val_selection2])

    #divide scanners with a N and a N eye
    selection3=list(range(37))
    selection3=random.sample(selection3,len(selection3))

    for i in range(37):
        if i<28:
            x_train.append(neg_scans[selection3[i]])
            y_train.append(neg_labels[selection3[i]])
            ids_train.append(neg_ids[selection3[i]])
        else:
            x_val.append(neg_scans[selection3[i]])
            y_val.append(neg_labels[selection3[i]])
            ids_val.append(neg_ids[selection3[i]])

    x_train2=[]
    y_train2=[]
    ids_train2=[]
    for i in range(len(x_train)):
        subject=x_train[i]
        a=subject[0]
        b=subject[1]
        x_train2.append(a)
        x_train2.append(b)

        subject2=y_train[i]
        a2=subject2[0]
        b2=subject2[1]
        y_train2.append(a2)
        y_train2.append(b2)

        subject3=ids_train[i]
        a3=subject3[0]
        b3=subject3[1]
        ids_train2.append(a3)
        ids_train2.append(b3)

    x_val2=[]
    y_val2=[]
    ids_val2=[]
    for i in range(len(x_val)):
        subject=x_val[i]
        a=subject[0]
        b=subject[1]
        x_val2.append(a)
        x_val2.append(b)

        subject2=y_val[i]
        a2=subject2[0]
        b2=subject2[1]
        y_val2.append(a2)
        y_val2.append(b2)

        subject3=ids_val[i]
        a3=subject3[0]
        b3=subject3[1]
        ids_val2.append(a3)
        ids_val2.append(b3)

    x_train=x_train2
    y_train=y_train2
    ids_xtrain=ids_train2
    x_val=x_val2
    y_val=y_val2
    ids_xval=ids_val2


    #SEPARATE BETWEEN POSITIVE AND NEGATIVE
    lesio=0
    nolesio=0
    x_train_lesion=[]
    y_train_lesion=[]
    ids_xtrain_lesion=[]
    ids_xtrain_nolesion=[]
    x_train_nolesion=[]
    y_train_nolesion=[]
    for i in range(len(x_train)):
        if y_train[i]==1.0:
            lesio+=1
            x_train_lesion.append(x_train[i])
            y_train_lesion.append(y_train[i])
            ids_xtrain_lesion.append(ids_xtrain[i])
        else:
            nolesio+=1
            x_train_nolesion.append(x_train[i])
            y_train_nolesion.append(y_train[i])
            ids_xtrain_nolesion.append(ids_xtrain[i])

    print('Number of lesion scans in training dataset', lesio)
    print('Number of no-lesion scans in training dataset', nolesio)

    #SELECT A SUBSAMPLING OF NEGATIVE CASES TO MATCH POSITIVE SIZE

    ids_xtrain_nolesion_selection=random.sample(ids_xtrain_nolesion,lesio)
    ids_xtrain_list=list(ids_xtrain)

    x_train_nolesion_selection=[]
    y_train_nolesion_selection=[]
    for i in range(len(ids_xtrain_nolesion_selection)):
        selection=ids_xtrain_list.index(ids_xtrain_nolesion[i])

        x_train_nolesion=np.asarray(x_train_nolesion)
        x_train_nolesion_selection.append(x_train[selection])
        y_train_nolesion=np.asarray(y_train_nolesion)
        y_train_nolesion_selection.append(y_train[selection])

    x_train=np.concatenate([x_train_lesion,x_train_nolesion_selection], axis=0)
    y_train=np.concatenate([y_train_lesion,y_train_nolesion_selection], axis=0)
    ids_x_train=np.concatenate([ids_xtrain_lesion,ids_xtrain_nolesion_selection], axis=0)

    
    x_train_svm=[]
    for i in range(len(x_train)):
        x_train_svm.append(x_train[i].ravel())

    x_val_svm=[]
    for i in range(len(x_val)):
        x_val_svm.append(x_val[i].ravel())

    #convert to float
    x_train_svm=np.asarray(x_train_svm,dtype=np.float32)
    x_val_svm=np.asarray(x_val_svm,dtype=np.float32)
    y_train=np.asarray(y_train,dtype=np.float32)
    y_val=np.asarray(y_val,dtype=np.float32)

    print(len(x_train_svm))
    print(len(x_val_svm))

    if SVM_algorithm==True:
        clf = SVC()
        clf.fit(x_train_svm, y_train)
        acc=clf.score(x_val_svm,y_val)
        prediction=clf.predict(x_val_svm)
        acc_bal=balanced_accuracy_score(y_val,prediction)
        tn,fp,fn,tp=confusion_matrix(y_val,prediction).ravel()

    if RF_algorithm==True:
        clf = RandomForestClassifier()
        clf.fit(x_train_svm, y_train)
        acc=clf.score(x_val_svm,y_val)
        prediction=clf.predict(x_val_svm)
        acc_bal=balanced_accuracy_score(y_val,prediction)
        tn,fp,fn,tp=confusion_matrix(y_val,prediction).ravel()


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
print('Max value accuracy balanced = ' +str(np.max(accuracy)))
print('Min value accuracy balanced = ' + str(np.min(accuracy)))
print('Recall average='+str(recall_av))
print('Recall std='+str(recall_std))
print('Precision false average ='+str(precisionfalse_av))
print('Precision false std = '+str(precisionfalse_std))
print('Precision true average = ' + str(precisiontrue_av))
print('Precision true std = ' + str(precisiontrue_std))
print('Specificity average = ' + str(specificity_av))
print('Specificity std = ' + str(specificity_std))




