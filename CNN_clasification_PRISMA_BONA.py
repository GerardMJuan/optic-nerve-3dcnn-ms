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


path_scans=''
path_scans_test=''
path_excel=''
path_excel_test=''

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

def train_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=3)
    
    return volume, label

def validation_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=3)
    
    return volume, label


def data_aug(volume, label):
    noise = np.random.normal(0,0.05,(40,31,13))
    noise=np.asarray(noise,dtype=np.float32)
    noise = tf.convert_to_tensor(noise)
    noise = tf.expand_dims(noise, axis=3)

    r = random.random()
    if r<0.5:
        volume = volume + noise

    return volume, label


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
        label_left.append(np.float(0))

    right=row[3]
    if right=='Y':
        label_right.append(np.float(1))
    else:
        label_right.append(np.float(0))

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
        label_left_test.append(np.float(0))
        n_nolesio+=1

    right_test=row[3]
    if right_test=='Y':
        label_right_test.append(np.float(1))
        n_lesio+=1
    else:
        label_right_test.append(np.float(0))
        n_nolesio+=1

print('Amount of eyes with lesions in test = ', str(n_lesio))
print('Amount of eyes with no lesions in test = ', str(n_nolesio))



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
    scans.append(process_scan(path_scans+infile+'/Eye_n4_{}_T2FastSat_crop_Left_flipped.nii'.format(infile)))
    scans.append(process_scan(path_scans+infile+'/Eye_n4_{}_T2FastSat_crop_Right.nii'.format(infile)))

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
    x_test.append(process_scan(path_scans_test+infile+'/Eye_n4_{}_T2FastSat_crop_Left_flipped.nii'.format(infile)))
    x_test.append(process_scan(path_scans_test+infile+'/Eye_n4_{}_T2FastSat_crop_Right.nii'.format(infile)))

    position=ids_excel_test.index(infile)
    y_test.append(label_left_test[position])
    y_test.append(label_right_test[position])

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

tp_sm=[]
ids_tp_sm=[]
tn_sm=[]
ids_tn_sm=[]
fp_sm=[]
ids_fp_sm=[]
fn_sm=[]
ids_fn_sm=[]

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

    #Check that data is balanced both in train and val
    train_p=0
    train_n=0
    for i in range(len(y_train)):
        if y_train[i]==1.0:
            train_p+=1
        else: 
            train_n+=1
    train_p_pctg=train_p*100/len(y_train)
    train_n_pctg=train_n*100/len(y_train)

    test_p=0
    test_n=0
    for i in range(len(y_test)):
        if y_test[i]==1.0:
            test_p+=1
        else: 
            test_n+=1
    test_p_pctg=test_p*100/len(y_test)
    test_n_pctg=test_n*100/len(y_test)

    print("Positive percentage in train:", train_p_pctg)
    print("Negative percentage in train:", train_n_pctg)
    print("Positive percentage in val:", test_p_pctg)
    print("Negative percentage in val:", test_n_pctg)


    #convert to float
    x_train=np.asarray(x_train,dtype=np.float32)
    y_train=np.asarray(y_train,dtype=np.float32)


    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # batch size can be larger, as our images are smaller
    batch_size = 16

    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .map(data_aug)
        #.map(lambda x, y: (trainAug(x), y))
        .batch(batch_size)
        .prefetch(2)
    )


    def get_model(width=40, height=31, depth=13):

        inputs = keras.Input((width, height, depth, 1))

        x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
        #x = layers.Conv3D(filters=16, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        #x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        #x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(units=128, activation="relu")(x)
        x = layers.Dropout(0.25)(x)

        outputs = layers.Dense(units=1, activation="sigmoid")(x)

        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn")
        return model

    # Build model.
    model = get_model(width=40, height=31, depth=13)
    model.summary()

    # Compile model.
    initial_learning_rate = 0.001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate ),
        metrics=["acc"],
    )

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_image_classification.h5", monitor='acc', save_best_only=True , mode='max'
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="acc", patience=300)


    # Train the model, doing validation at the end of each epoch
    epochs = 100

    history = model.fit(train_dataset,epochs=epochs,shuffle=True,verbose=1,callbacks=[checkpoint_cb])
    
    plt.plot(history.history['loss'])
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.legend(['Train loss'])#, 'Val loss', 'Train acc', 'Val acc'])
    plt.title('Metrics results')
    plt.savefig('metric_plots/plot_{}.png'.format(p))
    plt.close()

    #model.load_weights("3d_image_classification.h5")

    #####################
    ## METRICS RESULTS ##
    #####################
     
     
    tp=[]
    fp=[]
    tn=[]
    fn=[]
    loss_list=[]
    x_te=x_test
    y_te=y_test

    id_scan_tp=[]
    id_scan_tn=[]
    id_scan_fp=[]
    id_scan_fn=[]
    y_p_list=[]
    y_r_list=[]

    for i in range(len(x_te)):
        images = np.expand_dims(x_te[i], axis=-1)
        images = tf.Variable(np.expand_dims(images, axis=0))

        with tf.GradientTape() as tape:
            pred = model(images, training=False)
            # class_idws_sorted = np.argsort(pred.numpy().flatten())[::-1]
            loss = pred[0] #[class_idws_sorted[0]]
            loss_list.append(np.around(loss))

        grads = tape.gradient(loss, images)

        dgrad_abs = tf.math.abs(grads)

        arr_min,arr_max = np.min(dgrad_abs),np.max(dgrad_abs)
        grad_eval = (dgrad_abs - arr_min)/(arr_max-arr_min+1e-18) # THIS is saliency map

        y_p=loss[0]
        y_p_list.append(np.around(y_p))
        y_r=y_te[i]
        y_r_list.append(y_r)
        comparison=[y_p,y_r]
        print(comparison)
        
        if y_te[i]==np.around(loss)[0] and y_te[i]==1.0: #tp
            tp.append(grad_eval)
            id_scan_tp.append(ids_scans_test[i])
    
        elif y_te[i]==np.around(loss)[0] and y_te[i]==0.0: #tn
            tn.append(grad_eval)
            id_scan_tn.append(ids_scans_test[i])

        elif y_te[i]!=np.around(loss)[0] and y_te[i]==0.0: #fp
            fp.append(grad_eval)
            id_scan_fp.append(ids_scans_test[i])

        elif y_te[i]!=np.around(loss)[0] and y_te[i]==1.0: #fn
            fn.append(grad_eval)
            id_scan_fn.append(ids_scans_test[i])
        
    print('True positives',len(tp))
    print('True negatives', len(tn))
    print('False positives',len(fp))
    print('False negatives', len(fn))

    accu=(len(tp)+len(tn))*100/len(x_te)
    print('Accuracy ='+str(accu))
    acc_bal=sklearn.metrics.balanced_accuracy_score(y_r_list,y_p_list)
    print('Accuracy balanced='+str(acc_bal))

    # print('Saliency tp single = ' + str(id_scan_tp))
    # print('Saliency tn single = ' + str(id_scan_tn))
    # print('Saliency fp single = ' + str(id_scan_fp))
    # print('Saliency fn single = ' + str(id_scan_fn))

    accuracy.append(accu)
    accuracy_bal.append(acc_bal)
    tp_permu.append(len(tp))
    tn_permu.append(len(tn))
    fn_permu.append(len(fn))
    fp_permu.append(len(fp))

    tp_sm.append(tp)
    ids_tp_sm.append(id_scan_tp)
    tn_sm.append(tn)
    ids_tn_sm.append(id_scan_tn)
    fp_sm.append(fp)
    ids_fp_sm.append(id_scan_fp)
    fn_sm.append(fn)
    ids_fn_sm.append(id_scan_fn)

    recall_permu.append((len(tp))/(len(tp)+len(fn)))
    precisionfalse_permu.append((len(tn))/(len(tn)+len(fn)))
    precisiontrue_permu.append((len(tp))/(len(tp)+len(fp)))
    specificity_permu.append((len(tn))/(len(tn)+len(fp)))

maxvalue = max(accuracy_bal)
minvalue = min(accuracy_bal)
maxpos = accuracy_bal.index(maxvalue)

tp,tn,fp,fn=[tp_sm[maxpos],tn_sm[maxpos],fp_sm[maxpos],fn_sm[maxpos]]
id_scan_tp,id_scan_tn,id_scan_fp,id_scan_fn=[ids_tp_sm[maxpos],ids_tn_sm[maxpos],ids_fp_sm[maxpos],ids_fn_sm[maxpos]]


#delete previous scans because each time it changes
import os
import glob
#save saliency maps of individual scans
general_scans=[tp,tn,fp,fn]
general_names=['tp','tn','fp','fn']
general_ids=[id_scan_tp,id_scan_tn,id_scan_fp,id_scan_fn]
for i in range(4):
    path='saliency maps test/{}'.format(general_names[i])
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        os.remove(f)

    for j in range(len(general_ids[i])):
        id_exemple=general_ids[i][j]
        id_value=str(id_exemple[:-2])
        if id_exemple[-1]=='r':
            scan_ex = nib.load(path_scans_test+id_value+'/Eye_n4_{}_T2FastSat_crop_Right.nii'.format(id_value,))
        else:
            scan_ex = nib.load(path_scans_test+id_value+'/Eye_n4_{}_T2FastSat_crop_Left_flipped.nii'.format(id_value,))

        img=nib.Nifti1Image(general_scans[i][j].numpy().squeeze(),scan_ex.affine,scan_ex.header)
        nib.save(img,'saliency maps test/{}/{}.nii.gz'.format(general_names[i],str(general_ids[i][j])+'_smap'))

###################
## FINAL METRICS ##
###################

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
print('Accuracy  average balanced='+str(accuracy_bal_av))
print('Min value accuracy balanced = ' +str(minvalue))
print('Max value accuracy balanced = ' + str(maxvalue))
print('Recall average='+str(recall_av))
print('Recall std='+str(recall_std))
print('Precision false average ='+str(precisionfalse_av))
print('Precision false std = '+str(precisionfalse_std))
print('Precision true average = ' + str(precisiontrue_av))
print('Precision true std = ' + str(precisiontrue_std))
print('Specificity average = ' + str(specificity_av))
print('Specificity std = ' + str(specificity_std))

########################
## SAVE EXCEL METRICS ##
########################

import csv

with open('results_test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Permutation', 'Accuracy', 'Accuracy Balanced', 'Recall', 'Precision Negative', 'Precision Positive', 'Specificity'])
    for i in range(iterations):
        writer.writerow([i,accuracy[i],accuracy_bal[i], recall_permu[i],precisionfalse_permu[i],precisiontrue_permu[i],specificity_permu[i]])
