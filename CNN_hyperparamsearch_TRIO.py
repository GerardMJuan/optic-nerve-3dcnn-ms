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
from keras_preprocessing.image import ImageDataGenerator
import glob
from random import random
import matplotlib.pyplot  as plt
import random
from itertools import product

from util_functions import process_scan

path_scans='/mnt/Bessel/Gproj/Gerard_DATA/FAT-SAT/TRIO'
path_excel='/mnt/Bessel/Gproj/Gerard_DATA/FAT-SAT/new_lesion2.xlsx'
out_dir = "/mnt/Bessel/Gproj/Gerard_DATA/FAT-SAT/TRIO_results"

def train_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    """
    Need to select first scan, the original one
    """
    volume = tf.expand_dims(volume, axis=3)
    
    return volume, label

    
def data_aug(volume, label):
#noise
    noise = np.random.normal(0,0.05,(40,31,13))
    noise= np.asarray(noise,dtype=np.float32)
    noise = tf.convert_to_tensor(noise)
    noise = tf.expand_dims(noise, axis=3)  
    r = random.random()
    if r<0.5:
        volume = volume + noise
    return volume,label

def data_aug_select_vol(volume, label):
    """
    This function selects, at compute time, a random volume from the available volumes.

    It is a combination of offline and online data augmentation, just having the slices precomputed.
    It selects a random slice each time.

    This means that we need to load all the slices in a dictionary, in the other files.
    
    It needs to go before the train_preprocessing values
    """
    # select a random one
    idx = np.random.choice(volume.shape[0])
    volume = volume[idx]
    return volume, label


def select_vol_validation(volume, label):
    """
    Same as before but without data validation

    selecting always the same volume
    
    """
    # select a random one
    volume = volume[0]
    return volume, label


###################
#### read excel ###
###################

#train
info=pd.read_excel(path_excel)
ids_excel=[]
label_right=[]
label_left=[]
ulls_lesio=0
ulls_nolesio=0
pacient_neg=0
for row in info.itertuples():
    ids_excel.append(str(row[1]))
    left=row[2]
    if left=='Y':
        label_left.append(float(1))
        ulls_lesio+=1
    else:
        label_left.append(float(0))
        ulls_nolesio+=1

    right=row[3]
    if right=='Y':
        label_right.append(float(1))
        ulls_lesio+=1
    else:
        label_right.append(float(0))
        ulls_nolesio+=1
    
    if left=='N' and right=='N':
        pacient_neg+=1
print('Amount of eyes with lesions in dataset = ', str(ulls_lesio)) #72
print('Amount of eyes with no lesions in dataset = ', str(ulls_nolesio)) #142
print('Amount of patients with no lesions in dataset = ', str(pacient_neg))

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
    scans_l.append(process_scan(f'{path_scans}/{infile}', 'n4_{}_T2FastSat_crop_Left_flipped.nii'.format(infile)))
    scans_r.append(process_scan(f'{path_scans}/{infile}', 'n4_{}_T2FastSat_crop_Right.nii'.format(infile)))

    position=ids_excel.index(infile)
    labels_l.append(label_left[position])
    labels_r.append(label_right[position])
    groups.append(cont)
    groups.append(cont)
    cont+=1

print('The number of nerves loaded is: ' + str(len(scans_l)+len(scans_r)))
print('The number of nerves loaded is: ' + str(len(labels_l)+len(labels_r)))

#################
# CNN STRUCTURE #
#################
#####################
# HYPERPARAM SEARCH #
#####################

"""
bn_list = [True, False]
dropout_list = [0, 0.1, 0.25]
lr_list = [3e-4]
filt_factor_list = [0.5, 1, 2]
dense_list = [64, 128, 256]
"""

bn_list = [False]
dropout_list = [0.1]
lr_list = [3e-4]
filt_factor_list = [2]
dense_list = [256]

# create new dir in out dir
if not os.path.exists(out_dir + "/hyperparam_search_lrlow"):
    os.makedirs(out_dir + "/hyperparam_search_lrlow")

hyperparam = product(bn_list, dropout_list, lr_list, filt_factor_list, dense_list)

for (bn, dropout, lr, filt_factor, dense) in hyperparam:
    # check if the iteration already exists in disk, if it does, continue
    hyperparam_csv = f"{out_dir}/hyperparam_search_lrlow/{bn}_{dropout}_{lr}_{filt_factor}_{dense}.csv"
    # if os.path.exists(hyperparam_csv): 
    #     print('Already run!')
    #    continue

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
        n_YN = len(pos_labels)
        selection=list(range(n_YN))
        selection=random.sample(selection,len(selection))
        for i in range(n_YN):
            if i<n_YN*0.75:
                x_train.append(pos_scans[selection[i]])
                y_train.append(pos_labels[selection[i]])
                ids_train.append(pos_ids[selection[i]])
            else:
                x_val.append(pos_scans[selection[i]])
                y_val.append(pos_labels[selection[i]])
                ids_val.append(pos_ids[selection[i]])

        print(np.shape)

        #divide scanners with a Y and a Y eye
        n_YY = len(pos2_scans)
        selection2=list(range(n_YY))
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
        n_NN = len(neg_scans)
        selection3=list(range(n_NN))
        selection3=random.sample(selection3,len(selection3))

        for i in range(n_NN):
            if i<n_NN*0.75:
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
        #import pdb; 
        #pdb.set_trace()
    
        ####################
        ## STRATIFICATION ##
        ####################
        '''
        gfk=StratifiedGroupKFold(n_splits=4, shuffle=True) #nsplit=4 means that splits in the ratio 75-25
        for train, test in gfk.split(scans,labels,groups=groups):
            ind_train=train
            ind_test=test
            break

        scans=np.asarray(scans)
        x_train=scans[ind_train]
        x_val=scans[ind_test]

        labels=np.asarray(labels)
        y_train=labels[ind_train]
        y_val=labels[ind_test]

        ids_scans=np.asarray(ids_scans)
        ids_xtrain=ids_scans[ind_train]
        ids_xval=ids_scans[ind_test]'''
        #TRAIN
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
        ### SUBSAMPLING IS DONE HERE
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

        #VALIDATION
        

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

        val_p=0
        val_n=0
        for i in range(len(y_val)):
            if y_val[i]==1.0:
                val_p+=1
            else: 
                val_n+=1
        val_p_pctg=val_p*100/len(y_val)
        val_n_pctg=val_n*100/len(y_val)

        print("Positive percentage in train:", train_p_pctg)
        print("Negative percentage in train:", train_n_pctg)
        print("Positive percentage in val:", val_p_pctg)
        print("Negative percentage in val:", val_n_pctg)


        #convert to float
        # TODO: POSSIBLE NEEDS TO BE CHANGED, given that X now is a list of lists
        x_train=np.asarray(x_train,dtype=np.float32)
        x_val=np.asarray(x_val,dtype=np.float32)
        y_train=np.asarray(y_train,dtype=np.float32)
        y_val=np.asarray(y_val,dtype=np.float32)

        # Define data loaders.
        train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        # batch size can be larger, as our images are smaller
        batch_size = 16

        # Augment the on the fly during training.
        # add a mapping to the new function
        train_dataset = (
            train_loader.shuffle(len(x_train))
            .map(data_aug_select_vol)
            .map(train_preprocessing)
            .map(data_aug)
            #.map(py_augment)
            #.map(lambda x, y: (trainAug(x), y))
            .batch(batch_size)
            .prefetch(2)
        )
        #
        # Only rescale.
        validation_dataset = (
            validation_loader.shuffle(len(x_val))
            .map(select_vol_validation)
            .map(validation_preprocessing)
            .batch(batch_size)
            .prefetch(2)
        )

        def get_model(width=40, height=31, depth=13):

            inputs = keras.Input((width, height, depth, 1))

            x = layers.Conv3D(filters=32*filt_factor, kernel_size=3, activation="relu")(inputs)
            #x = layers.Conv3D(filters=16, kernel_size=3, activation="relu")(x)
            x = layers.MaxPool3D(pool_size=2)(x)
            if bn: x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)

            x = layers.Conv3D(filters=64*filt_factor, kernel_size=3, activation="relu")(x)
            x = layers.MaxPool3D(pool_size=2)(x)
            if bn: x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)

            x = layers.GlobalAveragePooling3D()(x)
            x = layers.Dense(units=dense, activation="relu")(x)
            x = layers.Dropout(dropout)(x)

            outputs = layers.Dense(units=1, activation="sigmoid")(x)

            # Define the model.
            model = keras.Model(inputs, outputs, name="3dcnn")
            return model

        # Build model.
        model = get_model(width=40, height=31, depth=13)
        model.summary()

        # Compile model.
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            lr, decay_steps=100000, decay_rate=0.96, staircase=True
        )

        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=["acc"],
        )

        # Define callbacks.
        #checkpoint_cb = keras.callbacks.ModelCheckpoint(
        #    "3d_image_classification.h5", monitor='acc', save_best_only=True , mode='max'
        #)
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="loss", patience=15)

        # Train the model, doing validation at the end of each epoch
        epochs = 200

        history = model.fit(train_dataset,validation_data=validation_dataset,epochs=epochs,shuffle=True,verbose=1,callbacks=[early_stopping_cb])
        
        """
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        #plt.plot(history.history['acc'])
        #plt.plot(history.history['val_acc'])
        plt.legend(['Train loss', 'Val loss'])#, 'Train acc', 'Val acc'])
        plt.title('Metrics results')
        plt.savefig(f'{out_dir}/metric_plots/plot_{p}.png')
        plt.close()

        model.save(f'{out_dir}/models/model_{p}.h5')
        """

        #model.load_weights("3d_image_classification.h5")

        #####################
        ## METRICS RESULTS ##
        #####################
    
        # WHICH METRICS DO I NEED?
        # Need to output the probabilities , but not for the training, right?

        tp=[]
        fp=[]
        tn=[]
        fn=[]
        loss_list=[] ### NEEDS TO BE SAVED, AS WITH THIS WE WILL DO THE AUC
        x_te=x_val
        y_te=y_val

        id_scan_tp=[]
        id_scan_tn=[]
        id_scan_fp=[]
        id_scan_fn=[]
        y_p_list=[]
        y_r_list=[]

        for i in range(len(x_te)):
            # NO NEED TO COMPUTE GRADIENTS AGAIN, AT LEAST FOR NOW
            # TODO: COMMENT
            images = np.expand_dims(x_te[i][0], axis=-1)
            images = tf.Variable(np.expand_dims(images, axis=0).astype('float32'))
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
                id_scan_tp.append(ids_xval[i])
        
            elif y_te[i]==np.around(loss)[0] and y_te[i]==0.0: #tn
                tn.append(grad_eval)
                id_scan_tn.append(ids_xval[i])

            elif y_te[i]!=np.around(loss)[0] and y_te[i]==0.0: #fp
                fp.append(grad_eval)
                id_scan_fp.append(ids_xval[i])

            elif y_te[i]!=np.around(loss)[0] and y_te[i]==1.0: #fn
                fn.append(grad_eval)
                id_scan_fn.append(ids_xval[i])
            
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

        try:
            recall_permu.append((len(tp))/(len(tp)+len(fn)))
        except ZeroDivisionError:
            recall_permu.append(0.0)
        try:
            precisiontrue_permu.append((len(tp))/(len(tp)+len(fp)))
        except ZeroDivisionError:
            precisiontrue_permu.append(0.0)
        try:
            specificity_permu.append((len(tn))/(len(tn)+len(fp)))
        except ZeroDivisionError:
            specificity_permu.append(0.0)

    maxvalue = max(accuracy_bal)
    minvalue = min(accuracy_bal)
    maxpos = accuracy_bal.index(maxvalue)

    tp,tn,fp,fn=[tp_sm[maxpos],tn_sm[maxpos],fp_sm[maxpos],fn_sm[maxpos]]
    id_scan_tp,id_scan_tn,id_scan_fp,id_scan_fn=[ids_tp_sm[maxpos],ids_tn_sm[maxpos],ids_fp_sm[maxpos],ids_fn_sm[maxpos]]


    #delete previous scans because each time it changes
    # ALSO POSSIBLE TO JUST REMOVE THIS
    
    import os
    import glob
    #save saliency maps of individual scans
    general_scans=[tp,tn,fp,fn]
    general_names=['tp','tn','fp','fn']
    general_ids=[id_scan_tp,id_scan_tn,id_scan_fp,id_scan_fn]
    for i in range(4):
        path=f'{out_dir}/saliency_maps/{general_names[i]}'
        if not os.path.exists(path):
            os.makedirs(path)
        # for filename in os.listdir(path):
        #     f = os.path.join(path, filename)
        #     os.remove(f)

        for j in range(len(general_ids[i])):
            id_exemple=general_ids[i][j]
            id_value=str(id_exemple[:-2])
            if id_exemple[-1]=='r':
                scan_ex = nib.load(path_scans+'/'+id_value+'/Eye_1_n4_{}_T2FastSat_crop_Right.nii'.format(id_value,))
            else:
                scan_ex = nib.load(path_scans+'/'+id_value+'/Eye_1_n4_{}_T2FastSat_crop_Left_flipped.nii'.format(id_value,))

            img=nib.Nifti1Image(general_scans[i][j].numpy().squeeze(),scan_ex.affine,scan_ex.header)
            nib.save(img, f'{out_dir}/saliency_maps/{general_names[i]}/{general_ids[i][j]}_smap.nii.gz')

    accuracy_av=sum(accuracy)/iterations
    acc_av_std=np.std(accuracy)
    accuracy_bal_av=sum(accuracy_bal)/iterations
    accuracy_bal_std=np.std(accuracy_bal)
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
    print('Accuracy average balanced='+str(accuracy_bal_av))
    print('Min value accuracy balanced = ' +str(minvalue))
    print('Max value accuracy balanced = ' + str(maxvalue))
    print('Recall average='+str(recall_av))
    print('Recall std='+str(recall_std))
    print('Precision true average = ' + str(precisiontrue_av))
    print('Precision true std = ' + str(precisiontrue_std))
    print('Specificity average = ' + str(specificity_av))
    print('Specificity std = ' + str(specificity_std))

    ########################
    ## SAVE  METRICS #######
    ########################

    df_dict = {
        "bn": bn, 
        "dropout": dropout,
        "lr": lr, 
        "filt_factor": filt_factor, 
        "dense": dense,
        'Accuracy': accuracy_av, 
        'Accuracy Balanced': accuracy_bal_av,
        'Accuracy Balanced std': accuracy_bal_std,
        'Recall':recall_av,
        'Precision': precisiontrue_av,
        'Specificity': specificity_av,
        'Recall std': recall_std,
        'Specificity std': specificity_std,
    }

    df_results = pd.DataFrame(df_dict, index = [0])
    df_results.to_csv(hyperparam_csv, index=False)

"""
## Enter folder and merge all the csv in the folder
list_of_results = []
for csv_file in glob.glob(f"{out_dir}/hyperparam_search_lrlow/*.csv"):
    df_csv = pd.read_csv(csv_file)
    list_of_results.append(df_csv)

# save resulting csv
df_list_of_results = pd.concat(list_of_results)
df_list_of_results.to_csv(f"{out_dir}/hyperparam_file_lrlow.csv")
"""