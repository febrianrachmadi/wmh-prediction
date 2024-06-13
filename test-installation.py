
# coding: utf-8

''' SECTION 1: Call libraries
##
# '''

import os, io, h5py, math, datetime, sys, json, shutil

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import tensorflow as tf 
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, backend, callbacks

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score

import src.losses as loss
from src.models import ProbUNet2Prior_Y1Y2GAN, DiscriminatorSNGAN
from src.models import generator_optimizer, discriminator_optimizer 
from src.models import discriminator_loss

from src.datautils import load_data, data_prep_load_2D, data_prep_save
from src.datautils import convert_to_1hot, convert_from_1hot

np.seterr(divide='ignore', invalid='ignore')

print("\n>>> CHECK CUDA GPUS <<<")
check_gpu = tf.test.is_gpu_available(cuda_only=True)
print("CUDA GPU AVAILABLE:", check_gpu)

''' SECTION 2: Some variables need to be specified
##
# '''
print("\n>>> TEST CREATING FOLDERS <<<")
vol_dir = '/home/docker/data/IO/output' 
dir_test = vol_dir + '/test'
print("dir_test: ", dir_test)
try:
    os.makedirs(dir_test)
except OSError:
    if not os.path.isdir(dir_test):
        raise
print("DONE! Test creating at:", dir_test)

print("\n>>> TEST WRITING FILE <<<")
f = open(dir_test + ".txt", "w")
write_str = "Test writing at " + dir_test
f.write(write_str)
f.close()
print("DONE! Test writing:", write_str)

print("\n>>> TEST REMOVING FOLDER AND ITS CONTENTS <<<")
shutil.rmtree(dir_test)
print("DONE!")

print("\n>>> READ TEST FOLDERS AND MRI DATA <<<")
print("List of all folders and files at:", vol_dir)
for path, subdirs, files in os.walk(vol_dir):
    for name in files:
        print(os.path.join(path, name))

''' SECTION 3: Some variables need to be specified
##
# '''

print("\n>>> TEST ARGUMENTS <<<")
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

print("\n>>> TEST CREATING MODELS AND LOAD WEIGHTS <<<")
# number of labels
n_label = 5
# 0: background
# 1: shrinking WMH
# 2: growing WMH
# 3: stable WMH
# 4: stroke lesions

# Network parameters
n_chn_gen = 1 # brain T2-FLAIR MRI
n_chn_dsc = n_chn_gen + n_label + 1

print("n_label  : ", n_label)
print("n_chn_gen: ", n_chn_gen)
print("n_chn_dsc: ", n_chn_dsc)

# Input image size
crop = 0
imageSizeOri = 256
imageSize = imageSizeOri - crop

## Specify the location of .txt files for accessing the training data
trained_on_mss2 = False

''' SECTION 4: Define classes and functions
##
# '''

# Define the loss functions for the generator.
def generator_loss(misleading_labels, fake_logits, fake_imgs, real_imgs):
    bce = keras.losses.BinaryCrossentropy(from_logits=True)
    fcl = cost_func
    real_dem = layers.Lambda(lambda x : x[:,:,:,1:])(real_imgs)
    fake_dem = layers.Lambda(lambda x : x[:,:,:,1:])(fake_imgs)
    return bce(misleading_labels, fake_logits) + fcl(real_dem, fake_dem)

def uncertainty_map(y_gen, seg=None):
    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):
        log_samples = np.log(m_samp + eps)
        return -1.0*np.sum(m_gt*log_samples, axis=-1)
    
    if seg is not None:
        print("Calculating uncertainty map with label..")
    else:
        print("Calculating uncertainty map with no label..")

    y_pred = np.average(y_gen, axis=0)
    E_arr = np.zeros(y_gen.shape)
    for i in range(y_gen.shape[0]):
        for j in range(y_gen.shape[1]):
            if seg is None:
                E_arr[i,j,...] = np.expand_dims(pixel_wise_xent(y_gen[i,j,...], y_pred[j,...]), axis=-1)
            else:
                E_arr[i,j,...] = np.expand_dims(pixel_wise_xent(y_gen[i,j,...], seg[j,...]), axis=-1)

    return np.average(E_arr, axis=0)

''' SECTION 5: Testing 4 different networks in 4-fold
##
# '''

# Save the name of the data
name_all = []

## DSC Evaluation
mean_dsc_all = []
std_dsc_all = []
eval_ed_all = []

## VOL Evaluation
mean_vol_all = []
std_vol_all = []

## UNCERTAINTY Evaluation
mean_unc_all = []

C0 = 0.25  
C1 = 0.75
C2 = 0.75
C3 = 0.50
C4 = 0.50
cost_func = loss.categorical_focal_loss(alpha=[[C0, C1, C2, C3, C4]], gamma=2)

for fold in range(4):
    ## Create the DEP-UResNet
    backend.clear_session()

    discriminator = DiscriminatorSNGAN(
        encoder_filters=(32,64,128,256,512),
        downsample_rates=(2,2,2,2,2),
        filter_sizes=(3,3,3,3,3),
        n_downsamples=5,
        n_convs_per_block=4,
        conv_activation='relu',
        encoder_block_type='stride',
        input_shape=(imageSize, imageSize, n_chn_dsc),
        type='2D'
        )

    # Instantiate the ProbUNet_GAN model
    my_network = ProbUNet2Prior_Y1Y2GAN(
        discriminator=discriminator, # discriminator
        num_filters=[64,128,256,512,1024],
        latent_dim=4,
        discriminator_extra_steps=5,
        cost_function=cost_func,
        n_label=n_label,
        resolution_lvl=5,
        img_shape=(imageSize, imageSize, n_chn_gen),
        seg_shape=(imageSize, imageSize, n_label),
        downsample_signal=(2,2,2,2,2)
    )    

    # Compile the UNet_GAN3D model
    my_network.compile(
        batch_size=16,
        prior_opt=generator_optimizer, 
        posterior_opt=generator_optimizer,
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    my_network.built = True
    
    # print("\n>>> my_network SUMMARY <<<")
    # my_network.summary()

    if not trained_on_mss2:
        print("\n>>> SAVE WEIGHTS ON DIFFERENT FILE <<<")
        saved_model_name = 'LBC1936-weights/LBC1936-model-cv' + str(fold)
        print("saved_model_name: ", saved_model_name + '-weights.h5')
        my_network.load_weights(saved_model_name + '-weights.h5')
        print(">>> SAVED MODEL IS LOADED !! <<<")
        
    else:
        print("\n>>> SAVE WEIGHTS ON DIFFERENT FILE <<<")
        saved_model_name = 'MSS2-models/MSS2-model-cv' + str(fold)
        print("saved_model_name: ", saved_model_name + '-weights.h5')
        my_network.load_weights(saved_model_name + '-weights.h5')
        print(">>> SAVED MODEL IS LOADED !! <<<")

if check_gpu:
    print("\nFINISHED: Everything looks OK, including GPU is found and can be used.")
else:
    print("\nFINISHED: Everything looks OK, except GPU is NOT found.")


