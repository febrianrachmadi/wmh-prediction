
''' SECTION 1: Call libraries
##
# '''

import os, io, h5py, math, datetime, sys, json

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
from src.models import discriminator_loss, generator_loss

from src.datautils import load_data, data_prep_load_2D, data_prep_save
from src.datautils import convert_to_1hot, convert_from_1hot


print("\n>>> TEST CREATING MODELS AND LOAD WEIGHTS <<<")

# Network parameters
n_chn_gen = 1 # brain T2-FLAIR MRI

# print("n_label  : ", n_label)
# print("n_chn_gen: ", n_chn_gen)
# print("n_chn_dsc: ", n_chn_dsc)

# Input image size
crop = 0
imageSizeOri = 256
imageSize = imageSizeOri - crop

''' SECTION 4: Define classes and functions
## 
# number of labels
# n_label = 5
# 0: background
# 1: shrinking WMH
# 2: growing WMH
# 3: stable WMH
# 4: stroke lesions
# '''
def create_models(imageSize=256, n_chn_gen=1, n_label=5):

    backend.clear_session()
    n_chn_dsc = n_chn_gen + n_label + 1

    all_networks = []
    C0 = 0.25  
    C1 = 0.75
    C2 = 0.75
    C3 = 0.50
    C4 = 0.50
    cost_func = loss.categorical_focal_loss(alpha=[[C0, C1, C2, C3, C4]], gamma=2)

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
    my_network_1 = ProbUNet2Prior_Y1Y2GAN(
        discriminator=discriminator,
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
    my_network_1.compile(
        batch_size=16,
        prior_opt=generator_optimizer, 
        posterior_opt=generator_optimizer,
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    # Instantiate the ProbUNet_GAN model
    my_network_2 = ProbUNet2Prior_Y1Y2GAN(
        discriminator=discriminator,
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
    my_network_2.compile(
        batch_size=16,
        prior_opt=generator_optimizer, 
        posterior_opt=generator_optimizer,
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    # Instantiate the ProbUNet_GAN model
    my_network_3 = ProbUNet2Prior_Y1Y2GAN(
        discriminator=discriminator,
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
    my_network_3.compile(
        batch_size=16,
        prior_opt=generator_optimizer, 
        posterior_opt=generator_optimizer,
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    # Instantiate the ProbUNet_GAN model
    my_network_4 = ProbUNet2Prior_Y1Y2GAN(
        discriminator=discriminator,
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
    my_network_4.compile(
        batch_size=16,
        prior_opt=generator_optimizer, 
        posterior_opt=generator_optimizer,
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    all_networks.append(my_network_1)
    all_networks.append(my_network_2)
    all_networks.append(my_network_3)
    all_networks.append(my_network_4)

    all_networks[0].built = True
    all_networks[1].built = True
    all_networks[2].built = True
    all_networks[3].built = True

    return all_networks






