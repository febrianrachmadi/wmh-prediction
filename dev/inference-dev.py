
import os, io, h5py, math, datetime, sys, csv, glob, shutil

import numpy as np
import pandas as pd
import nibabel as nib
from tabulate import tabulate
import matplotlib.pyplot as plt

import tensorflow as tf 
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, backend, callbacks

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score

sys.path.append('/wmh-prediction/')

from src.datautils import load_data, data_prep_load_2D, data_prep_save
from src.datautils import convert_to_1hot, convert_from_1hot

import initialization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

''' SECTION 3: Define classes and functions
##
# '''

# Class to load NifTi (.nii) data
class load_data(object):
    # Load NII data
    def __init__(self, image_name):
        # Read nifti image
        nim = nib.load(image_name)
        image = nim.get_fdata()
        affine = nim.affine
        self.image = image
        self.affine = affine
        self.dt = nim.header['pixdim'][4]
        self.pixdim = nim.header['pixdim'][1:4]

# Function to prepare the data after opening from .nii file
def data_prep(image_data):
    # Extract the 2D slices from the cardiac data
    image = image_data.image
    images = []
    for z in range(image.shape[2]):
        image_slice = image[:, :, z]
        images += [image_slice]
    images = np.array(images, dtype='float32')

    # Both theano and caffe requires input array of dimension (N, C, H, W)
    # TensorFlow (N, H, W, C)
    # Add the channel dimension, swap width and height dimension
    images = np.expand_dims(images, axis=3)

    return images

# Function to prepare the data before saving to .nii file
def data_prep_save(image_data):
    image_data = np.squeeze(image_data)
    output_img = np.swapaxes(image_data, 0, 2)
    output_img = np.rot90(output_img)
    output_img = output_img[::-1,...]   # flipped

    return output_img

def convert_to_1hot(label, n_class):
    # Convert a label map (N x H x W x 1) into a one-hot representation (N x H x W x C)
    print(" --> SIZE = " + str(label.shape))
    print(" --> MAX = " + str(np.max(label)))
    print(" --> MIN = " + str(np.min(label)))
    label_flat = label.flatten().astype(int)
    n_data = len(label_flat)
    print(" --> SIZE = " + str(label_flat.shape))
    print(" --> LEN = " + str(n_data))
    label_1hot = np.zeros((n_data, n_class), dtype='int16')
    print(" --> 1HOT-SIZE = " + str(label_1hot.shape))
    label_1hot[range(n_data), label_flat] = 1
    label_1hot = label_1hot.reshape((label.shape[0], label.shape[1], label.shape[2], n_class))

    return label_1hot

# Convert a label map (N x H x W x C) from a one-hot representation (N x H x W x 1)
def convert_from_1hot(label, to_float=False):
    N, H, W, C = label.shape
    label_flat = label.reshape((N * H * W, C))
    n_data = len(label_flat)

    if to_float:
        label_n_class = np.zeros((n_data, 1), dtype='float32')
        max_class = np.argmax(label_flat, axis=1)
        label_n_class[range(n_data), 0] = label_flat[range(n_data), max_class]
    else:
        label_n_class = np.zeros((n_data, 1), dtype='uint8')
        label_n_class[range(n_data), 0] = np.argmax(label_flat, axis=1)

    label_n_class = np.squeeze(label_n_class.reshape((N, H, W, 1)))

    return label_n_class


print("")
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

brain_dir = str(sys.argv[1])
stroke_dir = str(sys.argv[2])
brain_list = str(sys.argv[3])
output_dir = str(sys.argv[4])
if len(sys.argv) > 5:
    sampling_number = int(sys.argv[5])
else:
    sampling_number = 10

print("\n>>> CREATING MODELS AND LOAD WEIGHTS <<<")
networks_lbc = initialization.create_models()
networks_mss = initialization.create_models()

print("networks_lbc length:", len(networks_lbc))
print("networks_mss length:", len(networks_mss))
print(">> ALL MODELS ARE CREATED SUCCESSFULLY..")

weights_loc = "/wmh-prediction/"
for fold in range(4):
    saved_model_name = weights_loc + 'LBC1936-weights/LBC1936-model-cv' + str(fold) + '-weights.h5'
    print("LOAD WEIGHTS FROM: ", saved_model_name)
    networks_lbc[fold].load_weights(saved_model_name)

    saved_model_name = weights_loc + 'MSS2-weights/MSS2-model-cv' + str(fold) + '-weights.h5'
    print("LOAD WEIGHTS FROM: ", saved_model_name)
    networks_mss[fold].load_weights(saved_model_name)
    print(">>> SAVED MODEL FROM FOLD", str(fold), "IS LOADED !! <<<")

print("\n>>> CHECK CUDA GPUS <<<")
check_gpu = tf.test.is_gpu_available(cuda_only=True)
print("CUDA GPU AVAILABLE:", check_gpu)

print("")
print("brain_dir :", brain_dir)
print("stroke_dir:", stroke_dir)
print("brain_list:", brain_list)
print("sampling_number:", sampling_number)
print("")

print("Create output folder at: ", output_dir)
try:
    os.makedirs(output_dir)
except OSError:
    if not os.path.isdir(output_dir):
        raise
print("DONE! Created at:", output_dir)

all_names = []
all_minve = []
all_pve   = []
all_maxve = []

with open(brain_list) as f:
    reader = csv.reader(f)
    print("\n>>> PERFORMING INFERENCE <<<")
    for row in reader:
        brain_mri_loc = brain_dir + "/" + row[0]
        stroke_mask_loc = stroke_dir + "/"+ row[0]
        if os.path.isfile(brain_mri_loc):
            name = row[0].replace(".nii.gz", "")
            print(name, end='')
            all_names.append(name)

            ## LOAD BRAIN MRI .nii.gz
            img_bran_mri = load_data(brain_mri_loc)
            dat_bran_mri = data_prep(img_bran_mri)

            # Exclude Stroke Lesions (SL) tissues on 1st time point data
            is_stroke_avail = False
            if os.path.isfile(stroke_mask_loc):
                is_stroke_avail = True
                img_stroke_mask  = load_data(stroke_mask_loc)
                dat_stroke_mask = data_prep(img_stroke_mask)
            else:
                dat_stroke_mask = np.zeros(dat_bran_mri.shape)
            dat_stroke_mask = np.squeeze(dat_stroke_mask)
            
            dat_bran_mri = ((dat_bran_mri - np.mean(dat_bran_mri)) / np.std(dat_bran_mri)) # normalise to zero mean unit variance 3D
            input_prediction = np.nan_to_num(dat_bran_mri)

            output_img_pred_nparray = None
            for fold in range(4):
                for si in range(sampling_number):
                    output_img_pred = networks_lbc[fold].predict(input_prediction, batch_size=16)

                    if output_img_pred_nparray is None:
                        output_img_pred_nparray = np.expand_dims(output_img_pred, axis=0)
                        output_img_pred_1hot_nparray = np.expand_dims(convert_from_1hot(output_img_pred), axis=0)
                    else:
                        temp_nparray = np.expand_dims(output_img_pred, axis=0)
                        output_img_pred_nparray = np.append(output_img_pred_nparray, temp_nparray, axis=0)
                        temp_nparray = np.expand_dims(convert_from_1hot(output_img_pred), axis=0)
                        output_img_pred_1hot_nparray = np.append(output_img_pred_1hot_nparray, temp_nparray, axis=0)
            output_img_pred = np.average(output_img_pred_nparray, axis=0)
            output_img_pred_lbl = convert_from_1hot(output_img_pred)

            # Save prediction labels
            output_img = data_prep_save(output_img_pred_lbl)
            nim = nib.Nifti1Image(output_img.astype('int8'), img_bran_mri.affine)
            nib.save(nim, output_dir + '/' + name + '_DEM.nii.gz')

            ## CALCULATE VPE
            dem_mask = np.zeros(output_img_pred_lbl.shape)
            index_where = np.where(np.logical_and(output_img_pred_lbl >= 2, output_img_pred_lbl < 4))
            dem_mask[index_where] = 1
            pve = (np.count_nonzero(dem_mask) * np.prod(img_bran_mri.pixdim)) / 1000

            ## CALCULATE MaxVE
            wmh_change_mask_fake = output_img_pred
            wmh_change_mask_fake_temp = np.delete(wmh_change_mask_fake, [1, 4], 3)
            wmh_change_mask_fake_temp = convert_from_1hot(wmh_change_mask_fake_temp)
            
            vol_out_mm3 = np.count_nonzero(wmh_change_mask_fake_temp) * np.prod(img_bran_mri.pixdim)
            max_ve = vol_out_mm3 / 1000

            ## CALCULATE MinVE
            wmh_change_mask_fake = output_img_pred
            wmh_change_mask_fake_temp = np.delete(wmh_change_mask_fake, [2, 4], 3)
            wmh_change_mask_fake_temp = convert_from_1hot(wmh_change_mask_fake_temp)

            vol_out_mm3 = np.count_nonzero(wmh_change_mask_fake_temp) * np.prod(img_bran_mri.pixdim)
            min_ve = vol_out_mm3 / 1000

            print(" -> min-ve:", min_ve, "ml, pve", pve, "ml, max-ve:", max_ve, "ml")
            all_minve.append(min_ve)
            all_pve.append(pve)
            all_maxve.append(max_ve)

# initialize data of lists.
data = {
    'ID': all_names,
    'Min-VE': all_minve,
    'Point-VE': all_pve,
    'Max-VE': all_maxve,
}
 
# Create DataFrame
df = pd.DataFrame(data)

print("")
print(">>> SAVE AND DISPLAY RECAP <<<")
df.to_csv(output_dir + '/recap.csv')

# displaying the DataFrame
print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
print("FINISHED!\n")