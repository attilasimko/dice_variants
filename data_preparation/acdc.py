import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pydicom
import numpy as np
import cv2
import nibabel as nib
from skimage import draw
import matplotlib.pyplot as plt
import SimpleITK as sitk
import shutil
from rt_utils import RTStructBuilder
import sys
sys.path.insert(1, os.path.abspath('.'))
import utils

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/dice_variants/ACDC/database"
base_path = "/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/ACDC_0"

if (os.path.isdir(base_path)):
    shutil.rmtree(base_path)
os.mkdir(base_path)
os.mkdir(base_path + "/train")
os.mkdir(base_path + "/val")
os.mkdir(base_path + "/test")

def resize(img):
    if (np.shape(img)[0] < 256):
        img = np.pad(img, ((int(np.floor((256 - img.shape[0]) / 2)), int(np.ceil((256 - img.shape[0]) / 2))), (0, 0), (0, 0)), mode='constant', constant_values=0)
    elif (np.shape(img)[0] > 256):
        start_idx = int(np.floor((np.shape(img)[0] - 256) / 2))
        img = img[start_idx:int(start_idx + 256), :, :]

    if (np.shape(img)[1] < 256):
        img = np.pad(img, ((0, 0), (int(np.floor((256 - img.shape[1]) / 2)), int(np.ceil((256 - img.shape[1]) / 2))), (0, 0)), mode='constant', constant_values=0)
    elif (np.shape(img)[1] > 256):
        start_idx = int(np.floor((np.shape(img)[1] - 256) / 2))
        img = img[:, start_idx:int(start_idx + 256), :]
    return img
    # new_img = np.zeros((256, 256, np.shape(img)[2]))
    # smaller_shape = np.min(np.shape(img)[:2])
    # new_shape = np.array([int(np.shape(img)[1] * 256 / smaller_shape), int(np.shape(img)[0] * 256 / smaller_shape)])
    # start_idx = np.array([int((new_shape[0] - 256) / 2), int((new_shape[1] - 256) / 2)])
    # for i in range(np.shape(img)[2]):
    #     if (mask):
    #         new_img[:,:,i] = cv2.resize(np.array(img[:,:,i], dtype=np.float64), new_shape, interpolation=cv2.INTER_NEAREST).T[start_idx[0]:start_idx[0]+256, start_idx[1]:start_idx[1]+256]
    #     else:
    #         new_img[:,:,i] = cv2.resize(np.array(img[:,:,i], dtype=np.float64), new_shape, interpolation=cv2.INTER_CUBIC).T[start_idx[0]:start_idx[0]+256, start_idx[1]:start_idx[1]+256]
    # return new_img

def znorm(img):
    if (np.max(img) == np.min(img)):
        return img
    
    return (img - np.mean(img)) / np.std(img)

def get_data(path):
    img = nib.load(path).get_fdata()
    img = resize(img)
    return img

samples = os.listdir(os.path.join(data_path))
for sample in samples:
    patients = os.listdir(os.path.join(data_path, sample))
    np.random.shuffle(patients)
    for patient in patients:
        frames = [k.split('_')[1] for k in os.listdir(os.path.join(data_path, sample, patient)) if (("frame" in k) & ("gt" in k))]
        for frame_num in frames:
            try:
                # st1 = get_data(f"{data_path}/{sample}/{patient}/{patient}_4d.nii.gz")
                frame = get_data(f"{data_path}/{sample}/{patient}/{patient}_{frame_num}.nii.gz")
                frame_gt = get_data(f"{data_path}/{sample}/{patient}/{patient}_{frame_num}_gt.nii.gz")

                Background = frame_gt == 0
                RV = frame_gt == 1
                Myo = frame_gt == 2
                LV = frame_gt == 3
                
                for i in range(np.shape(Background)[2]):
                    np.savez_compressed(base_path + "/" + sample + "/" + patient + "_" + frame_num + "_" + str(i),
                                        MRI = np.array(znorm(frame[:, :, i]), dtype=np.float64),
                                        Background = np.array(Background[:, :, i], dtype=bool),
                                        LV = np.array(LV[:, :, i], dtype=bool),
                                        RV = np.array(RV[:, :, i], dtype=bool),
                                        Myo = np.array(Myo[:, :, i], dtype=bool)
                                        )
                
            except Exception as e:
                print("Error in patient: ", patient + " - " + str(e))