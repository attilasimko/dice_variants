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

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/dice_variants/dataverse_files/training"
base_path = "/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/WMH"

if (os.path.isdir(base_path)):
    shutil.rmtree(base_path)
os.mkdir(base_path)
os.mkdir(base_path + "/train")
os.mkdir(base_path + "/val")
os.mkdir(base_path + "/test")

def resize(img, mask=False):
    new_img = np.zeros((256, 256, np.shape(img)[2]))
    smaller_shape = np.min(np.shape(img)[:2])
    new_shape = np.array([int(np.shape(img)[1] * 256 / smaller_shape), int(np.shape(img)[0] * 256 / smaller_shape)])
    start_idx = np.array([int((new_shape[0] - 256) / 2), int((new_shape[1] - 256) / 2)])
    for i in range(np.shape(img)[2]):
        if (mask):
            new_img[:,:,i] = cv2.resize(np.array(img[:,:,i], dtype=np.float64), new_shape, interpolation=cv2.INTER_NEAREST).T[start_idx[0]:start_idx[0]+256, start_idx[1]:start_idx[1]+256]
        else:
            new_img[:,:,i] = cv2.resize(np.array(img[:,:,i], dtype=np.float64), new_shape, interpolation=cv2.INTER_CUBIC).T[start_idx[0]:start_idx[0]+256, start_idx[1]:start_idx[1]+256]
    return new_img

def znorm(img):
    if (np.max(img) == np.min(img)):
        return img
    
    return (img - np.mean(img)) / np.std(img)

def get_data(path, mask=False):
    img = nib.load(path).get_fdata()
    img = resize(img, mask)
    return img

sites = os.listdir(os.path.join(data_path))
for site in sites:
    patients = os.listdir(os.path.join(data_path, site))
    np.random.shuffle(patients)
    for patient in patients:
        print(100 * patients.index(patient) / len(patients))
        try:
            T1 = get_data(f"{data_path}/{site}/{patient}/pre/T1.nii.gz")
            FLAIR = get_data(f"{data_path}/{site}/{patient}/pre/FLAIR.nii.gz")
            Structures = get_data(f"{data_path}/{site}/{patient}/wmh.nii.gz", True)
            Background = Structures == 0
            WMH = Structures == 1
            Other = Structures == 2

            if (patients.index(patient) / len(patients) < 0.6):
                sample_path = "/train/"
            elif (patients.index(patient) / len(patients) < 0.8):
                sample_path = "/val/"
            else:
                sample_path = "/test/"
            
            for i in range(np.shape(T1)[2]):
                if (sample_path == "/train/"):
                    if ((np.sum(T1[:, :, i]) == 0) | (np.sum(FLAIR[:, :, i]) == 0)):
                        print("Skipping slice: ", i)
                        continue

                np.savez_compressed(base_path + sample_path + site + "_" + patient + "_" + str(i),
                                    T1 = np.array(znorm(T1[:, :, i]), dtype=np.float64),
                                    FLAIR = np.array(znorm(FLAIR[:, :, i]), dtype=np.float64),
                                    Background = np.array(Background[:, :, i], dtype=bool),
                                    WMH = np.array(WMH[:, :, i], dtype=bool),
                                    Other = np.array(Other[:, :, i], dtype=bool)
                                    )
        except Exception as e:
            print("Error in patient: ", patient + " - " + str(e))