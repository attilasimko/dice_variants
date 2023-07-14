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

def resize(img):
    new_img = np.zeros((256, 256, np.shape(img)[2]))
    for i in range(np.shape(img)[2]):
        new_img[:,:,i] = cv2.resize(np.array(img[:,:,i], dtype=np.float32), (256, 256), interpolation=cv2.INTER_CUBIC)
    return new_img

def znorm(img):
    return (img - np.mean(img)) / np.std(img)

def get_data(path):
    img = nib.load(path).get_fdata()
    img = resize(img)
    return img

sites = os.listdir(os.path.join(data_path))
for site in sites:
    patients = os.listdir(os.path.join(data_path, site))
    for patient in patients:
        print(100 * patients.index(patient) / len(patients))
        try:
            T1 = get_data(f"{data_path}/{site}/{patient}/pre/T1.nii.gz")
            FLAIR = get_data(f"{data_path}/{site}/{patient}/pre/FLAIR.nii.gz")
            Structures = get_data(f"{data_path}/{site}/{patient}/wmh.nii.gz")
            Background = Structures < 0.5
            WMH = (Structures >= 0.5) & (Structures < 1.5)
            Other = Structures >= 1.5

            sample = np.random.rand()
            if (sample < 0.8):
                sample_path = "/train/"
            elif (sample < 0.9):
                sample_path = "/val/"
            else:
                sample_path = "/test/"
            
            for i in range(np.shape(T1)[2]):
                np.savez_compressed(base_path + sample_path + site + "_" + patient + "_" + str(i),
                                    T1 = np.array(znorm(T1[:, :, i]), dtype=np.float32),
                                    FLAIR = np.array(znorm(FLAIR[:, :, i]), dtype=bool),
                                    Background = np.array(Background[:, :, i], dtype=bool),
                                    WMH = np.array(WMH[:, :, i], dtype=bool),
                                    Other = np.array(Other[:, :, i], dtype=bool)
                                    )
        except Exception as e:
            print("Error in patient: ", patient + " - " + str(e))