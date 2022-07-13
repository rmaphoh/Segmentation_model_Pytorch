import torch
import logging
import random
import numpy as np
from glob import glob
from os import listdir
from PIL import Image
from os.path import splitext
from scipy.ndimage import rotate
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, img_size, train_or=True):
        self.imgs_dir_ = imgs_dir + 'images/'
        self.label_dir = imgs_dir + 'labels/'
        self.img_size = img_size
        self.train_or = train_or
        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir_)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def random_perturbation(self,imgs):
        for i in range(imgs.shape[0]):
            im=Image.fromarray(imgs[i,...].astype(np.uint8))
            en=ImageEnhance.Color(im)
            im=en.enhance(random.uniform(0.8,1.2))
            imgs[i,...]= np.asarray(im).astype(np.float32)
        return imgs 

    @classmethod
    def preprocess(self, img, label, train_or):

        img_array = np.array(img)
        label_array = np.array(label).astype(np.float32)
        vessel_max = np.amax(label_array)

        if vessel_max>1:
            label_array = label_array/255.0
        
        if train_or:
            if np.random.random()>0.5:
                img_array=img_array[:,::-1,:]    # flipped imgs
                label_array=label_array[:,::-1]

            angle = np.random.randint(360)
            img_array = rotate(img_array, angle, axes=(0, 1), reshape=False)
            img_array = self.random_perturbation(img_array)
            label_array = np.round(rotate(label_array, angle, axes=(0, 1), reshape=False))
        
        mean_value=np.mean(img_array[img_array[...,0] > 40.0],axis=0)
        std_value=np.std(img_array[img_array[...,0] > 40.0],axis=0)
        img_array=(img_array-mean_value)/std_value

        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        if len(label_array.shape) == 2:
            label_array = np.expand_dims(label_array, axis=2)
        
        img_array = img_array.transpose((2, 0, 1))
        label_array = label_array.transpose((2, 0, 1))
        label_array = np.where(label_array > 0.5, 1, 0)

        return img_array, label_array


    def __getitem__(self, index):

        idx = self.ids[index]
        label_file = glob(self.label_dir + idx  + '*')
        img_file = glob(self.imgs_dir_ + idx + '*')    

        label = Image.open(label_file[0]).resize(self.img_size)
        img = Image.open(img_file[0]).resize(self.img_size)

        img, label = self.preprocess(img, label, self.train_or)

        return {
            'name': idx,
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'label': torch.from_numpy(label).type(torch.FloatTensor)
        }


    
class BasicDataset_outside(Dataset):
    def __init__(self, imgs_dir, img_size):
        self.imgs_dir_ = imgs_dir
        self.img_size = img_size
        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir_)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def random_perturbation(self,imgs):
        for i in range(imgs.shape[0]):
            im=Image.fromarray(imgs[i,...].astype(np.uint8))
            en=ImageEnhance.Color(im)
            im=en.enhance(random.uniform(0.8,1.2))
            imgs[i,...]= np.asarray(im).astype(np.float32)
        return imgs 

    @classmethod
    def preprocess(self, img):

        img_array = np.array(img)
        mean_value=np.mean(img_array[img_array[...,0] > 40.0],axis=0)
        std_value=np.std(img_array[img_array[...,0] > 40.0],axis=0)
        img_array=(img_array-mean_value)/std_value

        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        
        img_array = img_array.transpose((2, 0, 1))

        return img_array


    def __getitem__(self, index):

        idx = self.ids[index]
        img_file = glob(self.imgs_dir_ + idx + '*')    

        img = Image.open(img_file[0]).resize(self.img_size)

        img = self.preprocess(img)

        return {
            'name': idx,
            'image': torch.from_numpy(img).type(torch.FloatTensor)
        }
