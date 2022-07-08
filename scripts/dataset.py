import torch
import logging
from glob import glob
from os import listdir
from PIL import Image
from os.path import splitext
from torch.utils.data import Dataset



class BasicDataset(Dataset):
    def __init__(self, imgs_dir, transforms):
        self.imgs_dir_ = imgs_dir + 'images/'
        self.label_dir = imgs_dir + 'labels/'
        self.transform = transforms
        self.ids = [splitext(file)[0] for file in listdir(self.imgs_dir_)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        idx = self.ids[index]
        label_file = glob(self.label_dir + idx  + '*')
        img_file = glob(self.imgs_dir_ + idx + '*')    

        label = Image.open(label_file[0])
        img = Image.open(img_file[0])


        if self.transform is not None:
            img, label = self.transform(img, label)

        label[label==0]=0
        label[label==255]=1

        return {
            'name': idx,
            'image': img.type(torch.FloatTensor),
            'label': label.type(torch.FloatTensor)
        }


    
class BasicDataset_pipeline(Dataset):
    def __init__(self, imgs_dir, transforms):
        self.imgs_dir = imgs_dir + 'images/'
        self.transform = transforms
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):

        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')         

        img = Image.open(img_file[0])
        ori_height,ori_width = img.size
        
        if self.transform is not None:
            img = self.transform(img)
        
        return {
            'name': idx,
            'width': ori_width,
            'height': ori_height,
            'image': img.type(torch.FloatTensor)
        }
