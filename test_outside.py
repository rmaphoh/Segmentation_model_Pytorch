import argparse
import logging
import os
import torch
import cv2
from skimage import io
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pycm import *
import matplotlib.pyplot as plt
from scripts.dataset import BasicDataset_outside
from scripts.model import arch_select
import scripts.paired_transforms_tv04 as p_tr
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize,remove_small_objects



def filter_frag(data_path):

    image_list=os.listdir(data_path)

    for i in sorted(image_list):
        img=io.imread(data_path + i, as_gray=True).astype(np.int64)
        img2=img>0
        img2 = remove_small_objects(img2, 20, connectivity=5)
        
        io.imsave(data_path + i , 255*(img2.astype('uint8')),check_contrast=False)

        skeleton = skeletonize(img2)
        
        if not os.path.isdir('../Results/M2/binary_skeleton/'):
            os.makedirs('../Results/M2/binary_skeleton/') 
        io.imsave('../Results/M2/binary_skeleton/' + i, 255*(skeleton.astype('uint8')),check_contrast=False)
        




def test_net(model_fl,
              csv_path,
              device,
              epochs=1,
              image_size=(512,512)
              ):

    storage_path ="../Results/M2/binary_vessel/".format(csv_path.split('/')[1])
    n_classes = args.n_class
    if not os.path.isdir(storage_path):
        os.makedirs(storage_path)

    test_dataset = BasicDataset_outside(imgs_dir=csv_path, img_size=image_size)
    n_test = len(test_dataset)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)


    for epoch in range(epochs):

        model_fl.eval()

        with tqdm(total=n_test, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in val_loader:
                imgs = batch['image']
                filename = batch['name'][0]

                imgs = imgs.to(device=device, dtype=torch.float32)

                prediction = model_fl(imgs)

                if n_classes==1:
                    prediction_sigmoid = torch.sigmoid(prediction)
                    prediction_decode=((prediction_sigmoid)>0.5).to(float)

                else:
                    prediction_softmax = nn.Softmax(dim=1)(prediction)
                    _,prediction_decode = torch.max(prediction_softmax, 1)     

                from torchvision.utils import save_image

                save_image(prediction_decode.to(torch.float32), storage_path+ filename+ '.png')

                pbar.update(imgs.shape[0])
    

    filter_frag(storage_path)


    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument( '-dir', '--test_csv_dir', metavar='tcd', type=str,
                        help='path to the csv', dest='test_dir')
    parser.add_argument( '-n', '--n_classes', dest='n_class', type=int, default=False,
                        help='number of class')
    parser.add_argument( '-m', '--model', dest='model_structure', type=str, 
                        help='Backbone of the model')     
    parser.add_argument( '-s', '--image-size', dest='image_size', type=int, 
                        help='size of image')   


    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    csv_path = args.test_dir
    img_size= (args.image_size,args.image_size)

    model_fl = arch_select(model_structure = args.model_structure, input_channels=3, n_classes=args.n_class)

    checkpoint_path = './checkpoint_/{}/best_checkpoint.pth'.format(args.model_structure) 
    
    model_fl.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    logging.info(f'Model loaded from {checkpoint_path}')

    model_fl.to(device=device)


    test_net(model_fl,
                csv_path,
                device=device,
                epochs=args.epochs,
                image_size=img_size)


