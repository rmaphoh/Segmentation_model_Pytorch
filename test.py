import argparse
import logging
import os
import torch
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pycm import *
import matplotlib.pyplot as plt
from scripts.dataset import BasicDataset
from scripts.model import arch_select
import scripts.paired_transforms_tv04 as p_tr
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, mean_squared_error 


def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm=confusion_matrix(true_vessel_arr, pred_vessel_arr)
    mse = mean_squared_error(true_vessel_arr, pred_vessel_arr)

    try:
        sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
        precision=1.*cm[1,1]/(cm[1,1]+cm[0,1])
        F1_score = 2*precision*sensitivity/(precision+sensitivity)
        iou = 1.*cm[1,1]/(cm[1,0]+cm[0,1]+cm[1,1])
        return F1_score, mse, iou
    
    except:

        return 0,0,0


def test_net(model_fl,
              csv_path,
              device,
              epochs=1,
              image_size=(512,512)
              ):

    storage_path ="./Results/{}/".format(csv_path.split('/')[1])
    n_classes = args.n_class
    if not os.path.isdir(storage_path):
        os.makedirs(storage_path)

    resize = p_tr.Resize(image_size)
    tensorizer = p_tr.ToTensor()
    val_transforms = p_tr.Compose([resize, tensorizer])

    test_dataset = BasicDataset(csv_path, transforms=val_transforms)
    n_test = len(test_dataset)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)

    running_f1_score = 0
    running_mse = 0
    running_iou = 0
    running_corrects = 0


    for epoch in range(epochs):

        model_fl.eval()

        with tqdm(total=n_test, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in val_loader:
                imgs = batch['image']
                true_label = batch['label']
                filename = batch['name'][0]

                imgs = imgs.to(device=device, dtype=torch.float32)
                label_type = torch.float32 if n_classes == 1 else torch.long
                true_label = true_label.to(device=device, dtype=label_type)

                prediction = model_fl(imgs)
                prediction_softmax = nn.Softmax(dim=1)(prediction)
                _,prediction_decode = torch.max(prediction_softmax, 1)

                from torchvision.utils import save_image

                save_image(prediction_decode.to(torch.float32), storage_path+ filename+ '.png')
                #cv2.imwrite(storage_path+ filename+ '.png', np.float32(prediction_decode)*255)

                running_corrects += torch.sum(prediction_decode == true_label.data)
                f1_score_, mse_, iou_ = misc_measures(true_label.cpu().flatten(), prediction_decode.cpu().flatten()) * imgs.size(0)
                running_f1_score += f1_score_
                running_mse += mse_
                running_iou += iou_

                pbar.update(imgs.shape[0])

        epoch_acc = running_corrects.double() / n_test / (true_label.shape[1]*true_label.shape[2])
        epoch_f1 = running_f1_score / n_test 
        epoch_mse = running_mse / n_test 
        epoch_iou = running_iou / n_test

        print('Sklearn Testing Metrics - Acc: {:.4f} F1-score: {:.4f} MSE: {:.4f} IoU: {:.4f}'.format(epoch_acc, epoch_f1, epoch_mse, epoch_iou)) 

    




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


