import argparse
import logging
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import f1_score 
from scripts.model import arch_select
from scripts.dataset import BasicDataset
import scripts.paired_transforms_tv04 as p_tr
from scripts.pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
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
        

def train_net(model_fl,
              csv_path,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              image_size=(512,512),
              save_cp=True,
              ):

    dir_checkpoint="./checkpoint_/{}/".format(args.model_structure)
    n_classes = args.n_class
    # create files
    if not os.path.isdir(dir_checkpoint):
        os.makedirs(dir_checkpoint)


    # define image processing and augmentation
    resize = p_tr.Resize(image_size)
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    rotate = p_tr.RandomRotation(degrees=60, fill=(0, 0, 0), fill_tg=0)
    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.10))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    #elastic = p_tr.RandomElastic(alpha=1, sigma=0.05)
    gaussblur = p_tr.RandomGaussBlur(radius=(0,1.1))
    # either translate, rotate, scale, elastic, gaussblur
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate, gaussblur])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = p_tr.Compose([resize,  scale_transl_rot, jitter, h_flip, tensorizer])
    

    train_dataset_all = BasicDataset(csv_path, transforms=train_transforms)
    n_val = int(len(train_dataset_all) * val_percent)
    n_train = len(train_dataset_all) - n_val
    train, val = random_split(train_dataset_all, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=batch_size, pin_memory=False)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    writer = SummaryWriter(comment=f'OUT_MODEL_{lr}_BS_{args.model_structure}')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(model_fl.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    

    if n_classes > 1:
        L_class_CE = nn.CrossEntropyLoss()
    else:
        L_class_CE = nn.BCEWithLogitsLoss()



    early_stop_path = dir_checkpoint + 'es_checkpoint.pth'
    early_stop = EarlyStopping(patience=15,verbose=True, path=early_stop_path)
    best_F1 =0.0
    best_loss = 100
    for epoch in range(epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model_fl.train()
            else:
                model_fl.eval()


            running_loss = 0
            running_corrects = 0
            running_f1_score = 0
            running_mse = 0
            running_iou = 0

            if phase == 'train':
                with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                    for batch in train_loader:

                        imgs = batch['image']
                        true_label = batch['label']

                        imgs = imgs.to(device=device, dtype=torch.float32)
                        label_type = torch.float32 if n_classes == 1 else torch.long
                        true_label = true_label.to(device=device, dtype=label_type)

                        if n_classes==1:
                            true_label = torch.unsqueeze(true_label, 1)

                        optimizer.zero_grad()

                        prediction = model_fl(imgs)

                        if n_classes==1:
                            prediction_sigmoid = torch.sigmoid(prediction)
                            prediction_decode=((prediction_sigmoid)>0.5).to(float)
                        else:
                            prediction_softmax = nn.Softmax(dim=1)(prediction)
                            _,prediction_decode = torch.max(prediction_softmax, 1)

                        loss_class_CE = L_class_CE(prediction, true_label)

                        loss_class = loss_class_CE 
                        loss_class.backward()

                        running_loss += loss_class.item() * imgs.size(0)
                        
                        running_corrects += torch.sum(prediction_decode == true_label.data)

                        optimizer.step()

                        pbar.set_postfix(**{'loss (batch)': loss_class.item()})

                        pbar.update(imgs.shape[0])

                epoch_loss = running_loss / n_train
                epoch_acc = running_corrects.double() / n_train / (imgs.shape[2]*imgs.shape[3])
                
                writer.add_scalar('Acc/training_acc', epoch_acc, epoch) 
                writer.add_scalar('Loss/training_loss', epoch_loss, epoch) 
                        
                print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))                
                        
            else:
                with torch.no_grad():
                    with tqdm(total=n_val, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                        for batch in val_loader:
                            imgs = batch['image']
                            true_label = batch['label']

                            imgs = imgs.to(device=device, dtype=torch.float32)
                            label_type = torch.float32 if n_classes == 1 else torch.long
                            true_label = true_label.to(device=device, dtype=label_type)
                            
                            if n_classes==1:
                                true_label = torch.unsqueeze(true_label, 1)

                            optimizer.zero_grad()

                            prediction = model_fl(imgs)

                            if n_classes==1:
                                prediction_sigmoid = torch.sigmoid(prediction)
                                prediction_decode=((prediction_sigmoid)>0.5).to(float)
                            else:
                                prediction_softmax = nn.Softmax(dim=1)(prediction)
                                _,prediction_decode = torch.max(prediction_softmax, 1)                                
                            
                            loss_class_CE = L_class_CE(prediction, true_label)

                            running_loss += loss_class_CE.item() * imgs.size(0)
                            running_corrects += torch.sum(prediction_decode == true_label.data)
                            f1_score_, mse_, iou_ = misc_measures(true_label.cpu().flatten(), prediction_decode.cpu().flatten()) * imgs.size(0)
                            running_f1_score += f1_score_
                            running_mse += mse_
                            running_iou += iou_

                    epoch_loss = running_loss / n_val
                    epoch_acc = running_corrects.double() / n_val / (imgs.shape[2]*imgs.shape[3])
                    epoch_f1 = running_f1_score / n_val

                    writer.add_scalar('Acc/val_acc', epoch_acc, epoch) 
                    writer.add_scalar('Loss/val_loss', epoch_loss, epoch) 
                    writer.add_scalar('F1/val_f1', epoch_f1, epoch) 

                    scheduler.step(epoch_loss)

                    print('Validation Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1)) 

                    early_stop(epoch_loss, model_fl)  

                    if early_stop.early_stop:
                        print('Early stopping')
                        return              
                    if best_F1<epoch_f1:
                        best_F1 = epoch_f1
                        try:
                            os.mkdir(dir_checkpoint)
                            logging.info('Created checkpoint directory')
                        except OSError:
                            pass
                        torch.save(model_fl.state_dict(), dir_checkpoint + f'best_checkpoint.pth')
                        logging.info(f'Checkpoint {epoch + 1} saved !')


    writer.close()



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=240,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=6,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=2e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    parser.add_argument( '-dir', '--train_csv_dir', metavar='tcd', type=str,
                        help='path to the csv', dest='train_dir')
    parser.add_argument( '-n', '--n_classes', dest='n_class', type=int, default=False,
                        help='number of class')
    parser.add_argument( '-d','--dataset', dest='dataset', type=str, 
                        help='dataset name')
    parser.add_argument( '-v', '--validation', dest='val', type=float, default=5.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument( '-r', '--round', dest='round', type=int, 
                        help='Number of round')   
    parser.add_argument( '-ms', '--model_structure', dest='model_structure', type=str, 
                        help='Model structure') 
    parser.add_argument('--seed_num', type=int, default=42, help='Validation split seed', dest='seed')
    parser.add_argument( '-s', '--image-size', dest='image_size', type=int, default=512,
                        help='size of image')   


    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    csv_path = args.train_dir
    img_size= (args.image_size,args.image_size)

    model_fl = arch_select(model_structure = args.model_structure, input_channels=3, n_classes=args.n_class)


    if args.load:
        model_fl.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    model_fl.to(device=device)
    
    print('Model parameter counting')
    pytorch_total_params = sum(p.numel() for p in model_fl.parameters())
    print('Total parameter is: ',pytorch_total_params)
    pytorch_total_params_t = sum(p.numel() for p in model_fl.parameters() if p.requires_grad)
    print('Trainable parameter is: ',pytorch_total_params_t)


    train_net(model_fl,
                csv_path,
                epochs=args.epochs,
                batch_size=args.batchsize,
                lr=args.lr,
                device=device,
                val_percent=args.val / 100,
                image_size=img_size)

