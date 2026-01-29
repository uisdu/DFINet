
import argparse
import os
import time
import datetime
import random
import numpy as np
from glob import glob
import albumentations as A
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from utils import *
from model import *
from loss import DiceBCELoss

def get_args():
    parser = argparse.ArgumentParser(description="DFINet Training Script")

    # Dataset
    parser.add_argument('--data_path', type=str, default='/home/LC/NUDT-SIRST/',
                        help='Path to dataset root')
    
    # Image
    parser.add_argument('--img_size', type=int, nargs=2, default=[256,256],
                        help='Input image size (H W)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.00007, help='Learning rate')

    # Directories / Output
    parser.add_argument('--save_dir', type=str, default='/home/LC/NUDT-SIRST', help='Directory to save checkpoints and logs')
    parser.add_argument('--checkpoint_name', type=str, default='/home/LC/NUDT-SIRST/checkpoint.pth', help='Checkpoint file name')
    parser.add_argument('--log_name', type=str, default='train_log.txt', help='Training log file name')

    # Device
    parser.add_argument('--gpu', type=int, default=1, help='GPU id to use')

    return parser.parse_args()

def train(model, loader, mask, optimizer, loss_fn, device):
    epoch_loss = 0
    return_mask = [] # 保存预测结果对应的 RLE 编码掩膜

    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        b, c, h, w  = y.shape
        m = []
        for k in range(b):
            idx = min(i*loader.batch_size + k, len(mask)-1)
            edata = mask[idx]
            edata = cv2.resize(edata, (w, h))
            edata = np.expand_dims(edata, axis=0)
            m.append(edata)

        m=np.stack(m,axis=0)
        m = torch.from_numpy(m).to(device, dtype=torch.float32)

        optimizer.zero_grad() # 清空梯度
        y_pred = model([x, m])  #前向传播
        loss = loss_fn(y_pred, y) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新模型参数

        with torch.no_grad():
            y_pred = torch.sigmoid(y_pred)     # [B,1,H,W]
            y_pred = y_pred.cpu().numpy()

            for py in y_pred:
                py = np.squeeze(py, axis=0)    # [H,W]
                py = py.astype(np.float32)     # soft mask
                return_mask.append(py)


        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss, return_mask

def evaluate(model, loader, mask_list, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    return_mask = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            B, _, H, W = y.shape

            m = []
            for k in range(B):
                idx = min(i * loader.batch_size + k, len(mask_list) - 1)
                edata = mask_list[idx]              # soft mask (H0,W0)
                edata = cv2.resize(edata, (W, H))
                edata = np.expand_dims(edata, axis=0)
                m.append(edata)

            m = np.stack(m, axis=0)                 # [B,1,H,W]
            m = torch.from_numpy(m).float().to(device)

            y_pred = model([x, m])
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            prob = torch.sigmoid(y_pred)
            prob = prob.cpu().numpy()

            for k, p in enumerate(prob):
                # ===== 保存 soft mask，用于下一轮 =====
                soft_mask = np.squeeze(p, axis=0)  # [H,W]
                return_mask.append(soft_mask.astype(np.float32))

    epoch_loss /= len(loader)
    return epoch_loss, return_mask

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('  + Number of parameters: %.4f M' % (param_count / 1e6))
    print('  + Number of parameters: %d' % (param_count ))
    return param_count

if __name__ == "__main__":

    args = get_args()
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir(args.save_dir)

    """ Training logfile """
    train_log_path = os.path.join(args.save_dir, args.log_name)
    if not os.path.exists(train_log_path):
        with open(train_log_path, "w") as f:
            f.write("\n")

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)

    """ Hyperparameters """
    size = tuple(args.img_size)
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr
    checkpoint_path = args.checkpoint_name

    """ Dataset """
    path = args.data_path
    (train_x, train_y), (valid_x, valid_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = IRDATASET(train_x, train_y, size, transform=transform)
    valid_dataset = IRDATASET(valid_x, valid_y, size, transform=None)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = DFINet()
    model = model.to(device)
    param = count_param(model) #计算参数量
    print(param/ 1e6)
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"

    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model. """
    best_valid_loss = float('inf')
    train_mask = init_mask(train_x, size)
    valid_mask = init_mask(valid_x, size)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, return_train_mask = train(model, train_loader, train_mask, optimizer, loss_fn, device)
        valid_loss, return_valid_mask = evaluate(model, valid_loader, valid_mask, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            data_str = f"Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)
            torch.save(model.state_dict(), checkpoint_path)

            train_mask = return_train_mask
            # valid_mask = return_valid_mask

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print_and_save(train_log_path, data_str)
