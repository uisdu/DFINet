import os
import time
import datetime
import argparse
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from model import *
from IRdataset import *
from loss import DiceBCELoss


def get_args():
    parser = argparse.ArgumentParser(description="DFINet Training")

    parser.add_argument("--save_path", type=str, default="/home/visionx/EXT-4/lcj/FIFLNet-3/")
    parser.add_argument("--exp_name", type=str, default="NUDT-SRIST")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume",type=str,default="",help="resume training")


    # ===== data =====
    parser.add_argument("--data_path", type=str,default="/home/visionx/EXT-4/lcj/new/NUDT-SIRST/")
    parser.add_argument("--img_size", nargs=2, type=int, default=[256, 256], help="IRSTD-1k: 512, NUDT,Aug:256")

    # ===== train params =====
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=7e-5, help="IRSTD-1k:7e-5, SIRST-Aug:0.0001")
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()



def train(model, loader, mask, optimizer, loss_fn, device, size):
    model.train()
    epoch_loss = 0
    return_mask = []

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        b = y.shape[0]
        m = []
        for edata in mask[i*b: i*b+b]:
            edata = " ".join(map(str, edata))
            edata = rle_decode(edata, size)
            m.append(np.expand_dims(edata, 0))

        m = torch.from_numpy(np.array(m)).permute(0, 1, 3, 2).float().to(device)

        optimizer.zero_grad()
        y_pred = model([x, m])
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = torch.sigmoid(y_pred).cpu().numpy()
            for py in y_pred:
                py = (py.squeeze(0) > 0.5).astype(np.uint8)
                return_mask.append(rle_encode(py))

        epoch_loss += loss.item()

    return epoch_loss / len(loader), return_mask

def evaluate(model, loader, mask, loss_fn, device, size):
    model.eval()
    epoch_loss = 0
    return_mask = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            b = y.shape[0]
            m = []
            for edata in mask[i*b: i*b+b]:
                edata = " ".join(map(str, edata))
                edata = rle_decode(edata, size)
                m.append(np.expand_dims(edata, 0))

            m = torch.from_numpy(np.array(m)).permute(0, 1, 3, 2).float().to(device)

            y_pred = model([x, m])
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred = torch.sigmoid(y_pred).cpu().numpy()
            for py in y_pred:
                py = (py.squeeze(0) > 0.5).astype(np.uint8)
                return_mask.append(rle_encode(py))

    return epoch_loss / len(loader), return_mask

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('  + Number of parameters: %.4f M' % (param_count / 1e6))
    print('  + Number of parameters: %d' % (param_count ))
    return param_count

def main():
    args = get_args()

    device = torch.device(f"cuda:{args.gpu}")
    save_dir = os.path.join(args.save_path,args.exp_name)
    create_dir(save_dir)

    log_path = os.path.join(save_dir, "train_log.txt")
    print_and_save(log_path, str(datetime.datetime.now()))
    
    (train_x, train_y), (valid_x, valid_y) = load_data(args.data_path)
    train_x, train_y = shuffling(train_x, train_y)

    print_and_save(
        log_path,
        f"Dataset:\nTrain: {len(train_x)} | Valid: {len(valid_x)}"
    )

    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.2,num_holes_range=(1, 4),hole_height_range=(4, 16),hole_width_range=(4, 16),)
    ])

    size = tuple(args.img_size)

    train_dataset = IRDATASET(train_x, train_y, size, transform)
    valid_dataset = IRDATASET(valid_x, valid_y, size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    model = DFINet().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, verbose=True
    )

    loss_fn = DiceBCELoss()

    # ===== train =====
    best_valid_loss = float("inf")
    train_mask = init_mask(train_x, size)
    valid_mask = init_mask(valid_x, size)

    ckpt_path = os.path.join(save_dir, "checkpoint.pth")

    for epoch in range(args.epochs):
        start = time.time()

        train_loss, train_mask = train(
            model, train_loader, train_mask, optimizer, loss_fn, device, size
        )
        valid_loss, valid_mask = evaluate(
            model, valid_loader, valid_mask, loss_fn, device, size
        )

        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)
            print_and_save(log_path, f"Saved checkpoint @ epoch {epoch+1}")

        mins, secs = epoch_time(start, time.time())
        print_and_save(
            log_path,
            f"Epoch {epoch+1:03d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {valid_loss:.4f} | "
            f"Time: {mins}m {secs}s"
        )

if __name__ == "__main__":
    main()
