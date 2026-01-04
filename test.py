
import os, time
from operator import add
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch
import glob
from model import *
from utils import create_dir, seeding, rle_encode, rle_decode
from metrics import *
import argparse
from IRdataset import *

def get_args():
    parser = argparse.ArgumentParser(description="DFINet Test with Iterative Refinement")

    # ===== Dataset =====
    parser.add_argument("--data_path", type=str,
                        default="/home/visionx/EXT-4/lcj/new/SIRST-UAVB",
                        help="Dataset root path")

    # ===== Image =====
    parser.add_argument("--img_size", type=int, nargs=2,
                        default=[640, 512],
                        help="Input image size (W H)")

    # ===== Model =====
    parser.add_argument("--num_iter", type=int, default=8,
                        help="Number of iterative refinements")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/visionx/EXT-4/lcj/FIFLNet-3/uavb-update2/checkpoint.pth",
                        help="Model checkpoint path")

    # ===== Device =====
    parser.add_argument("--gpu", type=int, default=3,
                        help="GPU id")

    # ===== Output =====
    parser.add_argument("--save_csv", type=str,
                        default="/home/visionx/EXT-4/lcj/FIFLNet-3/uavb-update2/test_results.csv",
                        help="CSV file to save results")

    return parser.parse_args()

if __name__ == "__main__":
    
    """ Parse arguments """
    args = get_args()
    """ Seeding """
    seeding(42)

    """ Load dataset """
    path = args.data_path
    (train_x, train_y), (test_x, test_y) = load_data(path)
    """ Hyperparameters """
    size = tuple(args.img_size)
    num_iter = args.num_iter
    checkpoint_path = args.checkpoint

    """ Device """
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = DFINet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # model = CustomDataParallel(model).to(device)
    model.eval()
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()

    """ Testing """
    prev_masks = init_mask(test_x, size)
    save_data = []
    file = open(args.save_csv, "w")
    file.write("Iteration,mIoU,F1,Pd,Fa\n")

    for iter in range(num_iter):

        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # tmp_masks = []
        time_taken = []

        for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
            name1 = y.split("/")[-1].split(".")[0]
            ## Image
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            image = cv2.resize(image, size)
            img_x = image
            image = np.transpose(image, (2, 0, 1))
            image = image/255.0
            image = np.expand_dims(image, axis=0)
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            image = image.to(device)

            ## Mask
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, size)
            mask = np.expand_dims(mask, axis=0)
            mask = mask/255.0
            mask = np.expand_dims(mask, axis=0)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
            mask = mask.to(device)

            ## Prev mask
            pmask = prev_masks[i]
            pmask = " ".join(str(d) for d in pmask)
            pmask = str(pmask)
            pmask = rle_decode(pmask, size)
            pmask = np.expand_dims(pmask, axis=0)
            pmask = np.expand_dims(pmask, axis=0)
            pmask = pmask.astype(np.float32)
            # if iter == 0:
            pmask = np.transpose(pmask, (0, 1, 3, 2))
            pmask = torch.from_numpy(pmask)
            pmask = pmask.to(device)

            with torch.no_grad():
                """ FPS Calculation """
                start_time = time.time()
                pred_y = torch.sigmoid(model([image, pmask]))
                end_time = time.time() - start_time
                time_taken.append(end_time)
                eval_mIoU.update((pred_y>0.5).cpu(),mask.cpu())
                eval_PD_FA.update((pred_y[0,0,:,:]>0.5).cpu(),mask[0,0,:,:].cpu(),size)

                score = calculate_metrics(mask, pred_y, img_x)
                metrics_score = list(map(add, metrics_score, score))
               
                pred_y = pred_y[0][0].cpu().numpy()*255.0
                # if iter==7:
                #    cv2.imwrite(f'/home/LCH/FIFLNet-3/result-aug/{name1}.png', pred_y)# 保存图像文件
                pred_y=pred_y /255.0
                pred_y = pred_y > 0.5
                pred_y = np.transpose(pred_y, (0, 1))
                pred_y = np.array(pred_y, dtype=np.uint8)
                pred_y = rle_encode(pred_y)
                prev_masks[i] = pred_y

        """ Mean Metrics Score """
        results1=eval_mIoU.get()
        results2=eval_PD_FA.get()
        f1 = metrics_score[1]/len(test_x)
        recall = metrics_score[2]/len(test_x)
        precision = metrics_score[3]/len(test_x)
        f11=2*precision*recall /(recall+precision )


        """ Mean Time Calculation """
        mean_time_taken = np.mean(time_taken)
        # print("Mean Time Taken: ", mean_time_taken)
        mean_fps = 1/mean_time_taken
        print(f"pixAcc, mIoU: {results1} | "f"PD, FA: {results2} | F1: {f11:1.4f}")

        save_str = f"{iter+1},{results1[1]:1.4f},{f11:1.4f},{results2[0]:1.4f},{results2[1]:.3e}\n"
        file.write(save_str)