
import os, time
from operator import add
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import glob
from model import *
from utils import create_dir, seeding, init_mask, rle_encode, rle_decode, load_data,load_data1
from metrics import *
import matplotlib.pyplot as plt

def precision_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2_score(y_true, y_pred, beta=2):
    p = precision_score(y_true,y_pred)
    r = recall_score(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def calculate_metrics(y_true, y_pred, img):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_fbeta = F2_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    confusion = confusion_matrix(y_true, y_pred)
    # if float(confusion[0,0] + confusion[0,1]) != 0:
    #     score_specificity = float(confusion[0,0]) / float(confusion[0,0] + confusion[0,1])
    # else:
    score_specificity = 0.0

    return [score_jaccard, score_f1, score_recall, score_precision, score_specificity, score_acc, score_fbeta]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

# class CustomDataParallel(torch.nn.DataParallel):
# 	""" A Custom Data Parallel class that properly gathers lists of dictionaries. """
# 	def gather(self, outputs, output_device):
# 		# Note that I don't actually want to convert everything to the output_device
# 		return sum(outputs, [])
# def hook_fn(module, input, output):
#     feature_maps.append(output)
    # 定义 Forward Hook
# def hook_before_r2(module, input, output):
#     global activation_before_r2
#     activation_before_r2 = output  # r1 的输出

# def hook_after_r2(module, input, output):
#     global activation_after_r2
#     activation_after_r2 = output  # r2 的输出
# def save_feature_map(feature_map, save_path):
#     """ 保存单张特征图（通道加权平均） """
#     if isinstance(feature_map, torch.Tensor):
#         feature_map = feature_map.detach().cpu().numpy()  # 确保 tensor 转 numpy

#     feature_map = np.mean(feature_map, axis=0)  # 对通道求均值
#     eps = 1e-8  # 防止除 0
#     feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + eps)  # 归一化

#     heatmap = cv2.applyColorMap(np.uint8(255 * feature_map), cv2.COLORMAP_JET)  # 伪彩色

#     cv2.imwrite(save_path, heatmap)  # 保存图像
def forward_hook_after_d4(module, inp, outp):
    """ 记录 d4 层的输出（r2 层的输入） """
    global activation_after_d4
    activation_after_d4 = outp

def forward_hook_after_d3(module, inp, outp):
    """ 记录 d3 层的输出 """
    global activation_after_d3
    activation_after_d3 = outp

def forward_hook_after_d2(module, inp, outp):
    """ 记录 d2 层的输出 """
    global activation_after_d2
    activation_after_d2 = outp

def forward_hook_after_d1(module, inp, outp):
    """ 记录 d1 层的输出 """
    global activation_after_d1
    activation_after_d1 = outp



def backward_hook_after_d4(module, grad_in, grad_out):
    """ 记录 r1 进入 r2 的梯度 """
    global gradient_after_d4
    gradient_after_d4 = grad_out[0]
    # print(f"[DEBUG] Gradient d4 shape: {gradient_after_d4.shape}")

def backward_hook_after_d3(module, grad_in, grad_out):
    """ 记录 r2 的梯度 """
    global gradient_after_d3
    gradient_after_d3 = grad_out[0]
def backward_hook_after_d2(module, grad_in, grad_out):
    """ 记录 r2 的梯度 """
    global gradient_after_d2
    gradient_after_d2 = grad_out[0]

def backward_hook_after_d1(module, grad_in, grad_out):
    """ 记录 r2 的梯度 """
    global gradient_after_d1
    gradient_after_d1 = grad_out[0]


def compute_gradcam(activation, gradient):
    """ 计算 Grad-CAM 热力图 """
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)  # GAP 操作
    grad_cam_map = F.relu(torch.sum(weights * activation, dim=1)).squeeze().cpu().detach().numpy()
    
    # 归一化到 0-1 之间
    grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
    return grad_cam_map


def save_gradcam(image, grad_cam_map, save_path):
    """ 生成 Grad-CAM 可视化并保存 """
    grad_cam_map = cv2.resize(grad_cam_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR) # 调整大小
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map), cv2.COLORMAP_JET)  # 伪彩色
    # overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)  # 叠加原图
    cv2.imwrite(save_path, heatmap)




if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Load dataset """
    path = "/home/LCH/IRSTD-1k-80_20/"
    (train_x, train_y), (test_x, test_y) = load_data1(path)

    """ Hyperparameters """
    size = (512, 512)
    num_iter = 5
    checkpoint_path = "/home/LCH/FIFLNet-3/files-1k-0.4/checkpoint.pth"

    """ Directories """
    # create_dir("files-1k-snr4.73")

    

    """ Load the checkpoint """
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # feature_maps = []

    model = FANet()

    # 选择目标层
    target_layer_after_d4 = model.d4
    target_layer_after_d3 = model.d4.r2
    target_layer_after_d2 = model.d2.b1
    target_layer_after_d1 = model.d1.r2





    # 注册 Forward & Backward Hook
    # target_layer_before_d4.register_forward_hook(forward_hook_before_r2)
    target_layer_after_d4.register_forward_hook(forward_hook_after_d4)
    # target_layer_before_d2.register_backward_hook(backward_hook_before_r2)
    target_layer_after_d3.register_forward_hook(forward_hook_after_d3)
    target_layer_after_d2.register_forward_hook(forward_hook_after_d2)
    target_layer_after_d1.register_forward_hook(forward_hook_after_d1)

    target_layer_after_d4.register_backward_hook(backward_hook_after_d4)
    target_layer_after_d3.register_backward_hook(backward_hook_after_d3)
    target_layer_after_d2.register_backward_hook(backward_hook_after_d2)
    target_layer_after_d1.register_backward_hook(backward_hook_after_d1)



  
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # model = CustomDataParallel(model).to(device)
    model.eval()
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()

    """ Testing """
    prev_masks = init_mask(test_x, size)
    save_data = []
    file = open("files1k/test_results.csv", "w")
    file.write("Iteration,mIoU,F1,Recall,Precision,Pd,Fa,Mean Time,Mean FPS\n")

    for iter in range(num_iter):

        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tmp_masks = []
        time_taken = []
        # filenames = sorted(glob.glob('/home/LCH/IRSTD-FZU/Dataset/irst_6b2b2_dataset20240321/NUDT-SIRST/test/masks/*.png'))
        # filenames = [x.split('/')[-1] for x in filenames]

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
            if iter == 0:
                pmask = np.transpose(pmask, (0, 1, 3, 2))
            pmask = torch.from_numpy(pmask)
            pmask = pmask.to(device)

            # with torch.no_grad():
            """ FPS Calculation """
            start_time = time.time()
            # pred_y = torch.sigmoid(model([image, pmask]))
            pred_y = model([image, pmask])
            end_time = time.time() - start_time

            # **计算 Grad-CAM**
            model.zero_grad()
            loss = pred_y.mean()  # 确保是标量
            loss.backward(retain_graph=True)  # 计算梯度

            grad_cam_map_after_d4 = compute_gradcam(activation_after_d4, gradient_after_d4)
            grad_cam_map_after_d3 = compute_gradcam(activation_after_d3, gradient_after_d3)
            grad_cam_map_after_d2= compute_gradcam(activation_after_d2, gradient_after_d2)
            grad_cam_map_after_d1= compute_gradcam(activation_after_d1, gradient_after_d1)
           

            # **保存 Grad-CAM 结果**
            save_gradcam(img_x,  grad_cam_map_after_d4, f"/home/LCH/FIFLNet-3/result-D1D4/{name1}_after2_d4.png")
            # save_gradcam(img_x,  grad_cam_map_after_d3, f"/home/LCH/FIFLNet-3/result-D1D4/{name1}_afterm_d3.png")
            save_gradcam(img_x,  grad_cam_map_after_d2, f"/home/LCH/FIFLNet-3/result-D1D4/{name1}_after2_d2.png")
            save_gradcam(img_x,  grad_cam_map_after_d1, f"/home/LCH/FIFLNet-3/result-D1D4/{name1}_after2_d1.png")

            pred_y=torch.sigmoid(pred_y)
           
             
            time_taken.append(end_time)
            eval_mIoU.update((pred_y>0.5).cpu(),mask.cpu())
            eval_PD_FA.update((pred_y[0,0,:,:]>0.5).cpu(),mask[0,0,:,:].cpu(),size)

            # score = calculate_metrics(mask, pred_y, img_x)
            # metrics_score = list(map(add, metrics_score, score))
               
            pred_y = pred_y[0][0].detach().cpu().numpy()*255.0
            cv2.imwrite("/home/LCH/FIFLNet-3/vis1.png",pred_y)


            # if iter==8:
                #  cv2.imwrite(f'/home/LCH/FIFLNet-3/results-1k-roc/{name1}.png', pred_y)# 保存图像文件
            pred_y=pred_y /255.0
            pred_y = pred_y > 0.5
            pred_y = np.transpose(pred_y, (1,0))
            pred_y = np.array(pred_y, dtype=np.uint8)
            pred_y = rle_encode(pred_y)
            prev_masks[i] = pred_y
            tmp_masks.append(pred_y)

    #     """ Mean Metrics Score """
    #     results1=eval_mIoU.get()
    #     results2=eval_PD_FA.get()
    #     f1 = metrics_score[1]/len(test_x)
    #     recall = metrics_score[2]/len(test_x)
    #     precision = metrics_score[3]/len(test_x)
    #     f11=2*precision*recall /(recall+precision )


    #     """ Mean Time Calculation """
    #     mean_time_taken = np.mean(time_taken)
    #     print("Mean Time Taken: ", mean_time_taken)
    #     mean_fps = 1/mean_time_taken
    #     print("pixAcc, mIoU:\t" + str(results1))
    #     print("PD, FA:\t" + str(results2))
    #     print(f"F1: {f11:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} -Mean Time: {mean_time_taken:1.7f} - Mean FPS: {mean_fps:1.7f}")

    #     save_str = f"{iter+1},{results1[1]:1.4f},{f11:1.4f},{recall:1.4f},{precision:1.4f},{results2[0]:1.4f},{results2[1]:1.7f},{mean_time_taken:1.7f},{mean_fps:1.7f}\n"
    #     file.write(save_str)

    #     save_data.append(tmp_masks)
    # # save_data = np.array(save_data)

    # """ Saving the masks. """
    # for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
    #     image = cv2.imread(x, cv2.IMREAD_COLOR)
    #     image = cv2.resize(image, size)

    #     mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    #     mask = cv2.resize(mask, size)
    #     # mask = mask / 255
    #     # mask = (mask > 0.5) * 255
    #     mask = mask_parse(mask)

    #     name = y.split("/")[-1].split(".")[0]
    #     sep_line = np.ones((size[0], 10, 3)) * 128
    #     tmp = [image, sep_line, mask]

    #     for data in save_data:
    #         tmp.append(sep_line)
    #         d = data[i]
    #         d = " ".join(str(z) for z in d)
    #         d = str(d)
    #         d = rle_decode(d, size)
    #         d = d * 255
    #         d = mask_parse(d)

    #         tmp.append(d)

    #     cat_images = np.concatenate(tmp, axis=1)
    #     cv2.imwrite(f"/home/LCH/FIFLNet-3/results-1k-duibi/{name}.png", cat_images)
