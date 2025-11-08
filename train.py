import cv2
import torch


import torchvision

from model import *
import os
from PIL import Image
import numpy as np
import glob

def cut_feature(feature):
    feature_light = feature[:,:96]
    feature_content = feature[:,96:]
    return feature_light, feature_content
def match_feature(feature1, feature2):
    new_feature = torch.cat([feature1, feature2],1)
    return new_feature


def _torch_fspecial_gauss(size, sigma, device=None):
    """Mimic the 'fspecial' Gaussian function from MATLAB and TF version"""
    # Create coordinate grid
    coords = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=device)
    x_data, y_data = torch.meshgrid(coords, coords, indexing='ij')

    g = torch.exp(-((x_data ** 2 + y_data ** 2) / (2.0 * sigma ** 2)))
    g = g / g.sum()
    # reshape to [1, 1, size, size] for conv2d kernel
    g = g.unsqueeze(0).unsqueeze(0)
    return g
def torch_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    """
    PyTorch equivalent of TensorFlow's tf_ssim.
    Args:
        img1, img2: tensors of shape (B, C, H, W), with values in [0, 1]
        cs_map: if True, returns both SSIM and contrast-structure map
        mean_metric: if True, returns mean SSIM value; else full map
        size: Gaussian window size
        sigma: Gaussian window std
    """
    device = img1.device
    b, c, h, w = img1.shape

    # Prepare Gaussian window
    window = _torch_fspecial_gauss(size, sigma, device=device)
    window = window.expand(c, 1, size, size)

    # Constants
    K1 = 0.01
    K2 = 0.03
    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # VALID convolution (no padding)
    mu1 = F.conv2d(img1, window, stride=1, padding=0, groups=c)
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=c)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, stride=1, padding=0, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, stride=1, padding=0, groups=c) - mu1_mu2

    if cs_map:
        ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))
        cs = ((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        if mean_metric:
            return ssim_map.mean(), cs.mean()
        else:
            return ssim_map, cs
    else:
        ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))
        if mean_metric:
            return ssim_map.mean()
        else:
            return ssim_map
def compute_ssim(img1, img2):
    ssim_r = torch_ssim(img1[:, 0:1, :, :], img2[:, 0:1, :, :])
    ssim_g = torch_ssim(img1[:, 1:2, :, :], img2[:, 1:2, :, :])
    ssim_b = torch_ssim(img1[:, 2:3, :, :], img2[:, 2:3, :, :])
    return (ssim_r + ssim_g + ssim_b) / 3.0


def restoration_loss(input_high, output_high):
    loss_restoration = torch.mean(torch.abs(input_high - output_high))
    return loss_restoration


def instance_normalization(x, eps=1e-5):
    m = x.mean(dim=1, keepdim=True)
    v = x.var(dim=1, unbiased=False, keepdim=True)
    return (x - m) / torch.sqrt(v + eps)

def feature_loss(low_content, output_content, low_light, high_light, output_light):
    # Normalize lighting features
    low_light_norm = instance_normalization(low_light)
    high_light_norm = instance_normalization(high_light)
    output_light_norm = instance_normalization(output_light)

    # Positive and negative distances
    d_ap = torch.mean((high_light_norm - output_light_norm) ** 2)
    d_an = torch.mean((low_light_norm - output_light_norm) ** 2)

    # Triplet-like margin loss on light features
    margin = 0.08
    loss_light_feature = torch.maximum(d_ap - d_an + margin, torch.tensor(0.0, device=d_ap.device))

    # Content reconstruction (L2 loss)
    loss_content_feature = torch.mean((low_content - output_content) ** 2)

    return loss_light_feature + loss_content_feature


def rgb2hsv(img):
    if isinstance(img, torch.Tensor):
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        maxc, _ = img.max(dim=1, keepdim=True)
        minc, _ = img.min(dim=1, keepdim=True)
        v = maxc
        delt = (maxc - minc).clamp(min=1e-8)
        s = delt / (maxc + 1e-8)
        h = torch.zeros_like(maxc)
        mask = (maxc == r)
        h[mask] = ((g - b)[mask] / delt[mask]) % 6
        mask = (maxc == g)
        h[mask] = (2.0 + (b - r)[mask] / delt[mask]) % 6
        mask = (maxc == b)
        h[mask] = (4.0 + (r - g)[mask] / delt[mask]) % 6
        h = (h / 6.0) % 1.0
        return torch.cat([h, s, v], dim=1)
    else:
        raise NotImplementedError("rgb2hsv placeholder expects torch tensors")
def content_loss(input_low, output_high):

    # Convert to HSV
    input_low_hsv = rgb2hsv(input_low)
    output_high_hsv = rgb2hsv(output_high)

    # Split channels
    input_h, input_s, input_v = torch.chunk(input_low_hsv, 3, dim=1)
    output_h, output_s, output_v = torch.chunk(output_high_hsv, 3, dim=1)

    B = input_low.shape[0]

    # Flatten spatial dimensions
    input_h = input_h.view(B, -1)
    output_h = output_h.view(B, -1)
    input_s = input_s.view(B, -1)
    output_s = output_s.view(B, -1)

    # Normalize along feature dimension (like tf.nn.l2_normalize)
    norm_input_h = F.normalize(input_h, p=2, dim=-1)
    norm_output_h = F.normalize(output_h, p=2, dim=-1)
    norm_input_s = F.normalize(input_s, p=2, dim=-1)
    norm_output_s = F.normalize(output_s, p=2, dim=-1)

    # Cosine distance = 1 - cosine similarity
    loss_h = torch.mean(1 - F.cosine_similarity(norm_input_h, norm_output_h, dim=-1))
    loss_s = torch.mean(1 - F.cosine_similarity(norm_input_s, norm_output_s, dim=-1))

    # Final averaged loss
    return (loss_h + loss_s) / 2.0
def augmentation(img_arr, mode):
    # img_arr: numpy HxWxC [0,1]
    img = Image.fromarray((img_arr * 255).astype('uint8'))
    if mode == 0:
        out = img
    elif mode == 1:
        out = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == 2:
        out = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif mode == 3:
        out = img.rotate(90)
    return np.asarray(out).astype(np.float32) / 255.0
def load_images(path):
    im = Image.open(path).convert('RGB')
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr
def compute_psnr(a, b, maxval=1.0):
    # supports numpy BxHxWxC
    mse = np.mean((a - b) ** 2, axis=(1, 2, 3))
    return 10.0 * np.log10((maxval ** 2) / (mse + 1e-8))
def computer_error(a, b):
    a = np.asarray(a).ravel();
    b = np.asarray(b).ravel()
    mean = np.mean(a - b);
    var = np.var(a - b);
    mse = np.mean((a - b) ** 2)
    return mean, var, mse
def save_images(path, arr):
    # arr: numpy BxHxWxC or HxWxC
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


input_low = torch.rand((1,3,256,256))
input_high = torch.rand((1,3,256,256))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8
change_patch_size = 100


train_low_glob = '../Low_Light_Image_Enhancement/LOLdataset/train/low/'
train_high_glob = '../Low_Light_Image_Enhancement/LOLdataset/train/high/'
eval_low_glob = '../Low_Light_Image_Enhancement/LOLdataset/val/low/'
eval_high_glob = '../Low_Light_Image_Enhancement/LOLdataset/val/high/'


sample_dir = './lol_nocut_96cos_changeda2/'
checkpoint_dir = './checkpoint/lol_nocut_96cos_changeda2/'
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

epoch_num = 1000
learning_rate = 1e-4
eval_every_epoch = 100

enc = EncoderNet().to(device)
dec = DecoderNet().to(device)

params = list(enc.parameters()) + list(dec.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

train_low_names = [train_low_glob+file for file in sorted( os.listdir(train_low_glob))]
train_high_names =[train_high_glob +file for file in sorted(os.listdir(train_high_glob))]
eval_low_names = [eval_low_glob+file for file in sorted(os.listdir(eval_low_glob))]
eval_high_names = [eval_high_glob+file for file in sorted(os.listdir(eval_high_glob))]



train_low_data = [load_images(p) for p in train_low_names]
train_high_data = [load_images(p) for p in train_high_names]
eval_low_data = [load_images(p) for p in eval_low_names]
eval_high_data = [load_images(p) for p in eval_high_names]


numBatch = len(train_low_data) // int(batch_size)


ckpt_path = os.path.join(checkpoint_dir, 'model.pth')

if os.path.exists(ckpt_path):
    print("[*] Loading checkpoint:", ckpt_path)
    st = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(st['enc'])
    dec.load_state_dict(st['dec'])
    print("[*] Loaded checkpoint.")

psnr_for_save_epoch = []
ssim_for_save_epoch = []
mean_l_save_epoch = []
mean_c_save_epoch = []
var_l_save_epoch = []
var_c_save_epoch = []
mse_l_save_epoch = []
mse_c_save_epoch = []
import time
import random

start_epoch = 0
image_id = 0
print("[*] Start training from epoch", start_epoch)

for epoch in range(start_epoch, epoch_num):
    start_time = time.time()

    tmp = list(zip(train_low_data, train_high_data))

    random.shuffle(tmp)
    train_low_data, train_high_data = zip(*tmp)
    train_low_data = list(train_low_data)
    train_high_data = list(train_high_data)

    for batch_id in range(numBatch):
        B_H, B_W = 400, 600
        batch_input_low = np.zeros((batch_size, B_H, B_W, 3), dtype=np.float32)
        batch_input_high = np.zeros((batch_size, B_H, B_W, 3), dtype=np.float32)
        for patch_id in range(batch_size):
            img_low = train_low_data[image_id]
            img_high = train_high_data[image_id]

            if img_low.shape[0] < B_H or img_low.shape[1] < B_W:
                img_low = np.array(Image.fromarray((img_low * 255).astype('uint8')).resize((B_W, B_H))).astype(
                    np.float32) / 255.0
                img_high = np.array(Image.fromarray((img_high * 255).astype('uint8')).resize((B_W, B_H))).astype(
                    np.float32) / 255.0

            rand_mode = random.randint(0, 3)
            low_aug = augmentation(img_low, rand_mode)
            high_aug = augmentation(img_high, rand_mode)

            h, w, _ = low_aug.shape
            x = random.randint(0, max(0, h - change_patch_size))
            y = random.randint(0, max(0, w - change_patch_size))
            z = random.uniform(0, 1.0)

            change_patch = z * low_aug[x:x + change_patch_size, y:y + change_patch_size, :] + (1.0 - z) * high_aug[
                                                                                                          x:x + change_patch_size,
                                                                                                          y:y + change_patch_size,
                                                                                                          :]
            cur_low = low_aug.copy()
            cur_high = high_aug.copy()
            cur_low[x:x + change_patch_size, y:y + change_patch_size, :] = change_patch

            batch_input_low[patch_id] = cur_low
            batch_input_high[patch_id] = cur_high

            image_id = (image_id + 1) % len(train_low_data)

        inp_low = torch.from_numpy(batch_input_low).permute(0, 3, 1, 2).to(device)
        inp_high = torch.from_numpy(batch_input_high).permute(0, 3, 1, 2).to(device)

        conv1, conv2, conv3, conv4, feature1 = enc(inp_low)
        _, _, _, _, feature2 = enc(inp_high)
        low_feature_light, low_feature_content = cut_feature(feature1)
        high_feature_light, high_feature_content = cut_feature(feature2)
        feature_new = match_feature(high_feature_light, low_feature_content)
        output_high = dec(conv1, conv2, conv3, conv4, feature_new)

        _, _, _, _, feature_output = enc(output_high)
        output_feature_light, output_feature_content = cut_feature(feature_output)

        loss_rest = restoration_loss(output_high, inp_high)
        loss_feat = feature_loss(low_feature_content, output_feature_content,
                                 low_feature_light, high_feature_light, output_feature_light)
        loss_cont = content_loss(inp_low, output_high)
        train_loss = loss_rest + 2.0 * loss_feat + 2.0 * loss_cont

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        out_np = output_high.detach().cpu().numpy().transpose(0, 2, 3, 1)
        cv2.imwrite('out.jpg',out_np[0]*255)
        high_np = inp_high.detach().cpu().numpy().transpose(0, 2, 3, 1)
        train_psnr = np.mean(compute_psnr(out_np, high_np))
        #train_ssim = compute_ssim(out_np, high_np) if callable(compute_ssim) else 0.0

        print(
            "Epoch: [%2d] [%4d/%4d] time: %.4f, restoration_loss: %.6f, content_loss: %.6f, total_loss: %.6f, psnr: %.6f" %
            (epoch + 1, batch_id + 1, numBatch, time.time() - start_time,
             loss_rest.item(), loss_cont.item(), train_loss.item(), train_psnr))

    if (epoch + 1) % eval_every_epoch == 0:
        print("[*] Evaluating for epoch %d..." % (epoch + 1))
        psnr_for_eval_epoch = []
        ssim_for_eval_epoch = []
        mean_l_eval_epoch = []
        mean_c_eval_epoch = []
        var_l_eval_epoch = []
        var_c_eval_epoch = []
        mse_l_eval_epoch = []
        mse_c_eval_epoch = []
        for idx in range(len(eval_low_data)):
            name = os.path.splitext(os.path.basename(eval_low_names[idx]))[0]
            input_low_eval = torch.from_numpy(np.expand_dims(eval_low_data[idx], 0).transpose(0, 3, 1, 2)).to(
                device).float()
            input_high_eval = torch.from_numpy(np.expand_dims(eval_high_data[idx], 0).transpose(0, 3, 1, 2)).to(
                device).float()
            conv1_e, conv2_e, conv3_e, conv4_e, feat_low = enc(input_low_eval)
            _, _, _, _, feat_high = enc(input_high_eval)
            low_feature_light, low_feature_content = cut_feature(feat_low)
            high_feature_light, high_feature_content = cut_feature(feat_high)
            feature_new = match_feature(high_feature_light, low_feature_content)
            output = dec(conv1_e, conv2_e, conv3_e, conv4_e, feature_new)

            eval_psnr = np.mean(compute_psnr(output.detach().cpu().numpy().transpose(0, 2, 3, 1),
                                             input_high_eval.detach().cpu().numpy().transpose(0, 2, 3, 1)))
            """eval_ssim = compute_ssim(output.detach().cpu().numpy().transpose(0, 2, 3, 1),
                                     input_high_eval.detach().cpu().numpy().transpose(0, 2, 3, 1)) if callable(
                compute_ssim) else 0.0"""

            mean_l, var_l, mse_l = computer_error(low_feature_light.detach().cpu().numpy(),
                                                  high_feature_light.detach().cpu().numpy())
            mean_c, var_c, mse_c = computer_error(low_feature_content.detach().cpu().numpy(),
                                                  high_feature_content.detach().cpu().numpy())

            save_images(os.path.join(sample_dir, f'{name}_{epoch + 1}-psnr:{eval_psnr:.6f}.png'),
                        output.detach().cpu().numpy().transpose(0, 2, 3, 1))

            psnr_for_eval_epoch.append(eval_psnr)
            """ssim_for_eval_epoch.append(eval_ssim)"""
            mean_l_eval_epoch.append(mean_l)
            mean_c_eval_epoch.append(mean_c)
            var_l_eval_epoch.append(var_l)
            var_c_eval_epoch.append(var_c)
            mse_l_eval_epoch.append(mse_l)
            mse_c_eval_epoch.append(mse_c)

        psnr_for_save_epoch.append(np.mean(psnr_for_eval_epoch))
        """ssim_for_save_epoch.append(np.mean(ssim_for_eval_epoch))"""
        mean_l_save_epoch.append(np.mean(mean_l_eval_epoch))
        mean_c_save_epoch.append(np.mean(mean_c_eval_epoch))
        var_l_save_epoch.append(np.mean(var_l_eval_epoch))
        var_c_save_epoch.append(np.mean(var_c_eval_epoch))
        mse_l_save_epoch.append(np.mean(mse_l_eval_epoch))
        mse_c_save_epoch.append(np.mean(mse_c_eval_epoch))

    torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()},
               os.path.join(checkpoint_dir, 'model.pth'))
print("The eval pictures:")
for idx in range(len(psnr_for_save_epoch)):
    number = (idx+1) * eval_every_epoch
    print(f"[*] The average psnr of {number} epoch is {psnr_for_save_epoch[idx]:.6f}.")
    """print(f"[*] The average ssim of {number} epoch is {ssim_for_save_epoch[idx]:.6f}.")"""
    print("[*] Finish training.")