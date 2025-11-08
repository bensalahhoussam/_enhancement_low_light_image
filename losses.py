import torch
import torch.nn as nn
import torch.nn.functional as F




class L1Loss(nn.Module):
    def __init__(self,):
        super(L1Loss, self).__init__()
        self.loss_fn = nn.L1Loss()
    def forward(self, out, ref):
        loss = self.loss_fn(out,ref)
        return loss

class ContentLoss(nn.Module):
    def __init__(self,):
        super(ContentLoss, self).__init__()
    def forward(self, cp , ci):
        cp_flat = cp.view(cp.size(0), -1)
        ci_flat = ci.view(ci.size(0), -1)
        loss = torch.sum((cp_flat - ci_flat) ** 2, dim=1)
        return loss.mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=0.08):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def forward(self, lp, lr, li):
        lp_flat = lp.view(lp.size(0), -1)

        lr_flat = lr.view(lr.size(0), -1)

        li_flat = li.view(li.size(0), -1)

        D_pos = torch.sum((lp_flat - lr_flat) ** 2, dim=1)
        D_neg = torch.sum((lp_flat - li_flat) ** 2, dim=1)
        loss = F.relu(D_pos - D_neg + self.margin)
        return loss.mean()

def rgb_to_hsv(image):


    if image.dim() == 3:
        image = image.unsqueeze(0)  # ensure batch dim

    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]

    maxc, _ = image.max(dim=1, keepdim=True)
    minc, _ = image.min(dim=1, keepdim=True)
    v = maxc
    delta = maxc - minc

    # Saturation
    s = delta / (maxc + 1e-8)
    s[maxc == 0] = 0

    # Hue
    h = torch.zeros_like(maxc)
    mask = delta > 1e-8

    # only compute hue where delta != 0
    r_eq = (maxc == r) & mask
    g_eq = (maxc == g) & mask
    b_eq = (maxc == b) & mask

    h[r_eq] = ((g - b) / delta)[r_eq] % 6
    h[g_eq] = ((b - r) / delta + 2)[g_eq]
    h[b_eq] = ((r - g) / delta + 4)[b_eq]

    h = (h / 6) % 1.0  # normalize to [0,1]

    return h, s, v

class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()

        self.ContentLoss= ContentLoss()

        self.TripletLoss = TripletLoss()

    def forward(self, cr,cp):
        [c_i, l_i, c_p, l_p]= cp

        [_, _, c_r, l_r] = cr

        c_loss = self.ContentLoss(c_p,c_i)

        f_loss = self.TripletLoss(l_p, l_r, l_i)

        loss = c_loss+f_loss
        return loss

class ContentConsistencyLoss(nn.Module):
    def __init__(self):
        super(ContentConsistencyLoss, self).__init__()
    def forward(self, pred, lowlight):

        Hp, Sp, _ = rgb_to_hsv(pred)
        Hi, Si, _ = rgb_to_hsv(lowlight)

        B, _, H, W = Hp.shape

        Hp_flat = Hp.view(B, -1)
        Hi_flat = Hi.view(B, -1)
        Sp_flat = Sp.view(B, -1)
        Si_flat = Si.view(B, -1)

        cos_H = F.cosine_similarity(Hp_flat, Hi_flat, dim=1)
        cos_S = F.cosine_similarity(Sp_flat, Si_flat, dim=1)

        Lc_H = 1 - cos_H

        Lc_S = 1 - cos_S

        loss = (Lc_H + Lc_S)
        return loss.mean()



