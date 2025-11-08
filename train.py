
from model import UNet
from optimizer import *
from engine import  *

from losses import *
from dataloader import train_dataloader,test_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = UNet(in_channels=3,out_h=512,out_w=512,base_ch=64).to(device)

optimizer = smart_optimizer(model, "AdamW", 0.0001, 0.9, 0.0005)
epochs = 1000
warmup_steps = len(train_dataloader)*int(epochs*0.2)
stable_steps = len(train_dataloader)*int(epochs*0.1)
decay_steps = len(train_dataloader)*int(epochs*0.7)



"""
scheduler = WarmupStableDecayLR( optimizer, warmup_steps, stable_steps, decay_steps, warmup_start_lr=1e-6, base_lr=1e-4, final_lr=1e-6)
"""


loss_1 = L1Loss()
loss_2 = FeatureLoss()
loss_3 = ContentConsistencyLoss()



results = train(model,train_dataloader,test_dataloader,optimizer,loss_1,loss_2,loss_3,epochs,1,device)







