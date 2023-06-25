import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
import dataset
import time
from tqdm import tqdm
import argparse
from tools import save_networks, load_networks
from PIL import Image
import wandb
from torchvision import transforms

def parse_opt():
#Set train options
    parser = argparse.ArgumentParser(description='Train options')
    parser.add_argument('--name', type=str, default='GTAV_source', help='experiment name')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of total epochs')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--init_lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--image_root', type=str, default='/root/Domain-Adaption-Experiment/data/GTA5/images',help='path to images')
    parser.add_argument('--mask_root', type=str, default='/root/Domain-Adaption-Experiment/data/GTA5/labels',help='path to labels (e.g. gray images with the values of 0-18 for 19 categories)')
    parser.add_argument('--num_classes', type=int, default=20, help='number of classes including background')
    parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models and Intermediate results are saved here')
    opt = parser.parse_args()
    return opt

opt = parse_opt()
if not os.path.exists(os.path.join(opt.checkpoints_dir,opt.name)):
    os.makedirs(os.path.join(opt.checkpoints_dir,opt.name))
    print('Folder %s has been created.'%(os.path.join(opt.checkpoints_dir,opt.name)))
else:
    print('Folder %s already exists.'%(os.path.join(opt.checkpoints_dir,opt.name)))
valid_classes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
print('*****************************************')
print('Experiment settings:', '\n', opt)
print('valid classes:', valid_classes)
print('*****************************************')
with open(os.path.join(opt.checkpoints_dir,opt.name,'loss_log.txt'), 'a') as f:
    f.write(str(opt))
    f.write('\n'+'valid classes: '+str(valid_classes))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=opt.num_classes, aux_loss=None)
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)

wandb.init(project="representation shift", entity="saintjay")
wandb.config = {
  "learning_rate": opt.init_lr,
  "epochs": opt.num_epochs,
  "batch_size": opt.batch_size,
  "num_classes": opt.num_classes
}

train_set = dataset.GTAVDataset(opt.image_root, opt.mask_root)
train_loader = DataLoader(dataset = train_set, batch_size = opt.batch_size, shuffle = True, drop_last=True)

model.train()
literation = 0
for epoch in range(opt.num_epochs):
    
    print('training epoch %d'%(epoch + 1))
    epoch_start = time.time()
    for inputs, labels, __ in tqdm(train_loader):
        literation = literation+1
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predict = model(inputs)['out']
        loss = loss_function(predict, labels)
        loss.backward()
        optimizer.step()
        if literation%10 == 0:
            wandb.log({"loss": loss, "epoch": (epoch + 1)})
            with open(os.path.join(opt.checkpoints_dir,opt.name,'loss_log.txt'), 'a') as f:
                f.write('\n'+'epoch: %s literation: %d loss: %s'%(str(epoch + 1),literation,str(loss.item())))
        if literation%2000 == 0:
            save_networks(literation+1, model, os.path.join(opt.checkpoints_dir, opt.name))
            optimizer = torch.optim.Adam(model.parameters(), lr=(opt.init_lr)/((literation/2000)+1))
save_networks(literation+1, model, os.path.join(opt.checkpoints_dir, opt.name))
