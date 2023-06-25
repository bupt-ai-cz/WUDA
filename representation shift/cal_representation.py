import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms
from models import deeplab
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description='path')
    parser.add_argument('--dataset_path', type=str, default='/root/representation_shift/datasets/cityscapes/train',help='path to target domain images')
    parser.add_argument('--model_path', type=str, default='/root/representation_shift/checkpoints/GTAV_source/net_12501.pth', help='pretrained model')
    parser.add_argument('--save_path', type=str, default='/root/representation_shift/checkpoints/GTAV_source/dataset_B', help='dataset representation is saved here')
    parser.add_argument('--H', type=int, default=1024, help='the heigth of the images')
    parser.add_argument('--W', type=int, default=2048, help='the width of the images')
    opt = parser.parse_args()
    return opt


class SegDataset(Dataset):
#mask and image should have the same name
    def __init__(self, image_path, H, W):
        self.image_path = image_path
        self.H = H
        self.W = W
        self.data_list = []
        for i in os.listdir(self.image_path):
            self.data_list.append(i)
    
    def __getitem__(self, index):
        image_name = self.data_list[index]
        image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        transform_image = transforms.Compose([
                                                transforms.Resize((self.H,self.W)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
        ])
        image_tensor = transform_image(image)
        return image_tensor

    def __len__(self):
        return len(self.data_list)


opt = parse_opt()
datasetB_path = opt.dataset_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=20, aux_loss=None)
model = model.to(device)
model.load_state_dict(torch.load(opt.model_path))


def hook(module, input, output):
    feature.append(output)
    return None

model.eval()


dataset = SegDataset(datasetB_path,opt.H,opt.W)
datasetloader = DataLoader(dataset = dataset, batch_size = 1, shuffle = True, drop_last=True)

literation = 0
for image in tqdm(datasetloader):
    literation = literation + 1
    feature = []
    model.backbone.layer4[2].conv3.register_forward_hook(hook)
    model(image.to(device))
    sample = feature[0]
    sample = sample.squeeze()
    sample = torch.mean(sample, (1,2))
    sample_numpy = sample.detach().cpu().numpy()
    sample_numpy=np.expand_dims(sample_numpy,1)
    if literation == 1:
        dataset_B = sample_numpy
    else:
        dataset_B = np.concatenate((dataset_B, sample_numpy),axis = 1)
np.save(opt.save_path, arr=dataset_B)