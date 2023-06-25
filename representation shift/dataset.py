import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import imageio
from tools import preprocess_label

class GTAVDataset(Dataset):
#mask and image should have the same name
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.data_list = []
        for i in os.listdir(self.image_path):
            self.data_list.append(i.split('.')[0])
    
    def __getitem__(self, index):
        image_name = self.data_list[index]
        image = Image.open(os.path.join(self.image_path, image_name+'.png')).convert("RGB")
        mask = Image.open(os.path.join(self.mask_path, image_name+'_labelTrainIds.png')).convert('L')
        #mask=torch.LongTensor(np.array(mask).astype('int32'))
        transform_image = transforms.Compose([
                                                transforms.Resize((1052,1914)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
        ])
        transform_mask = transforms.Compose([
                                                transforms.Resize((1052,1914)),
        ])
        image_tensor = transform_image(image)
        mask_tensor = torch.LongTensor(np.array(transform_mask(mask)).astype('int32'))
        return image_tensor, mask_tensor, image_name

    def __len__(self):
        return len(self.data_list)

# class GTAVDataset(Dataset):
# #mask and image should have the same name
    
#     def __init__(self, image_path, mask_path):
#         self.image_path = image_path
#         self.mask_path = mask_path
#         self.data_list = []
#         for i in os.listdir(self.image_path):
#             self.data_list.append(i.split('_leftImg8bit')[0])
    
#     def __getitem__(self, index):
#         image_name = self.data_list[index]
#         image = Image.open(os.path.join(self.image_path, image_name+'_leftImg8bit.png')).convert("RGB")
#         mask = Image.open(os.path.join(self.mask_path, image_name+'_gtFine_labelTrainIds.png')).convert('L')
#         #mask=torch.LongTensor(np.array(mask).astype('int32'))
#         transform_image = transforms.Compose([
#                                                 transforms.Resize((1024,2048)),
#                                                 transforms.ToTensor(),
#                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                 std = [0.229, 0.224, 0.225])
#         ])
#         transform_mask = transforms.Compose([
#                                                 transforms.Resize((1024,2048)),
#         ])
#         image_tensor = transform_image(image)
#         mask_tensor = torch.LongTensor(np.array(transform_mask(mask)).astype('int32'))
#         return image_tensor, mask_tensor, image_name

#     def __len__(self):
#         return len(self.data_list)

class SYNTHIADataset(Dataset):
#mask and image should have the same name
    __id_map = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5, 15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12, 8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.data_list = []
        for i in os.listdir(self.image_path):
            self.data_list.append(i.split('.')[0])
    
    def __getitem__(self, index):
        image_name = self.data_list[index]
        image = Image.open(os.path.join(self.image_path, image_name+'.png')).convert("RGB")
        lbl = np.asarray(imageio.imread(os.path.join(self.mask_path, image_name+'.png'), format='PNG-FI'))[:, :, 0]
        lbl = Image.fromarray(preprocess_label(lbl, self.__id_map))
        #mask=torch.LongTensor(np.array(mask).astype('int32'))
        transform_image = transforms.Compose([
                                                transforms.Resize((760,1280)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
        ])
        transform_mask = transforms.Compose([
                                                transforms.Resize((760,1280)),
        ])
        image_tensor = transform_image(image)
        mask_tensor = torch.LongTensor(np.array(transform_mask(lbl)).astype('int32'))
        return image_tensor, mask_tensor, image_name

    def __len__(self):
        return len(self.data_list)

class testDataset(Dataset):
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
        #mask=torch.LongTensor(np.array(mask).astype('int32'))
        transform_image = transforms.Compose([
                                                transforms.Resize((self.H,self.W)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
        ])
        image_tensor = transform_image(image)
        return image_tensor, image_name

    def __len__(self):
        return len(self.data_list)