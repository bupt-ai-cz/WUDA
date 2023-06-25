import cv2 as cv
import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

_MIN_AREA = 400
_ITER_COUNT = 3
#valid_classes=['0','2','7','9']

def creat_id_map(valid_classes):
    i = 0
    id_map = {}
    for classname in valid_classes:
        id_map[classname] = str(i)
        i = i + 1
    return id_map

def bbox2mask(bbox, class_id, init_mask):
    """Given a bbox and an image, return a mask.
    Param:
        bbox: bbox numpy array [y1, x1, y2, x2]
        image: an image read by PIL and convert to array
        class_id: class id of the object in the bbox
        init_mask: a initialized mask
    Returns:
        mask
    """
    init_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = class_id
    return init_mask

def bboxes2mask(txt, image, valid_classes):
    """Given a txt file and an image, return a mask
    Param:
        txt: a txt file contain bboxes, in yolov5 style: https://docs.ultralytics.com/tutorials/train-custom-datasets
        image: the image corresponding to the txt file
        valid_classes: a list contains valid_classes, for example: ['0','2','7','9']
    Retuens:
        mask
    """
    with open(txt, 'r') as f:
        bbox_list = f.readlines()
    valid_bbox = []
    for i in range(len(bbox_list)):
        bbox_list[i]=bbox_list[i].split()
        if bbox_list[i][0] in valid_classes:
            area = float(bbox_list[i][3])*float(bbox_list[i][4])
            bbox_list[i].append(area)
            valid_bbox.append(bbox_list[i])
        else:
            continue
    valid_bbox = sorted(valid_bbox,key=(lambda x:x[5]),reverse=True)
    for i in range(len(valid_bbox)):
        y1 = int((float(valid_bbox[i][2])-0.5*float(valid_bbox[i][4]))*image.shape[0])
        x1 = int((float(valid_bbox[i][1])-0.5*float(valid_bbox[i][3]))*image.shape[1])
        y2 = int((float(valid_bbox[i][2])+0.5*float(valid_bbox[i][4]))*image.shape[0])
        x2 = int((float(valid_bbox[i][1])+0.5*float(valid_bbox[i][3]))*image.shape[1])
        class_id = valid_bbox[i][0]
        valid_bbox[i] = [y1,x1,y2,x2,class_id]
    id_map = creat_id_map(valid_classes)
    mask = len(valid_classes)*np.ones(image.shape[0:2], np.uint8)
    for i in range(len(valid_bbox)):
        mask = bbox2mask(valid_bbox[i][:4], id_map[valid_bbox[i][4]], mask)
    return mask

def BDD100kbbox2mask(txt, image):
    with open(txt,'r') as f:
        bbox_list = f.readlines()
    for i in range(len(bbox_list)):
        bbox_list[i]=bbox_list[i].split()
        area = (int(bbox_list[i][4])-int(bbox_list[i][2]))*(int(bbox_list[i][3])-int(bbox_list[i][1]))
        bbox_list[i].append(area)
    bbox_list = sorted(bbox_list,key=(lambda x:x[5]),reverse=True)
    mask = 19*np.ones(image.shape[0:2], np.uint8)
    for i in range(len(bbox_list)):
        y1 = int(bbox_list[i][1])
        x1 = int(bbox_list[i][2])
        y2 = int(bbox_list[i][3])
        x2 = int(bbox_list[i][4])
        class_id = int(bbox_list[i][0])
        mask[y1:y2, x1:x2] = class_id
    return mask
    

        
def grabcut(txt, image_path, valid_classes):
    """Use Grabcut to create a binary segmentation of an image within a bounding box
    Param:
        image: path to input image astype('uint8')
        txt: a txt file contain bboxes, in yolov5 style: https://docs.ultralytics.com/tutorials/train-custom-datasets
        valid_classes: a list contains valid_classes, for example: ['0','2','7','9']
    Returns:
        mask with binary segmentation astype('uint8')
    Based on:
        https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html
    """
    image = np.array(Image.open(image_path))
    with open(txt, 'r') as f:
        bbox_list = f.readlines()
    valid_bbox = []
    for i in range(len(bbox_list)):
        bbox_list[i]=bbox_list[i].split()
        if bbox_list[i][0] in valid_classes:
            area = float(bbox_list[i][3])*float(bbox_list[i][4])
            bbox_list[i].append(area)
            valid_bbox.append(bbox_list[i])
        else:
            continue
    valid_bbox = sorted(valid_bbox,key=(lambda x:x[5]),reverse=True)
    for i in range(len(valid_bbox)):
        y1 = int((float(valid_bbox[i][2])-0.5*float(valid_bbox[i][4]))*image.shape[0])
        x1 = int((float(valid_bbox[i][1])-0.5*float(valid_bbox[i][3]))*image.shape[1])
        y2 = int((float(valid_bbox[i][2])+0.5*float(valid_bbox[i][4]))*image.shape[0])
        x2 = int((float(valid_bbox[i][1])+0.5*float(valid_bbox[i][3]))*image.shape[1])
        class_id = valid_bbox[i][0]
        valid_bbox[i] = [y1,x1,y2,x2,class_id]
    id_map = creat_id_map(valid_classes)

    img = cv.imread(image_path)
    mask = len(valid_classes)*np.ones(img.shape[0:2], np.uint8)
    #gct = np.zeros(img.shape[:2], np.uint8)
    for i in range(len(valid_bbox)):
        width, height = valid_bbox[i][3] - valid_bbox[i][1], valid_bbox[i][2] - valid_bbox[i][0]
        if width * height < _MIN_AREA:
            # OpenCV's Grabcut breaks if the rectangle is too small!
            # Fix: Draw a filled rectangle at that location, making the assumption everything in the rectangle is foreground
            assert(width*height > 0)
            mask = bbox2mask(valid_bbox[i][:4], id_map[valid_bbox[i][4]], mask)
            #mask = np.where((gct == 2) | (gct == 0), 0, 255).astype('uint8')
        else:
            # Use Grabcut to create a segmentation within the bbox
            rect = (valid_bbox[i][1], valid_bbox[i][0], width, height)
            gct = np.zeros(img.shape[:2], np.uint8)
            bgdModel, fgdModel = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            (gct, bgdModel, fgdModel) = cv.grabCut(img, gct, rect, bgdModel,fgdModel, _ITER_COUNT, cv.GC_INIT_WITH_RECT)
            mask = np.where((gct == cv.GC_FGD) | (gct == cv.GC_PR_FGD), id_map[valid_bbox[i][4]], mask).astype('uint8')
    return mask

def BDD100kgrabcut(txt, image_path):
    with open(txt,'r') as f:
        bbox_list = f.readlines()
    for i in range(len(bbox_list)):
        bbox_list[i]=bbox_list[i].split()
        area = (int(bbox_list[i][4])-int(bbox_list[i][2]))*(int(bbox_list[i][3])-int(bbox_list[i][1]))
        bbox_list[i].append(area)
    bbox_list = sorted(bbox_list,key=(lambda x:x[5]),reverse=True)

    img = cv.imread(image_path)
    mask = 19*np.ones(img.shape[0:2], np.uint8)
    for i in range(len(bbox_list)):
        width, height = int(bbox_list[i][4]) - int(bbox_list[i][2]), int(bbox_list[i][3]) - int(bbox_list[i][1])
        if (width * height < 0) | (width * height == 0):
            pass
        if width * height < _MIN_AREA:
            # OpenCV's Grabcut breaks if the rectangle is too small!
            # Fix: Draw a filled rectangle at that location, making the assumption everything in the rectangle is foreground
            mask = bbox2mask([int(bbox_list[i][1]),int(bbox_list[i][2]),int(bbox_list[i][3]),int(bbox_list[i][4])], int(bbox_list[i][0]), mask)
        elif int(bbox_list[i][0]) == 0:
            mask = bbox2mask([int(bbox_list[i][1]),int(bbox_list[i][2]),int(bbox_list[i][3]),int(bbox_list[i][4])], int(bbox_list[i][0]), mask)
        elif int(bbox_list[i][0]) == 10:
            mask = bbox2mask([int(bbox_list[i][1]),int(bbox_list[i][2]),int(bbox_list[i][3]),int(bbox_list[i][4])], int(bbox_list[i][0]), mask)
        else:
            # Use Grabcut to create a segmentation within the bbox
            rect = (int(bbox_list[i][2]), int(bbox_list[i][1]), width, height)
            gct = np.zeros(img.shape[:2], np.uint8)
            bgdModel, fgdModel = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            (gct, bgdModel, fgdModel) = cv.grabCut(img, gct, rect, bgdModel,fgdModel, _ITER_COUNT, cv.GC_INIT_WITH_RECT)
            mask = np.where((gct == cv.GC_FGD) | (gct == cv.GC_PR_FGD), int(bbox_list[i][0]), mask).astype('uint8')
    return mask

def cityscapes_grabcut_yolov5_format(txt, image_path):
    with open(txt,'r') as f:
        bbox_list = f.readlines()
    for i in range(len(bbox_list)):
        bbox_list[i]=bbox_list[i].split()
        area = float(bbox_list[i][4])*float(bbox_list[i][3])
        bbox_list[i].append(area)
    bbox_list = sorted(bbox_list,key=(lambda x:x[5]),reverse=True)

    img = cv.imread(image_path)
    mask = 19*np.ones(img.shape[0:2], np.uint8)
    for i in range(len(bbox_list)):
        y1 = int((float(bbox_list[i][2])-0.5*float(bbox_list[i][4]))*1024)
        x1 = int((float(bbox_list[i][1])-0.5*float(bbox_list[i][3]))*2048)
        y2 = int((float(bbox_list[i][2])+0.5*float(bbox_list[i][4]))*1024)
        x2 = int((float(bbox_list[i][1])+0.5*float(bbox_list[i][3]))*2048)
        width, height = int(float(bbox_list[i][3])*2048), int(float(bbox_list[i][4])*1024)
        if (width * height < 0) | (width * height == 0):
            continue
        if width * height < _MIN_AREA:
            # OpenCV's Grabcut breaks if the rectangle is too small!
            # Fix: Draw a filled rectangle at that location, making the assumption everything in the rectangle is foreground
            mask = bbox2mask([y1,x1,y2,x2], int(bbox_list[i][0]), mask)
        elif int(bbox_list[i][0]) == 0:
            mask = bbox2mask([y1,x1,y2,x2], int(bbox_list[i][0]), mask)
        elif int(bbox_list[i][0]) == 10:
            mask = bbox2mask([y1,x1,y2,x2], int(bbox_list[i][0]), mask)
        else:
            # Use Grabcut to create a segmentation within the bbox
            rect = (x1, y1, width, height)
            gct = np.zeros(img.shape[:2], np.uint8)
            bgdModel, fgdModel = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            (gct, bgdModel, fgdModel) = cv.grabCut(img, gct, rect, bgdModel,fgdModel, _ITER_COUNT, cv.GC_INIT_WITH_RECT)
            mask = np.where((gct == cv.GC_FGD) | (gct == cv.GC_PR_FGD), int(bbox_list[i][0]), mask).astype('uint8')
    return mask   

def GTAV_grabcut_yolov5_format(txt, image_path):
    with open(txt,'r') as f:
        bbox_list = f.readlines()
    for i in range(len(bbox_list)):
        bbox_list[i]=bbox_list[i].split()
        area = float(bbox_list[i][4])*float(bbox_list[i][3])
        bbox_list[i].append(area)
    bbox_list = sorted(bbox_list,key=(lambda x:x[5]),reverse=True)

    img = cv.imread(image_path)
    mask = 19*np.ones(img.shape[0:2], np.uint8)
    for i in range(len(bbox_list)):
        y1 = int((float(bbox_list[i][2])-0.5*float(bbox_list[i][4]))*img.shape[0])
        x1 = int((float(bbox_list[i][1])-0.5*float(bbox_list[i][3]))*img.shape[1])
        y2 = int((float(bbox_list[i][2])+0.5*float(bbox_list[i][4]))*img.shape[0])
        x2 = int((float(bbox_list[i][1])+0.5*float(bbox_list[i][3]))*img.shape[1])
        width, height = int(float(bbox_list[i][3])*img.shape[1]), int(float(bbox_list[i][4])*img.shape[0])
        if (width * height < 0) | (width * height == 0):
            continue
        if width * height < _MIN_AREA:
            # OpenCV's Grabcut breaks if the rectangle is too small!
            # Fix: Draw a filled rectangle at that location, making the assumption everything in the rectangle is foreground
            mask = bbox2mask([y1,x1,y2,x2], int(bbox_list[i][0]), mask)
        elif int(bbox_list[i][0]) == 0:
            mask = bbox2mask([y1,x1,y2,x2], int(bbox_list[i][0]), mask)
        elif int(bbox_list[i][0]) == 10:
            mask = bbox2mask([y1,x1,y2,x2], int(bbox_list[i][0]), mask)
        else:
            # Use Grabcut to create a segmentation within the bbox
            rect = (x1, y1, width, height)
            gct = np.zeros(img.shape[:2], np.uint8)
            bgdModel, fgdModel = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            (gct, bgdModel, fgdModel) = cv.grabCut(img, gct, rect, bgdModel,fgdModel, _ITER_COUNT, cv.GC_INIT_WITH_RECT)
            mask = np.where((gct == cv.GC_FGD) | (gct == cv.GC_PR_FGD), int(bbox_list[i][0]), mask).astype('uint8')
    return mask  

def save_networks(epoch, net, save_path):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_filename = 'net_%s.pth' % (epoch)
        torch.save(net.state_dict(), os.path.join(save_path, save_filename))        

def load_networks(epoch, net, load_path):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name 'net_%s.pth' % (epoch)
        """
        load_filename = 'net_%s.pth' % (epoch)
        load_path = os.path.join(load_path, load_filename)
        net.load_state_dict(torch.load(load_path))

# txt_path = '/root/simple_does_it/mine/datasets/GTA5/train/bbox'
# image_path = '/root/simple_does_it/mine/datasets/GTA5/train/image'
# save_path = '/root/simple_does_it/mine/datasets/GTA5/train/grabcut'
# for i in os.listdir(image_path):
#     mask = grabcut(os.path.join(txt_path,i.split(".")[0])+'.txt', os.path.join(image_path,i), valid_classes)
#     mask = Image.fromarray(mask)
#     mask.save(os.path.join(save_path,i))

txt_path = '/root/yolov5-pytorch/yolov5_for_BDD/runs/detect/exp/labels1'
image_path = '/root/yolov5-pytorch/cityscapes/train'
save_path = '/root/simple_does_it/mine/datasets/cityscapes/grabcut1'
for i in tqdm(os.listdir(txt_path)):
    mask = cityscapes_grabcut_yolov5_format(os.path.join(txt_path,i), os.path.join(image_path,i.split('.')[0]+'.png'))
    mask = Image.fromarray(mask).convert('L')
    mask.save(os.path.join(save_path,i.split('.')[0]+'.png'))
