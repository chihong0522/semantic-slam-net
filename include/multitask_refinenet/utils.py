import base64
import logging
import time
import numpy as np
import io
import json
import sys
import math

from models import net
import torch, torchvision
from torch.autograd import Variable
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch.nn.functional as F

CMAP = np.load('/home/chihung/semantic_map_ws/src/semantic_cloud/include/multitask_refinenet/cmap_nyud_bgr8.npy')
# CMAP = np.load('/home/chihung/semantic_map_ws/src/semantic_cloud/include/multitask_refinenet/cmap_nyud_to_sceneNN_bgr8.npy')
DEPTH_COEFF = 5000.  # to convert into metres
HAS_CUDA = torch.cuda.is_available()
# HAS_CUDA = None
IMG_SCALE = 1. / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
MAX_DEPTH = 8.
MIN_DEPTH = 0.
NUM_CLASSES = 40
NUM_TASKS = 2  # segm + depth

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


model = net(num_classes=NUM_CLASSES, num_tasks=NUM_TASKS)
if HAS_CUDA:
    _ = model.cuda()
_ = model.eval()

ckpt = torch.load('/home/chihung/semantic_map_ws/src/semantic_cloud/weights/ExpNYUD_joint.ckpt')
model.load_state_dict(ckpt['state_dict'])

def inference(img):
    start_ = time.time()
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
        if HAS_CUDA:
            img_var = img_var.cuda()
        segm, depth = model(img_var)
        
        
        # segm_ori = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
        #                     img.shape[:2][::-1],
        #                     interpolation=cv2.INTER_CUBIC)
        # depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
        #                     img.shape[:2][::-1],
        #                     interpolation=cv2.INTER_CUBIC)

        segm_output = segm[0, :NUM_CLASSES].unsqueeze(0)
        size = img.shape[:2]
        segm = F.interpolate(segm_output, size=size, mode='nearest').squeeze(0).permute(1,2,0)

        depth_output = depth[0, 0].unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth_output,size, mode='nearest').squeeze(0).squeeze(0)

    # Take 3 best predictions and their confidences (probabilities)
    pred_confidences, pred_labels = torch.topk(input=segm, k=3, dim=2, largest=True, sorted=True)
    
    depth_ = np.abs(depth.cpu().data.numpy())
    end_ = time.time()
    print("FPS=",1/(end_-start_))
    return pred_confidences, pred_labels, depth_

if __name__ == '__main__':
    img = np.ones((480, 640,3),dtype=np.uint8)
    r1, r2 = inference(img)
    print("gd")