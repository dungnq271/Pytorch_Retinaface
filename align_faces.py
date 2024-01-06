# %%
import os
import os.path as osp
import live
import random
import math
import importlib
import shutil

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style="white")
importlib.reload(live)

# %%
class Args():
    trained_model = "./weights/Resnet50_Final.pth"
    network = "resnet50"
    cpu = False
    confidence_threshold = 0.02
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    vis_thres = 0.6
    enlarge = 0

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def rotate_bound(image, angle):
    """
    From https://pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def align_face(img_raw, eyes):
    left_eye_x, left_eye_y = eyes[0], eyes[1]
    right_eye_x, right_eye_y = eyes[2], eyes[3]    

    left_eye_center = (left_eye_x, left_eye_y)
    right_eye_center = (right_eye_x, right_eye_y)

    #----------------------
    #find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
        # print("rotate to clock direction")
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        # print("rotate to inverse clock direction")

    #----------------------
    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, point_3rd)
    c = euclidean_distance(right_eye_center, left_eye_center)

    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = np.arccos(cos_a)

    angle = (angle * 180) / math.pi
    if direction == -1:
        angle = 90 - angle

    #--------------------
    #rotate image
    new_img = rotate_bound(img, angle*direction)    
    return new_img

# %%
args = Args()
det = live.Detector(args)

# %%
data_dir = "../../face-analysis-lightning/data/raw" 
img_dir = osp.join(data_dir, "cropped_faces")
img_fnames = os.listdir(img_dir)

# %%
img_fname = random.sample(img_fnames, 1)[0]
img_fpath = osp.join(img_dir, img_fname)
img = cv2.imread(img_fpath) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# %%
dets = det.detect(img)
# img_annt = det.annotate(img, dets)
# plt.imshow(img_annt)

# %%
new_img = align_face(img, dets[0][5:9])
plt.imshow(new_img)

# %%
aligned_img_dir = osp.join(data_dir, "align_faces")
if osp.exists(aligned_img_dir):
    shutil.rmtree(aligned_img_dir)

os.makedirs(aligned_img_dir, exist_ok=True)

# %%
for img_fname in tqdm(img_fnames):
    img_fpath = osp.join(img_dir, img_fname)
    img = cv2.imread(img_fpath) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = det.detect(img)    

    new_img = align_face(img, dets[0][5:9])
    new_img_fpath = osp.join(aligned_img_dir, img_fname)
    cv2.imwrite(new_img_fpath, new_img)

# %%
