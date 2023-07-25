# Copyright (c) 2023, University of California, Merced. All rights reserved.

# This file is part of the MummyNutsBench software package developed by
# the team members of Prof. Xiaoyi Lu's group (PADSYS Lab) at the University
# of California, Merced.

# For detailed copyright and licensing information, please refer to the license
# file LICENSE in the top level directory.


'''
It's best to make a ramdisk first.

mkdir /mnt/ramdisk
mount -t tmpfs -o size=1024m tmpfs /mnt/ramdisk

'''

import os
import shutil
from cv2 import threshold
import numpy as np
import logging
from PIL import Image, ImageDraw
from math import ceil
from yolov5 import detect
from nms import non_max_suppression_fast

WIN_WIDTH = 448
WIN_HEIGHT = 448
WIN_STRIDE_X = int(80*2)
WIN_STRIDE_Y = int(80*2)

# extra
BOX_WINDOWS = False
RECT_PADDING = 10

INPUT_IMG = "../datasets/hlight_mnut2/test/images/IMG_4445_png.rf.30009e9843b9a1516e7f63a1bd9fda2f.jpg"
INPUT_LBL = "../datasets/hlight_mnut2/test/labels/IMG_4445_png.rf.30009e9843b9a1516e7f63a1bd9fda2f.txt"
SLIDING_WINDOW_DIR = "/mnt/ramdisk/sliding_windows"
EXP_DIR = "/mnt/ramdisk/exp"
BEST_MODEL = "./best_hlight.pt"

CONF_THRES = 0.85

if not "yolov5" in os.listdir():
    os.system("git clone https://github.com/ultralytics/yolov5")

if not "sliding_windows" in os.listdir("/mnt/ramdisk"):
    os.mkdir(SLIDING_WINDOW_DIR)

img = Image.open(INPUT_IMG)
len_x, len_y = img.size

logging.info(f"{WIN_WIDTH}x{WIN_HEIGHT} windows ")
logging.info(f"{WIN_STRIDE_X}x {WIN_STRIDE_Y}y strides")
logging.info(f"{(ceil(len_x/WIN_STRIDE_X)*ceil(len_y/WIN_STRIDE_Y))} windows will be created")

# mute logging because it spams the terminal
logging.disable(logging.INFO)

count = 0
for i in range(0, len_y, WIN_STRIDE_Y):
    for j in range(0, len_x, WIN_STRIDE_X):
        x, y = min(j+WIN_WIDTH, len_x), min(i+WIN_HEIGHT, len_y)
        
        # yolov5 doesn't allow in-memory streams
        with open(f"{SLIDING_WINDOW_DIR}/{count}.jpg", "wb") as tmpf:
            window = img.crop((j, i, x, y))
            window.save(tmpf)
        
        count += 1

img.close()

logging.info(f"Running inference on {count} windows")
# run inference on all the windows
detect.run(weights=BEST_MODEL, 
    source=SLIDING_WINDOW_DIR, 
    imgsz=(WIN_WIDTH, WIN_HEIGHT), project=f"{SLIDING_WINDOW_DIR}/../", save_txt=True, nosave=False, save_conf=True, conf_thres=CONF_THRES)

logging.disable(logging.NOTSET)
logging.info("Stitching windows after inference")

# reset the counter
count = 0
boxes = []

out = Image.open(INPUT_IMG)
draw = ImageDraw.Draw(out)
for i in range(0, len_y, WIN_STRIDE_Y):
    for j in range(0, len_x, WIN_STRIDE_X):
        x, y = min(j+WIN_WIDTH, len_x), min(i+WIN_HEIGHT, len_y)
        
        if os.path.exists(f"{EXP_DIR}/labels/{count}.txt"):
            # read the coordinates
            with open(f"{EXP_DIR}/labels/{count}.txt") as f:
                labels = [[*map(float, line.split(' ')[1:])] for line in f] 
            
            rects = [(j+(x0-w/2)*WIN_WIDTH-RECT_PADDING, i+(y0-h/2)*WIN_HEIGHT-RECT_PADDING, 
                j+(x0+w/2)*WIN_WIDTH+RECT_PADDING, i+(y0+h/2)*WIN_HEIGHT+RECT_PADDING) for x0, y0, w, h, _ in labels]
            
            boxes.extend(rects)

        count += 1

# image space coordinates
ground_truth = []
with open(INPUT_LBL, 'r') as f:
    for r in f:
        x, y, w, h = map(float, r.split(' ')[1: ])
        ground_truth.append([(x-w/2)*len_x, (y-h/2)*len_y, (x+w/2)*len_x, (y+h/2)*len_y])

boxes = non_max_suppression_fast(np.array(boxes), 0.4)
for box in boxes:
    draw.rectangle(list(box), width=5, outline=0x0000FF)

for truth in ground_truth:
    draw.rectangle(list(truth), width=5, outline=0x00FF00)

if BOX_WINDOWS:
    for i in range(0, len_x, WIN_STRIDE_Y):
        for j in range(0, len_y, WIN_STRIDE_X):
            x, y = min(i+WIN_WIDTH, len_x), min(j+WIN_HEIGHT, len_y)
            draw.rectangle((j, i, x, y), width=2)

out.save(f"{SLIDING_WINDOW_DIR}/../output.jpg")
out.close()

# calculating iou https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def iou(b1, b2):
    if b1[0] < b1[0] or b1[1] < b1[1] or b2[0] < b2[0] or b2[1] < b2[1]:
        b1, b2 = b2, b1

    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    if xB < xA or yB < yA: return 0

    intersect = (yA - yB) * (xA - xB)

    A_Area = (b1[0]-b1[2])*(b1[1]-b1[3])
    B_Area = (b2[0]-b2[2])*(b2[1]-b2[3])

    return intersect/(A_Area+B_Area-intersect)

thresholds = [0.2+i/20 for i in range(12)]
ious = []
precision = []
recall = []
for thresh in thresholds:
    FP = FN = TP = TN = 0 # true negatives are unused
    for box in boxes:
        is_fp = True # no ground truths at this location marked positive
        for truth in ground_truth:
            if iou(box, truth) >= thresh:
                if_fp = False
                TP += 1
        FP += is_fp
    
    for truth in ground_truth:
        is_fn = True
        for box in boxes:
            if iou(truth, box) >= thresh:
                if_fn = False
                break
        FN += is_fn
    # print(FN, FP, TP)
    if FP+TP == 0 or FN + TP == 0: continue
    precision.append(TP/(FP+TP))
    recall.append(TP/(FN+TP))

recall.append(0)
precision.append(1)
vals = sorted([*zip(recall, precision)])
precision = [v[1] for v in vals]
recall = [v[0] for v in vals]

avg_prec = sum((vals[i][1]+vals[i-1][1])*.5*(vals[i][0]+vals[i-1][0]) for i in range(1, len(vals)))/len(recall)

# https://blog.paperspace.com/mean-average-precision/

shutil.rmtree(f"{EXP_DIR}")
shutil.rmtree(f"{SLIDING_WINDOW_DIR}")

print(f"mAP: {avg_prec}\nFalse Positives: {FP}\nFalse Negatives: {FN}\nTrue Positives: {TP}")
print(f"mAP: {avg_prec}\nPrecision: {precision}\nRecall: {recall}")