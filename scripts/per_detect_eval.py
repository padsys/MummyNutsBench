
# Copyright (c) 2023, University of California, Merced. All rights reserved.

# This file is part of the MummyNutsBench software package developed by
# the team members of Prof. Xiaoyi Lu's group (PADSYS Lab) at the University
# of California, Merced.

# For detailed copyright and licensing information, please refer to the license
# file LICENSE in the top level directory.



# Note: you'll need opencv installed to run this code.

import cv2
import uuid
import os
import numpy as np
import glob
import random
import re

# ==============================================================================================================

# path of folders: should be two folders, one for input images.jpg and annotations.txt, one for outputs
images_path = glob.glob(r"C:\Users\darre\Desktop\School\Tools\Per Detect Eval\inputs\*.JPG")
annotations_path = glob.glob(r"C:\Users\darre\Desktop\School\Tools\Per Detect Eval\inputs\*.txt")
output_path = 'C:/Users/darre/Desktop/School/Tools/Per Detect Eval/outputs'

# changes size of viewing window
crop_size_x = 200
crop_size_y = 200

# ==============================================================================================================

break_flag = False

def key_code(key):
    match key:
        case 49: #'1'
            return "[C]" #Camo
        case 50: #'2'
            return "[B]" #Blocking
        case 51: #'3'
            return "[O]" #Overlap
        case 52: #'4'
            return "[T]" #Tiny
        case 53: #'5'
            return "[N]" #Noisy
        case 54: #'6'
            return "[D]" #Dark
        case 55: #'7'
            return "[0]" #No Difficulty
        case _: #other
            return ""

img_counter = 0
counter = -1

for img_path in images_path:
    counter += 1

    img = cv2.imread(img_path)
    annotated_img = np.array(img)
    new_lines = []

    #print(img_horizontal, img_vertical)

    with open(annotations_path[counter], "r") as f_o:
        lines = f_o.readlines()

        for line in lines:
            numbers = re.findall("[0-9.]+", line)
           # text = "{} {} {} {} {}".format(int(numbers[0])-15, numbers[1], numbers[2], numbers[3], numbers[4])
            text = "{} {} {} {} {}".format(numbers[0], numbers[1], numbers[2], numbers[3], numbers[4])

            #print(text)

            img = cv2.imread(img_path)
            image_np = np.array(img)

            img_horizontal = image_np.shape[1]
            img_vertical = image_np.shape[0]

            anno_x = float(numbers[1]) * img_horizontal
            anno_y = float(numbers[2]) * img_vertical
            anno_width = float(numbers[3]) * img_horizontal
            anno_height = float(numbers[4]) * img_vertical

            temp_rule_follow = True

            img_counter += 1
            if (anno_width > crop_size_x) or (anno_height > crop_size_y):
                print('Warning: image {} has an annotation box size bigger than crop size'.format(img_counter))
                temp_rule_follow = False

            x1 = int(anno_x - crop_size_x/2)
            y1 = int(anno_y - crop_size_y/2)
            x2 = int(anno_x + crop_size_x/2)
            y2 = int(anno_y + crop_size_y/2)

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 < 0:
                x2 = 0
            if y2 < 0:
                y2 = 0
            if x1 > img_horizontal:
                x1 = img_horizontal
            if y1 > img_vertical:
                y1 = img_vertical
            if x2 > img_horizontal:
                x2 = img_horizontal
            if y2 > img_horizontal:
                y2 = img_horizontal

            a_x1 = int(anno_x - anno_width/2)
            a_y1 = int(anno_y - anno_height/2)
            a_x2 = int(anno_x + anno_width/2)
            a_y2 = int(anno_y + anno_height/2)

            crop_img = image_np[y1:y2,x1:x2]

            drawbox = cv2.imread(img_path)
            drawbox_np = np.array(drawbox)
            drawbox_img = drawbox_np
            cv2.rectangle(drawbox_img, (a_x1, a_y1), (a_x2, a_y2), (255,0,0), 2)
            cv2.rectangle(annotated_img, (a_x1, a_y1), (a_x2, a_y2), (255,0,0), 2)

            #cv2.imshow(',',drawbox_img)
            showbox_img = drawbox_img[y1:y2,x1:x2]

            print('Image Num: {}'.format(img_counter))
            print('[1] - Camo')
            print('[2] - Blocking')
            print('[3] - Overlap')
            print('[4] - Tiny')
            print('[5] - Noisy')
            print('[6] - Dark')
            print('[7] - No Difficulty')
            print('[Q] - Quit')

            cv2.imshow("Cropped Image", showbox_img)

            key = cv2.waitKey(0)

            #score = input("Enter Grade:")
            score = key_code(key)

            new_text = text + " " + score
            print()
            print(new_text)
            print()

            new_lines.append(new_text)

            cv2.putText(showbox_img, "{}".format(score),
                  (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                  (255,255,0), 3)

            cv2.putText(annotated_img, "{}".format(score),
                  (a_x1 + 25, a_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                  (255,255,0), 3)

            # imgname = ('IMG_{}.JPG'.format(img_counter))
            # cv2.imwrite(os.path.join(output_path, imgname), showbox_img)

            # filename = os.path.join(output_path, "IMG_{}.txt".format(img_counter))
            # file = open(filename,"w+")
            # file.write(new_text)

            if key == 113:
                break_flag = True
                break
        if break_flag:
            print()
            print('Quiting...')
            break

    #print(img_path[59:])
    imgname = ('eval_{}.JPG'.format(img_path[59:]))
    cv2.imwrite(os.path.join(output_path, imgname), annotated_img)

    #print(annotations_path[counter][59:])

    filename = os.path.join(output_path, "eval_{}.txt".format(annotations_path[counter][59:]))
    file = open(filename,"w+")
    for new_line in new_lines:
        file.write(new_line)
        file.write('\n')
    file.write(new_text)

cv2.destroyAllWindows()
