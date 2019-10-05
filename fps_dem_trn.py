from __future__ import print_function
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import pandas as pd
from opts import parser


def putIterationsPerSec(frame, prob, label, count):
    """
    Add iterations per second text to lower-left corner of a frame.
    """
    color = (255, 255, 255)
    if prob > 0.8:
        color = (0, 255, 0)
    elif prob > 0.6:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.putText(frame, label,
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)
    cv2.putText(frame, str(count),
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)

    cv2.rectangle(frame, (0, 0), (640, 480), color, 20)

    return frame


# created a *threaded *video stream, allow the camera senor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = VideoStream(src=0).start()
fps = FPS().start()

import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy

import torchvision
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from models import TSN
import transforms
from torch.nn import functional as F


def load_frames(frames, num_frames=8):
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise (ValueError('Video must have at least {} frames'.format(num_frames)))


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames

global args

args = parser.parse_args()

parser = argparse.ArgumentParser(description="test TRN on a single video")
# group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('--video_file', type=str, default='')
parser.add_argument('--frame_folder', type=str, default='')
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff'], )
parser.add_argument('--dataset', type=str, default='jester',
                    choices=['something', 'jester', 'moments', 'somethingv2'])
parser.add_argument('--rendered_output', type=str, default='test')
parser.add_argument('--arch', type=str, default="InceptionV3")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--num_motion', type=int, default=3)
parser.add_argument('--dropout', '--do', default=0.3, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--img_feature_dim', type=int, default=256)

# All BNInception TRNmultiscale
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--weights', type=str,
                   default='/home/machine/PROJECTS/OTHER/fubel/stmodeling/model/All_STModeling_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar')
categories_file = '/home/machine/PROJECTS/OTHER/fubel/stmodeling/category_all.txt'


# My BNInception TRNmultiscale
# parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
# parser.add_argument('--weights', type=str,
#                     default='/home/machine/PROJECTS/OTHER/fubel/stmodeling/model/STModeling_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar')
# categories_file = '/home/machine/PROJECTS/OTHER/fubel/stmodeling/category_taras.txt'


# All BNInception MLP
#parser.add_argument('--consensus_type', type=str, default='MLP')
#parser.add_argument('--weights', type=str,
#                    default='/home/machine/PROJECTS/OTHER/fubel/stmodeling/model/STModeling_jester_RGB_BNInception_MLP_segment8_best.pth.tar')
#categories_file = '/home/machine/PROJECTS/OTHER/fubel/stmodeling/category_all.txt'

args = parser.parse_args()

# Get dataset categories.
categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)

args.arch = 'InceptionV3' if args.dataset == 'moments' else 'BNInception'

# Load model.
net = TSN(num_class, args)

checkpoint = torch.load(args.weights)
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net.cuda().eval()

# Initialize frame transforms.
transform = torchvision.transforms.Compose([
    transforms.GroupScale(net.scale_size),
    transforms.GroupCenterCrop(net.input_size),
    transforms.Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    transforms.ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    transforms.GroupNormalize(net.input_mean, net.input_std),
])

frame_label = "";
prob = 0.0
bufferf = []
prev_category = 0
category_count = 0

# loop over some frames...this time using the threaded stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = putIterationsPerSec(frame, prob, frame_label, category_count)

    # frame = cv2.resize(frame, (640, 480))

    crop_img = frame[0:720, 0:720]
    # h,w = crop_img.shape[:2]
    input_img = crop_img;
    input_pill = Image.fromarray(input_img)
    cv2.imshow("Frame", crop_img)
    key = cv2.waitKey(1) & 0xFF

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if (len(bufferf) < 16):
        bufferf.append(input_pill)
    else:
        input_frames = load_frames(bufferf)
        data = transform(input_frames)
        input = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda(), volatile=True)

        # with torch.no_grad():
        logits = net(input)
        h_x = torch.mean(F.softmax(logits, 1), dim=0).data
        probs, idx = h_x.sort(0, True)
        prob = probs[0]
        frame_label = '{:.3f} -> {}'.format(prob, categories[idx[0]])
        if prob > 0.8:
            if categories[idx[0]] == prev_category:
                category_count += 1
            else:
                category_count = 0
            prev_category = categories[idx[0]]
            print(frame_label, category_count)
        bufferf[:-1] = bufferf[1:];
        bufferf[-1] = input_pill
        # check to see if the frame should be displayed to our screen

    # update the FPS counter
    fps.update()