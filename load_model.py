import os
import json
import pandas as pd
import numpy as np
import torch
import time
from torch.autograd import Variable
from PIL import Image
import cv2
from torch.nn import functional as F

from model import generate_model
from spatial_transforms_new import *           #######
from utils import  Queue

import numpy as np
import datetime
import time
from os import path


class Opt():
    def __init__(self,root_path='',path_det='',path_clf='',n_classes_clf=11,no_cuda=True):
        self.root_path=''
        self.resume_path=''
        self.sample_duration=''
        self.pretrain_path=''
        self.is_pretrained_model_has_dropout=''
        self.model=''
        self.model_depth=''
        self.width_mult=''
        self.modality=''
        self.resnet_shortcut=''
        self.n_classes=0
        self.n_finetune_classes=0
        self.arch=''
        self.scales=''
        self.initial_scale=1.0
        self.n_scales=5
        self.scale_step=0.84089641525
        self.ft_begin_index=0
        self.ft_portion='last_layer'
        self.mean=[114.7748, 107.7354, 99.475]
        self.std=[38.7568578, 37.88248729, 40.02898126]
        self.resnext_cardinality=32  
        self.pretrain_modality='RGB'
        self.no_cuda=no_cuda 
        self.norm_value=1 
        self.no_mean_norm=False
        self.std_norm=False 
        self.sample_size=112 
        self.manual_seed=1
        self.stride_len=1
        self.no_train=True

        self.feature_extracting=True
        self.resnext_dropout=True
        self.resnext_dropout_prob=.5

        self.resume_path_det=''
        self.sample_duration_det=8
        self.pretrain_path_det=path_det
        self.is_pretrained_det_model_has_dropout=False
        self.model_det='resnetl'
        self.model_depth_det=10
        self.width_mult_det=.5
        self.modality_det='RGB'
        self.resnet_shortcut_det='A'
        self.n_classes_det=2
        self.n_finetune_classes_det=2
        self.det_counter=2 
        self.det_queue_size=4
        self.det_strategy='median'


        self.clf_queue_size=16      
        self.resume_path_clf=''#path_clf
        self.sample_duration_clf=16
        self.pretrain_path_clf=path_clf
        self.is_pretrained_clf_model_has_dropout=True
        self.model_clf='resnext'
        self.model_depth_clf=101
        self.width_mult_clf=1.0
        self.modality_clf='RGB'
        self.resnet_shortcut_clf='B'
        self.n_classes_clf=n_classes_clf
        self.n_finetune_classes_clf=n_classes_clf
        self.clf_strategy='median'
        self.clf_threshold_pre=1
        self.clf_threshold_final=.15
        self.clf_strategy='median'
        self.with_egogesture=False

        #self.extract_roi_period=200        ### used in in inference time

def load_models(opt):
    opt.resume_path = opt.resume_path_det
    opt.pretrain_path = opt.pretrain_path_det
    opt.is_pretrained_model_has_dropout=opt.is_pretrained_det_model_has_dropout
    opt.sample_duration = opt.sample_duration_det
    opt.model = opt.model_det
    opt.model_depth = opt.model_depth_det
    opt.width_mult = opt.width_mult_det
    opt.modality = opt.modality_det
    opt.resnet_shortcut = opt.resnet_shortcut_det
    opt.n_classes = opt.n_classes_det
    opt.n_finetune_classes = opt.n_finetune_classes_det

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    #opt.scales = [opt.initial_scale]                      ###
    #for i in range(1, opt.n_scales):                      ###
    #    opt.scales.append(opt.scales[-1] * opt.scale_step)    ####
    opt.arch = '{}'.format(opt.model)
   # opt.mean = get_mean(opt.norm_value)
   # opt.std = get_std(opt.norm_value)


 #    with open(os.path.join(opt.result_path, 'opts_det.json'), 'w') as opt_file:
 #    with open(os.path.join('results/', 'opts_det.json'), 'w') as opt_file:
 #        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    
    detector = generate_model(opt)
    if isinstance(detector,tuple):
        detector, parameters = detector
    if not opt.no_cuda:
        detector = detector.cuda()
    else:
        detector=detector.cpu()

    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path,map_location=torch.device('cpu'))

        detector.load_state_dict(checkpoint['state_dict'])

   # print('Model 1 \n', detector)
    pytorch_total_params = sum(p.numel() for p in detector.parameters() if
                               p.requires_grad)
   # print("Total number of trainable parameters: ", pytorch_total_params)

    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.is_pretrained_model_has_dropout=opt.is_pretrained_clf_model_has_dropout
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.width_mult = opt.width_mult_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    #opt.scales = [opt.initial_scale]
    #for i in range(1, opt.n_scales):
    #    opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
  #  opt.mean = get_mean(opt.norm_value)
  #  opt.std = get_std(opt.norm_value)

   # print(opt)
 #    with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
 #        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    

    classifier=generate_model(opt)
    if isinstance(classifier,tuple):
        classifier,parameters=classifier
    
    if not opt.no_cuda:
        classifier = classifier.cuda()
    else:
        classifier=classifier.cpu()

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path,map_location=torch.device('cpu'))

        classifier.load_state_dict(checkpoint['state_dict'])

   # print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
   # print("Total number of trainable parameters: ", pytorch_total_params)

    return detector, classifier