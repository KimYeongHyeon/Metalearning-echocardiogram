

import os
from turtle import back
import pandas as pd
import numpy as np
import collections
import cv2
from operator import index

import torch
from torch.utils.data import Dataset
import glob

import utils
from utils.heatmaps import *

class EchoDataset_heatmap(Dataset):
    def __init__(self, root,
                 transforms=None,
                 split='train',
                 num_channels=1,
                 ):
        ''' 
        Inputs: 
            root(str): path to root including patient directories and labels (csv)
        
        '''
        assert split in ['train', 'val', 'test', 'train_100']
        self.root = os.path.join(root,split)
        self.data = pd.read_csv(os.path.join(root, 'labels.csv'), index_col=0)
        
        # 환자 폴더 받아오기
        patient_list = [patient.split('/')[-1] for patient in glob.glob(os.path.join(self.root, '*.png'))]
        
        # 환자 폴더가 존재하는 라벨 정보만 받아오기
        self.data = self.data[self.data['FileName'].apply(lambda x: x in patient_list)]
        self.fname = self.data['FileName'].unique().tolist()

        self.transforms = transforms

        # self.calc_list = ['1','2','3','4']
        self.calc_list = [1,2,3,4]
        # self.calc_list = self.data['Calc'].unique()

        self.num_channels = num_channels
        
    def __len__(self):
        return len(self.fname)

    def __getitem__(self, idx):
        df = self.data[self.data['FileName'] == self.fname[idx]]
        df = df.sort_values(by=['Calc']).reset_index(drop=True)

        image = self.fname[idx]
        image = cv2.imread(os.path.join(self.root, image))#, cv2.IMREAD_GRAYSCALE)
        
        ori_height, ori_width, _ = image.shape
        label = []
        for c in self.calc_list:
            try:
                coor = df[df['Calc'] == c][['X','Y']].to_numpy().squeeze().astype(float)
                label.append([*coor])

            except Exception as e:
                print(e)
                label.append((-1,-1))


        if self.transforms:
            transformed = self.transforms(image=image, keypoints=label)
            image, label = transformed['image'], transformed['keypoints']

        # 3 채널로 변경
        # image = np.array([image]*self.num_channels)
        data = torch.tensor(image)
        label = torch.tensor(label)
        
        # height, width = image.shape[1:]
        # label = torch.tensor(label).reshape(-1).to(torch.float32) # flatten 상태의 포인트를 예측하기 위해 reshape -1 를 수행
        sample = {
            'data':data.to(torch.float32),
            'label':label.to(torch.float32), 
            'id': df['FileName'][0],
            'height':ori_height,
            'width':ori_width
        }

        return sample
