

import os
from turtle import back
import pandas as pd
import numpy as np
import collections
import cv2
from operator import index

import torch
from torch.utils.data import Dataset


import utils
from utils.heatmaps import *

class EchoDataset(Dataset):
    def __init__(self, root,
                 transforms=None,
                 split='train',
                 ):
        ''' 
        Inputs: 
            root(str): path to root including patient directories and labels (csv)
        
        '''
        assert split in ['train', 'val', 'test']
        self.root = root
        self.data = pd.read_csv(os.path.join(self.root, 'labels.csv'), index_col=0)
        self.data = self.data[self.data['split']==split]
        # 환자 폴더 받아오기
        patient_list = [dirname for dirname in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, dirname))]
        # 환자 폴더가 존재하는 라벨 정보만 받아오기
        
        self.data = self.data[self.data['FileName'].apply(lambda x: x in patient_list)]
        self.fname = self.data['FileName'].unique().tolist()

        self.transforms = transforms

        self.calc_list = ['IVSd', 'LVPWd', 'LVIDd']
        # self.calc_list = self.data['Calc'].unique()

    def __len__(self):
        return len(self.fname)

    def __getitem__(self, idx):
        df = self.data[self.data['FileName'] == self.fname[idx]]
        df = df.sort_values(by=['Calc']).reset_index(drop=True)

        image = os.path.join(self.root, str(self.fname[idx]),
                             str(df.loc[0, 'Frame']).zfill(4)+'.png')
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        
        ori_height, ori_width = image.shape
        label = []
        for c in self.calc_list:
            try:
                c_df = df[df['Calc'] == c]
                coor = c_df[['X1', 'X2', 'Y1', 'Y2']].to_numpy().squeeze().astype(float)
                # xy순으로 묶음
                label.append((coor[0], coor[2]))
                label.append((coor[1], coor[3]))

            except Exception as e:
                print(e)
                label.append([-1,-1,-1,-1])

        if self.transforms:
            transformed = self.transforms(image=image, keypoints=label)
            image, label = transformed['image'], transformed['keypoints']

        # 3 채널로 변경
        image = np.array([image]*3)
        data = torch.tensor(image)

        # height, width = image.shape[1:]
        label = torch.tensor(label).reshape(-1).to(torch.float32) # flatten 상태의 포인트를 예측하기 위해 reshape -1 를 수행
        sample = {
            'data':data.to(torch.float32),
            'label':label, 
            'id': df['FileName'][0],
            'height':ori_height,
            'width':ori_width
        }

        return sample

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
        assert split in ['train', 'val', 'test']
        self.root = root
        self.data = pd.read_csv(os.path.join(self.root, 'labels_all.csv'), index_col=0)
        self.data = self.data[self.data['split']==split]
        # 환자 폴더 받아오기
        patient_list = [dirname for dirname in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, dirname))]
        # 환자 폴더가 존재하는 라벨 정보만 받아오기
        
        self.data = self.data[self.data['FileName'].apply(lambda x: x in patient_list)]
        self.fname = self.data['FileName'].unique().tolist()

        self.transforms = transforms

        self.calc_list = ['IVSd','LVIDd','LVPWd']
        # self.calc_list = self.data['Calc'].unique()

        self.num_channels = num_channels

    def __len__(self):
        return len(self.fname)

    def __getitem__(self, idx):
        df = self.data[self.data['FileName'] == self.fname[idx]]
        df = df.sort_values(by=['Calc']).reset_index(drop=True)

        image = os.path.join(self.root, str(self.fname[idx]),
                             str(df.loc[0, 'Frame']).zfill(4)+'.png')
        image = cv2.imread(image)#, cv2.IMREAD_GRAYSCALE)
        
        ori_height, ori_width, _ = image.shape
        label = []
        for c in self.calc_list:
            try:
                c_df = df[df['Calc'] == c]
                coor = c_df[['X1', 'X2', 'Y1', 'Y2']].to_numpy().squeeze().astype(float)
                
                # xy순으로 묶음
                if c == 'IVSd':
                    label.append((coor[1], coor[3]))

                if c == 'LVIDd':
                    label.append((coor[1], coor[3]))
                    label.append((coor[0], coor[2]))
                    
                if c == 'LVPWd':
                    label.append((coor[0], coor[2]))

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
