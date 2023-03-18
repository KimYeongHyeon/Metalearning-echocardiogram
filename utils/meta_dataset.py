

import os
import pandas as pd
import numpy as np
import glob
import collections
import cv2
from operator import index

import torch
from torch.utils.data import Dataset

import utils
from utils.heatmaps import *

class EchoDataset_Meta_heatmap(Dataset):
    def __init__(self, root_dir, task_list, transforms=None, 
                split='train', shot=5, num_channels=1, ):
        self.sampler = Sampler(root_dir, task_list, transforms, split, shot, num_channels, )
        self.data = self.sampler.sample()

    def __len__(self):
        return len(self.data['label'])
    
    def __getitem__(self, idx):
        sample = {'data': self.data['data'][idx].to(torch.float32), 
                  'label': torch.tensor(self.data['label'][idx]).to(torch.float32), 
                  'shape': self.data['shape'][idx],
        }
                #   'task': self.data['task'][idx]}
        return sample

    def resample(self):
        self.data = self.sampler.sample()
        # print('resampled')
        # print(self.data['label'][0])
        

class Sampler():
    def __init__(self, root_dir,
                task_list,
                 transforms=None,
                 split='train',
                 shot=5,
                 num_channels=1,
                 ):
        ''' 
        Inputs: 
            root_dir(str): path to root including patient directories and labels (csv)
        
        '''
        assert split in ['train', 'val', 'test', 'train+val', '5', '10', '20', '30', 'train_100']

        self.root_dir = root_dir
        # self.tasks = sorted(glob.glob(os.path.join(self.root, "*"))) # get all folders under root
        # self.tasks = [task.split('/')[-1] for task in self.tasks]
        self.tasks = task_list 
        
        self.shot = shot
        self.split = split
        self.task_label_list = []
        self.patient_list = [] 
        self.fname_list = [] 
        

        ## 데이터 목록 / 레이블 정보 불러오기
        # TODO: 경로 불러오는 방식 수정
        self.root_dir = os.path.join('/home/ohpely/kohi/code', self.root_dir)
        for task in self.tasks:
            
            patient_list_per_task = glob.glob(os.path.join(self.root_dir, task, self.split, '*'))
            patient_list_per_task = [path.split('/')[-1] for path in patient_list_per_task]
            # patient_list_per_task = glob.glob(os.path.join(task, self.split, '*'))
            self.patient_list.append(patient_list_per_task)
            
            data = pd.read_csv(os.path.join(self.root_dir, task, 'labels.csv'), index_col=0)
            # data = data[data['split']==self.split] # 폴더 구성을 어떻게 할지에 따라 달라짐
            data = data[data['FileName'].apply(lambda x: x in patient_list_per_task)]
            self.task_label_list.append(data)

            self.fname_list.append(data['FileName'].unique().tolist())

        self.transforms = transforms

        self.calc_dict = {
            'PLAX': ['1','2','3','4'],
            'PSAX': ['1','2','3','4'],
            '2CH': ['1','2','3','4'],
            '4CH': ['1','2','3','4']
        }

        self.num_channels = num_channels
        
    def sample(self):
        self.data = []
        self.label = []
        # 태스크(뷰)별로 수행
        data_list = []
        label_list = []
        shape_list = []
        task_list = []
        for task, patient_list_per_task, task_label_per_task, fname_per_task in zip(self.tasks, self.patient_list, self.task_label_list, self.fname_list):
            task_list.append(task)
            # 태스크 데이터 갯수중 랜덤하게 shot만큼 데이터 불러옴
            data_idx = np.random.choice(range(len(patient_list_per_task)), self.shot, replace=False)
            # print(data_idx)
            sampled_fname_per_task = np.array(fname_per_task)[data_idx]
            
            task_df = task_label_per_task[task_label_per_task['FileName'].apply(lambda x: x in sampled_fname_per_task)].reset_index(drop=True)
            # 이미지별로 파일 불러옴
            # print(task_df)
            for filename in task_df['FileName'].unique():
                one_patient_df = task_df[task_df['FileName'] == filename].reset_index(drop=True)
                image = cv2.imread(
                    os.path.join(self.root_dir, task, self.split, one_patient_df['FileName'].unique()[0])
                )
                

                label = []
                for c in self.calc_dict[task]:
                    try:
                        c_df = one_patient_df[one_patient_df['Calc'] == int(c)]
                        coor = c_df[['X', 'Y']].to_numpy().squeeze().astype(float)
                        label.append((coor[0], coor[1]))
                        
                    except Exception as e:
                        print(e)
                        print(one_patient_df)
                        label.append((-1,-1))
                
                shape_list.append(image.shape[:-1])
            
                if self.transforms:
                    transformed = self.transforms(image=image, keypoints=label)
                    image, label = transformed['image'], transformed['keypoints']
                data_list.append(image)
                label_list.append(label)
        
        sample = {
            "data": torch.stack(data_list),
            "label": np.array(label_list),
            "shape": np.array(shape_list),
            "task":np.array(task_list)
        }
        return sample
