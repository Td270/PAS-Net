from PIL import Image
import torch
from torch.utils.data import Dataset
import random


class MyDataSet(Dataset):

    def __init__(self, txt_path=None, ki=None, K=5, typ='train', transform=None, rand=False):

        self.all_data_info = self.get_img_info(txt_path)
        if rand:
            random.seed(1)
            random.shuffle(self.all_data_info)
        leng = len(self.all_data_info)
        fold_size = leng // K
        if typ == 'val':
            self.data_info = self.all_data_info[fold_size * ki: fold_size * (ki + 1)]
        elif typ == 'train':
            self.data_info = self.all_data_info[: fold_size * ki] + self.all_data_info[fold_size * (ki + 1):]

        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):

        img_pth, label = self.data_info[index]
        img = Image.open(img_pth).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def get_img_info(txt_path):

        data_info=[]
        data = open(txt_path,'r')
        data_lines = data.readlines()
        for data_line in data_lines:
            data_line = data_line.split(';')
            label = int(data_line[0])
            img_path = data_line[1].strip('\n')
            data_info.append((img_path,label))
        return data_info




