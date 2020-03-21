import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import os
import cv2
import numpy as np
from torchvision import datasets, models, transforms


label_len = 36
vocab =  "<,.+:-?$ 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ>"
# start symbol <
# end symbol >
char2token = {"PAD":0}
token2char = {0:"PAD"}
for i, c in enumerate(vocab):
    char2token[c] = i+1
    token2char[i+1] = c


def illegal(label):
    if len(label) > label_len-1:
        return True
    for l in label:
        if l not in vocab[1:-1]:
            return True
    return False


class ListDataset(Dataset):
    def __init__(self, fname):
        self.lines = []
        if not isinstance(fname, list):
            fname = [fname]
        for f in fname:
            lines = open(f).readlines()
            self.lines += [i for i in lines if not illegal(i.strip('\n').split('\t')[1])]

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        '''
        line: image path\tlabel
        '''
        line = self.lines[index]
        img_path, label_y_str = line.strip('\n').split('\t')
        img = cv2.imread(img_path) / 255.
        # Channels-first
        img = np.transpose(img, (2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float()
        label = np.zeros(label_len, dtype=int)
        for i, c in enumerate('<'+label_y_str):
            label[i] = char2token[c]
        label = torch.from_numpy(label)

        label_y = np.zeros(label_len, dtype=int)
        for i, c in enumerate(label_y_str+'>'):
            label_y[i] = char2token[c]
        label_y = torch.from_numpy(label_y) 
        
        return img, label_y, label

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, trg_y, trg, pad=0):
        self.imgs = Variable(imgs.cuda(), requires_grad=False)
        self.src_mask = Variable(torch.from_numpy(np.ones([imgs.size(0), 1, 36], dtype=np.bool)).cuda())
        if trg is not None:
            self.trg = Variable(trg.cuda(), requires_grad=False)
            self.trg_y = Variable(trg_y.cuda(), requires_grad=False)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return Variable(tgt_mask.cuda(), requires_grad=False)

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, name):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.name = name
    def forward(self, x):
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name is self.name:
                b = x.size(0)
                c = x.size(1)
                return x.view(b, c, -1).permute(0, 2, 1)
        return None

if __name__=='__main__':
    listdataset = ListDataset('your-lines')
    dataloader = torch.utils.data.DataLoader(listdataset, batch_size=2, shuffle=False, num_workers=0)
    for epoch in range(1):
        for batch_i, (imgs, labels_y, labels) in enumerate(dataloader):
            continue


















