import torch
from torch.autograd import Variable
import numpy as np
from model import make_model
from dataset import vocab, char2token, token2char
from dataset import subsequent_mask
import cv2
import sys, os

model = make_model(len(char2token))
model.load_state_dict(torch.load('your-model-path'))
model.cuda()
model.eval()
src_mask=Variable(torch.from_numpy(np.ones([1, 1, 36], dtype=np.bool)).cuda())
SIZE=96

def greedy_decode(src, max_len=36, start_symbol=1):
    global model
    global src_mask
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).long().cuda()
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .long().cuda()))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).long().cuda().fill_(next_word)], dim=1)
        if token2char[next_word.item()] == '>':
            break
    ret = ys.cpu().numpy()[0]
    out = [token2char[i] for i in ret]
    out = "".join(out[1:-1])
    return out

def resize(img):
    h, w, c = img.shape
    if w > h:
        nw, nh = SIZE, int(h * SIZE/w)
        if nh < 10 : nh = 10
        #print(h, w, nh, nw)
        img = cv2.resize(img, (nw, nh))
        a1 = int((SIZE-nh)/2)
        a2= SIZE-nh-a1
        pad1 = np.zeros((a1, SIZE, c), dtype=np.uint8)
        pad2 = np.zeros((a2, SIZE, c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=0)
    else:
        nw, nh = int(w * SIZE/h), SIZE
        if nw < 10 : nw = 10
        #print(h, w, nh, nw)
        img = cv2.resize(img, (nw, nh))
        a1 = int((SIZE-nw)/2)
        a2= SIZE-nw-a1
        pad1 = np.zeros((SIZE, a1, c), dtype=np.uint8)
        pad2 = np.zeros((SIZE, a2, c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=1)
    return img

def do_folder(root):
    hit = 0
    all = 0
    for line in open(root).readlines():
        all += 1
        imp, label = line.strip('\n').split('\t')
        img = cv2.imread(imp)
        img = resize(img) / 255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0).cuda()
        pred = greedy_decode(img)
        if pred != label:
            hit += 1
            print('imp:', imp, 'label:', label, 'pred:', pred, hit, all, hit/all)
    print(hit, all, hit/all)


if __name__ == '__main__':
    do_folder('your-test-lines')
















