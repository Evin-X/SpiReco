import os
from torch.utils.data import Dataset
import numpy as np


def TFI(spike, middle, window=50):
    '''
    Modified from the original codes of Rui Zhao (https://github.com/ruizhao26)
    '''
    C, H, W = spike.shape
    lindex, rindex = np.zeros([H, W]), np.zeros([H, W])
    l, r = middle+1, middle+1
    for r in range(middle+1, middle + window+1): 
        l = l - 1
        if l>=0:
            newpos = spike[l, :, :]*(1 - np.sign(lindex)) 
            distance = l*newpos
            lindex += distance
        if r<C:
            newpos = spike[r, :, :]*(1 - np.sign(rindex))
            distance = r*newpos
            rindex += distance
        if l<0 and r>=C:
            break

    rindex[rindex==0] = window+middle
    lindex[lindex==0] = middle-window
    interval = rindex - lindex
    tfi = 1.0 / interval 

    return tfi.astype(np.float32) 


class Spikedata(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir 
        self.T = 7

    def __getitem__(self, index):
        item = {}
        data = np.load(self.root_dir + '/{}.npz'.format(index))
        item['label'] = int(data['label'])
        spkmap = []
        dur = 50

        for t in range(self.T):
            tmp = TFI(data['spk'][0:100,13:237,13:237], 50+t, dur) # input size T*224*224
            tmp = np.where(tmp > 0.5, 0.5, tmp) / 0.5 # benefit for training
            spkmap.append(tmp)
        item['spkmap'] = np.array(spkmap)

        return item

    def __len__(self) -> int:
        return len(os.listdir(self.root_dir))


def build_dataset(path):
    train_path = path + '/trainset'
    val_path = path + '/testset'
    train_dataset = Spikedata(root_dir=train_path)
    val_dataset = Spikedata(root_dir=val_path)

    return train_dataset, val_dataset

