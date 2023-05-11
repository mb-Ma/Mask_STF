from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler


class Mask_STDataset(Dataset):
    """
    only for the incrementaly train dataset using mask mechanism.
    """
    def __init__(self, path, normalizer=None, incre_num=1) -> None:
        self.path = path
        self.normalizer = normalizer
        self.incre_num = incre_num

        X_list, Y_list, ts_list, gs_list = [], [], [], []
        for i in range (1, incre_num+1):
            # get the sequential data
            data = np.load(path, allow_pickle=True)['train'+str(i)]
            # x [samples, time, nodes, feature], gs [nodes, nodes]
            x, y, ts, gs = data
            X_list.append(x)
            Y_list.append(y)
            ts_list.append(ts)
            gs_list.append(gs)
        # import pdb; pdb.set_trace()
        # maks previous data with zero and generator mask matrix
        self.num_node = X_list[-1].shape[2]

        # first normalize, then do padding.
        # 1. get all train data, then get the mean-std or max-min
        if normalizer == 'std':
            self.scaler = self.get_MeanStd(X_list)
        elif normalizer == 'max':
            self.scaler = self.get_MaxMin(X_list)
        else:
            self.scaler = None
        
        # 2. do transformation
        for i in range(len(X_list)):
            X_list[i] = self.transform(X_list[i], self.scaler)
            Y_list[i] = self.transform(Y_list[i], self.scaler)

        # 3. do mask 3.1 fill then concatenate them
        _x, _y, mask = [], [], []
        for i in range(len(X_list)-1):
            # import pdb; pdb.set_trace()
            _x.append(np.pad(X_list[i], ((0,0), (0,0), (0, self.num_node-X_list[i].shape[2]), (0,0)), mode='constant'))
            _y.append(np.pad(Y_list[i], ((0,0), (0,0), (0, self.num_node-Y_list[i].shape[2]), (0,0)), mode='constant'))
            mask_matrix = np.concatenate((np.ones((X_list[i].shape[0], X_list[i].shape[2])), np.zeros((X_list[i].shape[0], self.num_node-X_list[i].shape[2]))), axis=1)
            mask.append(mask_matrix)    
        # 3.2 add the last one
        _x.append(X_list[-1])
        _y.append(Y_list[-1])
        mask.append(np.ones((X_list[-1].shape[0], X_list[-1].shape[2])))
        # 3. concatenate all elements
        self.x = np.vstack(_x)
        self.y = np.vstack(_y)
        self.mask = np.vstack(mask)

        self.to_CudaTensor()
        self.gs = gs_list[-1]

    def get_MaxMin(self, X):
        tmp = np.stack([x.reshape(x.shape[0], x.shape[1], -1) for x in X])
        max, min = tmp.max(), tmp.min()
        scaler = MinMax01Scaler(min=min, max=max)
        return scaler
            
    def get_MeanStd(self, X):
        tmp = np.vstack([x.reshape(-1, x.shape[1]) for x in X])
        mean, std = tmp.mean(), tmp.std()
        scaler = StandardScaler(mean, std)
        return scaler

    def transform(self, data, normalizer):
        data = normalizer.transform(data)
        return data
    
    def to_CudaTensor(self):
        cuda = True if torch.cuda.is_available() else False
        TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.x = TensorFloat(self.x)
        self.y = TensorFloat(self.y)
        self.mask = TensorFloat(self.mask)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            x: (N, T, S, F)
            y: (N, T, S, F)
        """
        return self.x[index], self.y[index], self.mask[index]


class STDataset(Dataset):
    """
    STDataset is a dataset for spatio-temporal data.
    
    """
    def __init__(self, path, types, normalizer=None) -> None:
        data = np.load(path, allow_pickle=True)[types]
        self.x, self.y, self.ts, self.gs = data
        print(f'{types}: X shape: {self.x.shape}, Y shape: {self.y.shape}, TS shape: {self.ts.shape}, G shape: {self.gs.shape}')
        # self.x = torch.from_numpy(self.x).float()
        # self.y = torch.from_numpy(self.y).float()
        if normalizer:
            self.transform(normalizer)
        

    def transform(self, normalizer):
        self.x = normalizer.transform(self.x)
        self.y = normalizer.transform(self.y)
        cuda = True if torch.cuda.is_available() else False
        TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.x, self.y = TensorFloat(self.x), TensorFloat(self.y)
    

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            x: (N, T, S, F)
            y: (N, T, S, F)
        """
        return self.x[index], self.y[index]
    
def normalize_dataset(data, normalizer):
    """
    TODO: add more normalizer / how to normalize the data
    """
    if normalizer == 'std':
        mean = data.mean()
        std = data.std()
        scaler = StandardScaler(mean, std)
    return scaler
    

def get_dataloader(path, batch_size, incre_num=1, if_last=False):
    '''
    incre_num: the number of incr       emental task
    if_last: if use the last incremental data
    '''
    if incre_num == 1:
        train = STDataset(path, 'train')
    elif if_last:
        train = STDataset(path, 'train'+str(incre_num))
    else:
        train = Mask_STDataset(path, normalizer='std', incre_num=incre_num)
    scaler = train.scaler
    gs = [train.gs]
    train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid = STDataset(path, 'valid', scaler)
    gs.append(valid.gs)
    valid = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=0)
    test = STDataset(path, 'test', scaler)
    gs.append(test.gs)
    test  = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0)
    return train, valid, test, scaler, gs

if __name__ == '__main__':
    # path config
    root = Path('../../data/processed')
    dataset = STDataset(Path(root / 'meta-noEx-20-5-10_12.npz'), 'train')
    get_dataloader(Path(root / 'meta-noEx-20-5-10_12.npz'), 32)
