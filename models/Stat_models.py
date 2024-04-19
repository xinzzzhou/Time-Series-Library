import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pmdarima as pm
import threading
from sklearn.ensemble import GradientBoostingRegressor
import math
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF
import pandas as pd

class Autoarima(nn.Module):
    def __init__(self, configs):
        super(Autoarima, self).__init__()
        self.pred_len = configs.pred_len
        self.season = configs.season
        self.freq = configs.freq
        
    def forward(self, x, y):
        sf = StatsForecast(models = [AutoARIMA(season_length = self.season)], freq = self.freq)
        a = x.squeeze().numpy()
        df = pd.DataFrame(a)
        sf.fit(df)
        sf.predict(h=self.pred_len)
        return sf.predictions, y[:,-self.pred_len:,:]
        
class Random(nn.Module):
    def __init__(self, configs):
        super(Random, self).__init__()
        self.pred_len = configs.pred_len
        self.distribution = configs.distribution
    
    def forward(self, x, y):
        if len(y[0]) <= 0:
            raise ValueError("Steps should be a positive integer.")
        shape = y[:,-self.pred_len:,:].shape
        if self.distribution=='Uniform':
            prediction = torch.rand(shape)
        elif self.distribution=='Normal':
            prediction = torch.randn(shape)
        elif self.distribution=='Gaussian':
            mean=0.2
            stddev=0.1
            prediction = torch.randn(shape) * stddev + mean
        else:
            prediction = np.random.randn(x.shape[0],self.pred_len,x.shape[2])
        return prediction, y[:,-self.pred_len:,:]
    
class Constant(nn.Module):
    def __init__(self, configs):
        super(Constant, self).__init__()
        self.pred_len = configs.pred_len
        self.data = configs.data
        self.constant = configs.constant
        # self.mean = configs.mean
        
    def forward(self, x, y, mean):
        if self.constant == 'mean':
            prediction = np.broadcast_to(int(mean), y.shape)
        else: # should be number
            prediction = np.broadcast_to(int(self.constant), y.shape)
            if 'wiki' in self.data:
                prediction = np.log1p(prediction)
        return prediction[:,-self.pred_len:,:], y[:,-self.pred_len:,:]
    
class Naive(nn.Module):
    def __init__(self, configs):
        super(Naive, self).__init__()
        self.pred_len = configs.pred_len
        
    def forward(self, x, y):
        return x[:,-self.pred_len:,:], y[:,-self.pred_len:,:] # [B, L, D]


class Naive_repeat(nn.Module):
    def __init__(self, configs):
        super(Naive_repeat, self).__init__()
        self.pred_len = configs.pred_len
        
    def forward(self, x, y):
        B,L,D = x.shape
        selected_data = x[:, -1, :]
        prediction = np.repeat(selected_data[:, np.newaxis, :], self.pred_len, axis=1)
        # x = x[:,-1,:].reshape(B,1,D).repeat(self.pred_len)
        return prediction, y[:,-self.pred_len:,:] # [B, L, D]

class Naive_seasonal(nn.Module):
    def __init__(self, configs):
        super(Naive_seasonal, self).__init__()
        self.pred_len = configs.pred_len
        self.season = configs.season
        
    def forward(self, x, y):
        times = math.ceil(self.pred_len/self.season)
        prediction = x[:,-times*self.season:,:]
        return prediction[:,-self.pred_len:,:], y[:,-self.pred_len:,:] # [B, L, D]
    
class Naive_thread(threading.Thread):
    def __init__(self,func,args=()):
        super(Naive_thread,self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.results = self.func(*self.args)
    
    def return_result(self):
        threading.Thread.join(self)
        return self.results

def _arima(seq,pred_len,bt,i):
    model = pm.auto_arima(seq)
    forecasts = model.predict(pred_len) 
    return forecasts,bt,i

class Arima(nn.Module):
    """
    Extremely slow, please sample < 0.1
    """
    def __init__(self, configs):
        super(Arima, self).__init__()
        self.pred_len = configs.pred_len
        
    def forward(self, x, y):
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        threads = []
        for bt,seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]
                one_seq = Naive_thread(func=_arima,args=(seq,self.pred_len,bt,i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast,bt,i = every_thread.return_result()
            result[bt,:,i] = forcast

        return result # [B, L, D]

def _sarima(season,seq,pred_len,bt,i):
    model = pm.auto_arima(seq, seasonal=True, m=season)
    forecasts = model.predict(pred_len) 
    return forecasts,bt,i

class SArima(nn.Module):
    """
    Extremely extremely slow, please sample < 0.01
    """
    def __init__(self, configs):
        super(SArima, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.season = 24
        if 'Ettm' in configs.data_path:
            self.season = 12
        elif 'ILI' in configs.data_path:
            self.season = 1
        if self.season >= self.seq_len:
            self.season = 1

    def forward(self, x):
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        threads = []
        for bt,seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]
                one_seq = Naive_thread(func=_sarima,args=(self.season,seq,self.pred_len,bt,i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast,bt,i = every_thread.return_result()
            result[bt,:,i] = forcast
        return result # [B, L, D]

def _gbrt(seq,seq_len,pred_len,bt,i):
    model = GradientBoostingRegressor()
    model.fit(np.arange(seq_len).reshape(-1,1),seq.reshape(-1,1))
    forecasts = model.predict(np.arange(seq_len,seq_len+pred_len).reshape(-1,1))  
    return forecasts,bt,i

class GBRT(nn.Module):
    def __init__(self, configs):
        super(GBRT, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
    
    def forward(self, x):
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        threads = []
        for bt,seqs in tqdm(enumerate(x)):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]
                one_seq = Naive_thread(func=_gbrt,args=(seq,self.seq_len,self.pred_len,bt,i))
                threads.append(one_seq)
                threads[-1].start()
        for every_thread in tqdm(threads):
            forcast,bt,i = every_thread.return_result()
            result[bt,:,i] = forcast
        return result # [B, L, D]
    