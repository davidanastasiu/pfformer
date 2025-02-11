import time,os,sys
import math

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from ..utils.utils2 import (
    log_std_denorm_dataset,
    sin_date,
    cos_date,
    log_std_normalization_1,
)
from .PFformer_model import EncoderLSTM, DecoderLSTM
from utils.metric import metric_rolling
from datetime import datetime, timedelta
import zipfile
random.seed('a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PFformer_I:

    def __init__(self, opt):

        self.opt = opt
        self.sensor_id = opt.stream_sensor
       
        self.train_days = opt.input_len
        self.predict_days = opt.output_len  
        self.output_dim = opt.output_dim
        self.hidden_dim = opt.hidden_dim
        self.is_prob_feature = 1 
        self.TrainEnd = opt.model
        self.ind_dim = opt.r_shift
        self.is_stream = opt.is_stream          
        self.is_over_sampling = 1

        self.batchsize = opt.batchsize
        self.epochs = opt.epochs
        self.layer_dim = opt.layer

        self.encoder = EncoderLSTM(self.opt).to(device)
        self.decoder = DecoderLSTM(self.opt).to(device)         

        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.val_dir = os.path.join(self.opt.outf, self.opt.name, 'val')
        self.test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')    

    def std_denorm_dataset(self, predict_y0):
        
        pre_y = []
        a2 = log_std_denorm_dataset(self.mean, self.std, predict_y0, pre_y)

        return a2

    def inference_test(self, x_test, y_input1):     
        
        y_predict = []
        d_out = torch.tensor([]).to(device)
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            
            x_test = torch.from_numpy(np.array(x_test, np.float32)).to(device)
            y_input1 = torch.from_numpy(np.array(y_input1, np.float32)).to(device)

            h0 = torch.zeros(self.layer_dim, x_test.size(0), self.hidden_dim).to(device)
            c0 = torch.zeros(self.layer_dim, x_test.size(0), self.hidden_dim).to(device)                     
            encoder_h, encoder_c = self.encoder(x_test, h0, c0)   
            out1, out2 = self.decoder(y_input1, x_test, encoder_h, encoder_c)  
                
            y_predict.extend(out1)
            y_predict = [y_predict[i].item() for i in range(len(y_predict))]
            y_predict = np.array(y_predict).reshape(1,-1) 
            
        return y_predict


    def model_load(self,zipf):       
        
        with zipfile.ZipFile(zipf, "r") as file:
            file.extract("Norm.txt")
        norm = np.loadtxt('Norm.txt',dtype=float,delimiter=None)
        os.remove('Norm.txt')
        print("norm is: ", norm)
        self.mean = norm[0]
        self.std = norm[1]
        self.R_mean = norm[2]
        self.R_std = norm[3]
        
        with zipfile.ZipFile(zipf, "r") as archive:
            with archive.open("PFformer_encoder.pt","r") as pt_file:
                self.encoder.load_state_dict(torch.load(pt_file), strict=False)    
        with zipfile.ZipFile(zipf, "r") as archive:
            with archive.open("PFformer_decoder.pt","r") as pt_file:
                self.decoder.load_state_dict(torch.load(pt_file), strict=False)   

    def get_data(self, test_point):

        # data prepare
        trainX = pd.read_csv('./data_provider/datasets/'+ self.opt.stream_sensor+'.csv', sep='\t')
        trainX.columns = ["id", "datetime", "value"] 
        trainX.sort_values('datetime', inplace=True),
        R_X = pd.read_csv('./data_provider/datasets/'+ self.opt.rain_sensor+'.csv', sep='\t')
        R_X.columns = ["id", "datetime", "value"] 
        R_X.sort_values('datetime', inplace=True)
        
        # read stream data        
        point = trainX[trainX["datetime"]==test_point].index.values[0]
        stream_data = trainX[point-self.train_days:point]["value"].values.tolist()
        gt = trainX[point:point+self.predict_days]["value"].values.tolist()
        
        # read rain data
        R_X = pd.read_csv('./data_provider/datasets/'+self.opt.rain_sensor+'.csv', sep='\t')
        R_X.columns = ["id", "datetime", "value"] 
        point = R_X[R_X["datetime"]==test_point].index.values[0]
        rain_data = R_X[point-self.train_days-self.ind_dim:point]["value"].values.tolist()
        NN = np.isnan(rain_data).any() or np.isnan(stream_data).any() or np.isnan(gt).any() 
        if NN:
            gt = None 
       
        return stream_data, rain_data, gt
    
    def test_single(self, test_point):
        
        stream_data, indicator_data, gt = self.get_data(test_point)  
        if gt is None:
            pre = None
            return pre, gt
        pre = self.predict(test_point, stream_data, indicator_data)
        
        return pre, gt
    
    def predict(self, test_point, stream_data, rain_data=None):
        
        
        time_str = test_point
        self.encoder.eval()
        self.decoder.eval()
        test_predict = np.zeros(self.predict_days*self.output_dim)
                                
        test_month = []
        test_day = []
        test_hour = []
        new_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        for i in range(self.predict_days):
            new_time_temp = new_time + timedelta(minutes=15)
            new_time = new_time.strftime("%Y-%m-%d %H:%M:%S")
            
            month = int(new_time[5:7])
            day = int(new_time[8:10])
            hour = int(new_time[11:13])
   
            test_month.append(month)
            test_day.append(day)
            test_hour.append(hour)  
            
            new_time = new_time_temp
            
        y2 = cos_date(test_month, test_day, test_hour) 
        y2 = [[ff] for ff in y2]
 
        y3 = sin_date(test_month, test_day, test_hour) 
        y3 = [[ff] for ff in y3]        
        
        y_input1 = np.array([np.concatenate((y2,y3),1)])

        x_test = np.array(log_std_normalization_1(stream_data, self.mean, self.std), np.float32).reshape(self.train_days,-1)

        if rain_data is None:
            raise ValueError("Rain data is required.")
        y4 = np.array(log_std_normalization_1(rain_data, self.R_mean, self.R_std)).reshape(len(rain_data), -1)
        x_test = np.concatenate((x_test, y4[-self.train_days:]), 1)    
        
        RR = y4.tolist()
        for iR in range(self.ind_dim):
            RR = RR[:-1]
            x_test = np.concatenate((x_test, np.array(RR[-self.train_days:])), 1)                    
        
        x_test = [x_test]
        y_predict = self.inference_test(x_test, y_input1)
        y_predict = np.array(y_predict.tolist())[0]
        y_predict = [y_predict[i].item() for i in range(len(y_predict))]
        test_predict = np.array(self.std_denorm_dataset(y_predict))
        diff_predict = []
        test_predict = (test_predict + abs(test_predict))/2 
        
        return test_predict
    
    def Inference(self):
        test_set = pd.read_csv('./data_provider/datasets/test_timestamps_24avg.tsv',sep='\t')
        test_points = test_set["Hold Out Start"]
        count = 0
        pre = []
        gt = []
        for testP in test_points:
            predicted, ground_truth = self.test_single(testP)
            if ground_truth is not None:
                count += 1        
                pre.extend(predicted)
                gt.extend(ground_truth)
        print("Valid points: ", count)
        metric_rolling(pre, gt)       
