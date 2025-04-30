import time,os,sys
import math

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from scipy import stats
from ..utils.utils2 import (
    log_std_denorm_dataset,
    cos_date,
    sin_date,
    adjust_learning_rate,
)
from ..utils.metric import metric_rolling
from sklearn.metrics import mean_absolute_percentage_error
from .PFformer_model import EncoderLSTM, DecoderLSTM
import zipfile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PFformer:

    def __init__(self, opt, dataset):
#         super(PFformer, self).__init__(opt)

        self.dataset = dataset
        self.opt = opt
        self.sensor_id = opt.stream_sensor
        self.dataloader = dataset.get_train_data_loader()
        self.trainX = dataset.get_trainX()
        self.val_data = np.array(dataset.get_val_points()).squeeze()
        self.data = dataset.get_data()
        self.sensor_data_norm = dataset.get_sensor_data_norm()
        self.sensor_data_norm_1 = dataset.get_sensor_data_norm1()
        self.R_norm_data = dataset.get_R_sensor_data_norm()
        self.mean = dataset.get_mean()
        self.std = dataset.get_std()
        self.month = dataset.get_month()
        self.day = dataset.get_day()
        self.hour = dataset.get_hour()
        
        self.train_days = opt.input_len
        self.predict_days = opt.output_len  
        self.output_dim = opt.output_dim
        self.hidden_dim = opt.hidden_dim
        self.is_prob_feature = 1 
        self.TrainEnd = opt.model
        self.os = opt.oversampling
        self.r_shift = opt.r_shift
        self.is_stream = opt.is_stream       
        self.is_over_sampling = 1
        self.iterval = opt.os_v
        self.lamda = opt.lamda
        self.beta = opt.beta

        self.batchsize = opt.batchsize
        self.epochs = opt.epochs
        self.layer_dim = opt.layer

        self.encoder = EncoderLSTM(self.opt).to(device)        
        self.decoder = DecoderLSTM(self.opt).to(device)

        self.criterion = nn.MSELoss(reduction='sum')
        self.criterion1 = nn.HuberLoss(reduction='sum')
        self.criterion_KL = nn.KLDivLoss(reduction="sum")
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),self.opt.learning_rate)  
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(),self.opt.learning_rate)  

        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.val_dir = os.path.join(self.opt.outf, self.opt.name, 'val')
        self.test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        
        self.train_loss_list = []
        self.val_loss_list = []
        
    def get_train_loss_list(self):
        
        return self.train_loss_list
    
    def get_val_loss_list(self):
        
        return self.val_loss_list


    def std_denorm_dataset(self, predict_y0, pre_y):
    
        a2 = log_std_denorm_dataset(self.mean, self.std, predict_y0, pre_y)

        return a2

    def inference_test(self, x_test, y_input1):
                                
        y_predict = []
        d_out = torch.tensor([]).to(device)
        self.encoder.eval()
        self.decoder.eval()
        sig = nn.Sigmoid()

        with torch.no_grad():

            
            x_test = torch.from_numpy(np.array(x_test, np.float32)).to(device)
            y_input1 = torch.from_numpy(np.array(y_input1, np.float32)).to(device)

            h0 = torch.zeros(self.layer_dim, x_test.size(0),self.hidden_dim).to(device)
            c0 = torch.zeros(self.layer_dim, x_test.size(0),self.hidden_dim).to(device)
   
            encoder_h, encoder_c = self.encoder(x_test, h0, c0)  
            out1, out2 = self.decoder(y_input1, x_test, encoder_h, encoder_c)     
        
            y_predict.extend(out1)
            y_predict = [y_predict[i].item() for i in range(len(y_predict))]
            y_predict = np.array(y_predict).reshape(1,-1) 
            
        return y_predict


    def test_single(self, test_point):

        self.encoder.eval()
        self.decoder.eval()

        test_predict = np.zeros(self.predict_days*self.output_dim)
                                
        #foot label of test_data
        point = self.trainX[self.trainX["datetime"]==test_point].index.values[0]
        start_num = self.trainX[self.trainX["datetime"]==self.opt.start_point].index.values[0]
        test_point = point - start_num
        pre_gt = self.trainX[point-1:point+self.opt.output_len-1]["value"].values.tolist()
        y = self.trainX[point:point+self.predict_days]["value"]

        b = test_point
        e = test_point+self.predict_days
     
        y2 = cos_date(self.month[b:e], self.day[b:e], self.hour[b:e]) # represent cos(int(data)) here
        y2 = [[ff] for ff in y2]
 
        y3 = sin_date(self.month[b:e], self.day[b:e], self.hour[b:e]) # represent sin(int(data)) here
        y3 = [[ff] for ff in y3]
        
        y_input1 = np.array([np.concatenate((y2,y3),1)])

        #inference
        norm_data = self.sensor_data_norm
        x_test = np.array(self.sensor_data_norm_1[test_point-self.train_days*1:test_point], np.float32).reshape(self.train_days,-1)
        y_test = np.array(norm_data[test_point:test_point+self.predict_days], np.float32).reshape(self.predict_days,-1)
        
        x_test = [x_test]        
        x_test = np.array(x_test) 
                
        y_predict = self.inference_test(x_test, y_input1)
        y_predict = np.array(y_predict.tolist())[0]
        y_predict = [y_predict[i].item() for i in range(len(y_predict))]
    
        pre_gt = self.data[test_point-1:test_point]
        pre_gt = pre_gt[0]

        test_predict = np.array(self.std_denorm_dataset(y_predict, pre_gt))
        test_predict = (test_predict + abs(test_predict))/2
        return test_predict, y

                      

    def generate_single_val_rmse(self, min_RMSE=500):
        
        total = 0 
        val_rmse_list = []
        val_pred_list = []
        val_pred_lists_print = []
        gt_mape_list = []
        val_mape_list = []
        val_points = self.val_data
        test_predict = np.zeros(self.predict_days * self.output_dim)

        non_flag = 0                        
        for i in range(len(val_points)):
            
            val_pred_list_print = []
            val_point = val_points[i]
            test_predict, ground_truth = self.test_single(val_point)  
            rec_predict = test_predict
            
            for j in range(len(rec_predict)):
                temp = [val_point, j, rec_predict[j]]
                val_pred_list.append(temp)
                val_pred_list_print.append(rec_predict[j])
            
            val_pred_lists_print.append(val_pred_list_print)
            val_MSE = np.square(np.subtract(ground_truth, test_predict)).mean() 
            val_RMSE = math.sqrt(val_MSE)
            val_rmse_list.append(val_RMSE)
            total += val_RMSE

            gt_mape_list.extend(ground_truth)
            val_mape_list.extend(test_predict)
            
        name = "%s" % (self.opt.model)
        file_name = os.path.join(self.val_dir, 'validation_timestamps_24avg.tsv')        

        new_min_RMSE = min_RMSE
        
        if total < min_RMSE :
            
            #save_model
            encoder_name = self.expr_dir + '/' + 'PFformer_encoder.pt'
            decoder_name = self.expr_dir + '/' + 'PFformer_decoder.pt'
    
            new_min_RMSE = total
            expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
            c_dir = os.getcwd()
            os.chdir(expr_dir)          
            with zipfile.ZipFile(self.opt.name+".zip", "w") as my_zip:
                with my_zip.open("PFformer_encoder.pt", "w") as data_file:
                    torch.save(self.encoder.state_dict(), data_file)
            with zipfile.ZipFile(self.opt.name+".zip", "a") as my_zip:
                with my_zip.open("PFformer_decoder.pt", "w") as data_file:
                    torch.save(self.decoder.state_dict(), data_file)                    
            os.chdir(c_dir)       
        print("val total RMSE: ", total)
        print("val min RMSE: ", new_min_RMSE)

        if non_flag == 0:
            mape = mean_absolute_percentage_error(np.array(gt_mape_list)+1, np.array(val_mape_list)+1)
        else:
            mape = 100
            
        return total, new_min_RMSE, mape
    
    def model_load(self):
        
        c_dir = os.getcwd()
        os.chdir(self.expr_dir) 
        
        model1 = EncoderLSTM(self.opt).to(device)        
        model2 = DecoderLSTM(self.opt).to(device)
        with zipfile.ZipFile(self.opt.name +'.zip', "r") as archive:
            with archive.open("PFformer_encoder.pt","r") as pt_file:
                model1.load_state_dict(torch.load(pt_file), strict=False)
                print("Importing the best PFformer_encoder pt file:", pt_file)
                
        with zipfile.ZipFile(self.opt.name +'.zip', "r") as archive:
            with archive.open("PFformer_decoder.pt","r") as pt_file:
                model2.load_state_dict(torch.load(pt_file), strict=False)
                print("Importing the best PFformer_decoder pt file:", pt_file)
                
        os.chdir(c_dir)        
        self.encoder = model1
        self.decoder = model2        
    
    def generate_test_rmse_mape(self):
        
        total = 0 
        val_rmse_list = []
        val_pred_list = []
        val_pred_lists_print = []
        gt_mape_list = []
        val_mape_list = []
        val_points = self.test_data
        test_predict = np.zeros(self.predict_days * self.output_dim)

        non_flag = 0     
        start = time.time()
        for i in range(len(val_points)):     
            start = time.time()
            val_pred_list_print = []
            val_point = val_points[i]
            test_predict, ground_truth = self.test_single(val_point)  
            test_predict = (test_predict + abs(test_predict))/2
            rec_predict = test_predict
            val_MSE = np.square(np.subtract(ground_truth, test_predict)).mean() 
            val_RMSE = math.sqrt(val_MSE)
            val_rmse_list.append(val_RMSE)
            total += val_RMSE         
            
            for j in range(len(rec_predict)):
                temp = [val_point, j, rec_predict[j]]
                val_pred_list.append(temp)
                val_pred_list_print.append(rec_predict[j])
            
            val_pred_lists_print.append(val_pred_list_print)
            gt_mape_list.extend(ground_truth)
            val_mape_list.extend(test_predict)
        end = time.time()
        print("Inferencing test points ", len(val_points), " use: ", end-start)
        

        pd_temp = pd.DataFrame(val_pred_list, columns=("start", "No.", "prediction"))
    
        if(self.is_over_sampling == 1):
            OS = "_OS" + str(self.opt.oversampling)
        else:
            OS = "_OS-null"

        watersheds = "Shed-ProbFeature"
                
        basic_path = self.test_dir + '/' + str(self.sensor_id) + OS
        basic_model_path = self.expr_dir + '/' + str(self.sensor_id) + OS    
        
        if self.opt.save == 1:
            aa = pd.DataFrame(data = val_pred_lists_print)
            i_dir = basic_path + '_' + watersheds + str(self.TrainEnd) + '_pred_lists_print.tsv'
            aa.to_csv(i_dir, sep = '\t')
            print("Inferencing result is saved in: ", i_dir)

        if non_flag == 0:
            mape = mean_absolute_percentage_error(np.array(gt_mape_list)+1, np.array(val_mape_list)+1)
        else:
            mape = 100
            
        return total, mape, val_pred_lists_print


    
    def inference(self): 
        start = time.time()
        self.dataset.gen_test_data() # generate test points file and test_data
        end = time.time()
        print("generate test points file and test_data: ", end-start)
        
        # refresh the related values
        self.test_data = np.array(self.dataset.get_test_points()).squeeze(1) # read the test set
        self.data = self.dataset.get_data()
        self.sensor_data_norm = self.dataset.get_sensor_data_norm()
        self.sensor_data_norm_1 = self.dataset.get_sensor_data_norm1()
        self.R_norm_data = self.dataset.get_R_sensor_data_norm()
        self.mean = self.dataset.get_mean()
        self.std = self.dataset.get_std()
        self.month = self.dataset.get_month()
        self.day = self.dataset.get_day()
        self.hour = self.dataset.get_hour()              
        rmse, mape, aa = self.generate_test_rmse_mape() # inference on test set
        return aa


    def compute_metrics(self, aa):
        val_set=pd.read_csv('./data_provider/datasets/test_timestamps_24avg.tsv',sep='\t')
        val_points=val_set["Hold Out Start"]
        trainX = pd.read_csv('./data_provider/datasets/'+self.opt.stream_sensor+'.csv', sep='\t')
        trainX.columns = ["id", "datetime", "value"] 
        count = 0
        for test_point in val_points:
            point = trainX[trainX["datetime"]==test_point].index.values[0]
            NN = np.isnan(trainX[point-self.train_days:point+self.predict_days]["value"]).any() 
            if not NN:
                count += 1
        vals4 = aa
        # compute metrics
        all_GT=[]
        all_PFformer=[]
        loop = 0
        ind = 0
        while loop<len(val_points):
            ii=val_points[loop]
            val_point=val_points[ind]
            point=trainX[trainX["datetime"]==ii].index.values[0]
            x=trainX[point-self.train_days:point+self.predict_days]["value"].values.tolist()    
            if (np.isnan(np.array(x)).any()):
                loop = loop + 1 # id for time list 
                continue
            loop = loop + 1
            if ind >= count - count%100:
                break   
            ind += 1  
            temp_vals4=list(vals4[ind-1])
            all_GT.extend(x[self.train_days:])
            all_PFformer.extend(temp_vals4) 
        metrics = metric_rolling(np.array(all_PFformer), np.array(all_GT))
        return metrics

        
    def train(self):

        num_epochs = self.epochs
        early_stop = 0
        old_val_loss = 1000
        min_RMSE = 500000
        sig = nn.Sigmoid()
        m = nn.Softmax(dim=1)
        dim_in = 3
   
        for epoch in range(num_epochs):
            print_loss_total = 0  # Reset every epoch
            self.encoder.train()
            self.decoder.train()
            random0 = 0
            start = time.time()
            
            for i, batch in enumerate(self.dataloader):
                x_train = [TrainData for TrainData, _ in batch]
                y_train = [TrainLabel for _, TrainLabel in batch]
                y_train = np.array(y_train).squeeze()
                    
                y_train1 = []
                for ii in range(len(y_train)):
                    y_train1.append([yy for yy in [xx[:][1:3] for xx in y_train[ii]]]) 

                y_train0 = []
                for ii in range(len(y_train)):
                    y_train0.append([yy for yy in [xx[:][0] for xx in y_train[ii]]])     
                
                x_train = torch.from_numpy(np.array(x_train, np.float32)).to(device)
                y_train = torch.from_numpy(np.array(y_train0, np.float32)).to(device)
                decoder_input1 = torch.from_numpy(np.array(y_train1,np.float32)).to(device)
                  
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                    
                loss = 0        
                
                h0 = torch.zeros(self.layer_dim, x_train.size(0), self.hidden_dim).to(device)
                c0 = torch.zeros(self.layer_dim, x_train.size(0), self.hidden_dim).to(device)    
                
                # Forward pass
                encoder_h, encoder_c = self.encoder(x_train, h0, c0)   
                out1, out2 = self.decoder(decoder_input1, x_train, encoder_h, encoder_c)  
                
                l_w = max(-1 * np.exp(epoch/45) + 1 + self.lamda, self.beta)
            
                loss = self.criterion(out1, y_train) + l_w*self.criterion(out2[:,:self.iterval],y_train[:,:self.iterval])
                   
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                print_loss_total += loss.item()                    
                
            self.encoder.eval()
            self.decoder.eval()
        
            val_loss, min_RMSE, mape = self.generate_single_val_rmse(min_RMSE)
            self.train_loss_list.append(print_loss_total)
            self.val_loss_list.append(val_loss)
            end = time.time()
            print('-----------Epoch: {}. train_Loss>: {:.6f}. --------------------'.format(epoch, print_loss_total)) 
            print('-----------Epoch: {}. val_Loss_rmse>: {:.6f}. --------------------'.format(epoch, val_loss)) 
            print('-----------Epoch: {}. val_Loss_mape>: {:.6f}. --------------------'.format(epoch, mape))
            print('-----------Epoch: {}. running time>: {:.6f}. --------------------'.format(epoch, end-start))
            #early stop
            if(val_loss > min_RMSE):
                early_stop += 1
            else:
                early_stop = 0
            if(early_stop > 3 ):
                break
            old_val_loss = val_loss     
            
            adjust_learning_rate(self.encoder_optimizer, epoch + 1, self.opt)
            adjust_learning_rate(self.decoder_optimizer, epoch + 1, self.opt)
                                


