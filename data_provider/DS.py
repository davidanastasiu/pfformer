# time_data = gen_time_feature(opt, sensor_datetime)
# cos_date = [math.cos(ff*2*(math.pi)/365) for ff in time_data]
# sin_date = [math.cos(ff*2*(math.pi)/365) for ff in time_data]
# tag_meaning = {0:"none", 1:"available", 2:"val point", 3:"val near point", 4:"train point"}

import numpy as np
import random
from ..utils.utils2 import (
    diff_order_1,
    log_std_normalization,
    gen_month_tag,
    gen_time_feature,
    cos_date,
    sin_date,
    RnnDataset,
    log_std_normalization_1,
)
import os
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader


class DS:

    def __init__(self, opt, trainX, R_X):
        #         super(DS, self).__init__()

        self.opt = opt
        self.trainX = trainX
        self.R_X = R_X
        self.mean = 0
        self.std = 0
        self.mini = 0
        self.R_mean = 0
        self.R_std = 0
        self.is_stream = 1
        self.tag = []
        self.sensor_data = []
        self.diff_data = []
        self.data = []
        self.data_time = []
        self.sensor_data_norm = []
        self.sensor_data_norm1 = []        
               
        self.R_sensor_data = []
        self.R_data = []
        self.R_data_time = []
        self.R_sensor_data_norm = []
        self.R_sensor_data_norm1 = []
        
        self.val_points = []
        self.test_points = []
        self.start_num = 0
        self.test_start_time = self.opt.test_start
        self.test_end_time = self.opt.test_end
        self.opt_hinter_dim = opt.watershed
        self.ind_dim = opt.r_shift
        self.gm3 = GaussianMixture(n_components=3,)
        
        self.is_over_sampling = 0
        self.norm_percen = 0
        self.oversampling = int(opt.oversampling)
            
        self.train_days = self.opt.input_len
        self.predict_days = self.opt.output_len
        self.val_near_days = self.predict_days
        self.lens = self.train_days + self.predict_days+1
        self.batch_size = opt.batchsize
        self.thre1 = 0
        self.thre2 = 0
        self.os_h = opt.os_s
        self.os_l = opt.os_s
        self.iterval = opt.os_v

        self.is_prob_feature = 1
        self.val_data_loader = []
        self.train_data_loader = []
        self.month = []
        self.day = []
        self.hour = []

        self.h_value = []
        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, "train")
        self.sampled_h_value = []

        self.read_dataset()
        norm = []
        norm.append(self.get_mean())
        norm.append(self.get_std())
        norm.append(self.get_R_mean())
        norm.append(self.get_R_std())
        np.savetxt(self.expr_dir + "/" + "Norm.txt", norm)
        norm = np.loadtxt(self.expr_dir + "/" + "Norm.txt", dtype=float, delimiter=None)
        print("norm is: ", norm)
        if self.opt.mode == "train":
            self.val_dataloader()
            self.train_dataloader()
        else:
            self.refresh_dataset(trainX, R_X)
        self.roll = 8        
        
    def get_trainX(self):

        return self.trainX

    def get_data(self):

        return self.data

    def get_diff_data(self):

        return self.diff_data

    def get_sensor_data(self):

        return self.sensor_data

    def get_sensor_data_norm(self):

        return self.sensor_data_norm

    def get_sensor_data_norm1(self):

        return self.sensor_data_norm1

    def get_R_data(self):

        return self.R_data

    def get_R_sensor_data(self):

        return self.R_sensor_data

    def get_R_sensor_data_norm(self):

        return self.R_sensor_data_norm

    def get_R_sensor_data_norm1(self):

        return self.R_sensor_data_norm1

    def get_val_data_loader(self):

        return self.val_data_loader

    def get_train_data_loader(self):

        return self.train_data_loader

    def get_val_points(self):

        return self.val_points

    def get_test_points(self):

        return self.test_points

    def get_mean(self):

        return self.mean

    def get_mini(self):

        return self.mini

    def get_std(self):

        return self.std

    def get_R_mean(self):

        return self.R_mean

    def get_R_std(self):

        return self.R_std

    def get_month(self):

        return self.month

    def get_day(self):

        return self.day

    def get_hour(self):

        return self.hour

    def get_tag(self):

        return self.tag

    # Fetch dataset from data file, do preprocessing, generate a tag for the time series where 0 means None value, 1 means valid vauel
    def read_dataset(self):

        # read sensor data to vector
        start_num = self.trainX[
            self.trainX["datetime"] == self.opt.start_point
        ].index.values[0]
        self.start_num = start_num
        print("for sensor ", self.opt.stream_sensor, "start_num is: ", start_num)

        # foot label of train_end
        train_end = self.trainX[self.trainX["datetime"]==self.opt.train_point].index.values[0] - start_num 
        print("train set length is : ", train_end)

        # the whole dataset
        self.sensor_data = self.trainX[start_num: train_end + start_num]  # e.g. 2011/7/1  22:30:00 - 2020/6/22  23:30:00
        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.diff_data = diff_order_1(self.data)
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))
        self.sensor_data_norm, self.mean, self.std = log_std_normalization(self.data)

        self.sensor_data_norm1 = np.array(self.sensor_data_norm).reshape(-1, 1)
        if self.opt_hinter_dim >= 1:
            # read Rain data to vector
            R_start_num = self.R_X[
                self.R_X["datetime"] == self.opt.start_point
            ].index.values[0]
            print("for sensor ", self.opt.rain_sensor, "start_num is: ", R_start_num)
            R_train_end = (self.R_X[self.R_X["datetime"] == self.opt.train_point].index.values[0]- R_start_num)
            print("R_X set length is : ", R_train_end)
            self.R_sensor_data = self.R_X[R_start_num: R_train_end + R_start_num]
            self.R_data = np.array(self.R_sensor_data["value"].fillna(np.nan))
            self.R_data_time = np.array(self.R_sensor_data["datetime"].fillna(np.nan))
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(self.R_data)
            self.R_sensor_data_norm1 = np.array(self.R_sensor_data_norm).reshape(-1, 1)
            self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, self.R_sensor_data_norm1), 1)
            RR = self.R_sensor_data_norm
            RR = RR.tolist()
            for iR in range(int(self.ind_dim)):
                RR = [0] + RR[:-1]
                RRt = [[ff] for ff in RR]
                self.sensor_data_norm1 = np.concatenate(
                    (self.sensor_data_norm1, np.array(RRt)), 1
                )

        GMM_input = self.R_sensor_data_norm 
        if(self.is_prob_feature==1):
            clean_data = []
            for ii in range(len(GMM_input)):
                if (GMM_input[ii] is not None) and (np.isnan(GMM_input[ii]) != 1):
                    clean_data.append(GMM_input[ii])
            sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)   
            print("sensor_data_prob,", sensor_data_prob.shape)
            # dataset-wise gmm
            self.gm3.fit(sensor_data_prob)
            self.gm_means = np.squeeze(self.gm3.means_)
            self.z0 = np.min(self.gm_means)
            self.z1 = np.median(self.gm_means)
            self.z2 = np.max(self.gm_means)
            
            print("gm3.means are: ", self.gm_means)
            print("z : ", self.z0,self.z1,self.z2)
            print("gm3.covariances are: ", self.gm3.covariances_)
            print("gm3.weights are: ", self.gm3.weights_)
    
        self.tag = gen_month_tag(self.sensor_data)
        print("self.tag len, ", len(self.tag))
        
        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)        
        cos_d = cos_date(self.month, self.day, self.hour)
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.month, self.day, self.hour)
        sin_d = [[x] for x in sin_d]  

    # Randomly choose a point in timesequence,
    # if it is a valid start time (with no nan value in the whole sequence, between Sep and May), tag it as 3
    # For those points near this point (near is defined as a parameter), tag them as 4

    def val_dataloader(self):

        print("Begin to generate val_dataloader!")

        self.opt.name = "%s" % (self.opt.model)
        val_dir = os.path.join(self.opt.outf, self.opt.name, "val")
        near_len = self.predict_days  # self.val_near_days*24*4

        random.seed(self.opt.val_seed)
        if self.is_over_sampling == 1:
            print(
                "Over sampling on validation set is not supported by now, please wait..."
            )
        else:
            ii = 0
            while ii < self.opt.val_size:

                # use randomly chosen validation set
                i = random.randint(self.predict_days, len(self.data) - self.lens - 1)
                a1 = -9
                a2 = -6
                if (
                    (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any())
                    and (not np.isnan(self.R_data[i: i + self.lens]).any())
                    and (
                        self.tag[i + self.train_days] <= a1
                        or a2 < self.tag[i + self.train_days] < 0
                        or 2 <= self.tag[i + self.train_days] <= 3
                    )
                ):
                    point = self.data_time[i + self.train_days]

                    self.tag[i + self.train_days] = 2  # tag 2 means in validation set
                    for k in range(near_len):
                        self.tag[i + self.train_days - k] = (
                            3  # tag 3 means near points of validation set
                        )
                        self.tag[i + self.train_days + k] = 3
                    self.val_points.append([point])
                    ii = ii + 1

    # Can only be run after val_dataloader
    # Randomly choose a point in timesequence,
    # if it is a valid start time (with no nan value in the whole sequence between Sep and May, and tag is not 3 and 4),
    # select it as a train point, tag it as 5

    def train_dataloader(self):

        print("Begin to generate train_dataloader!")
        DATA = []
        Label = []
        a3 = 1

        # randomly choose train data
        random.seed(self.opt.train_seed)

        if self.is_over_sampling == 1:

            print(
                "Over sampling on validation set is not supported by now, please wait..."
            )

        else:
            ii = 0
            jj = 0
            while ii < self.opt.train_volume * (
                1 - self.oversampling / 100
            ) or jj < self.opt.train_volume * (self.oversampling / 100):
                i = random.randint(
                    self.predict_days,
                    len(self.sensor_data_norm) - 31 * self.predict_days - 1,
                )
                pre1 = np.array(self.data[(i+self.train_days):(i+self.train_days+self.predict_days)]) 
                pre2 = np.array(self.R_sensor_data_norm[(i+self.train_days-int(self.predict_days/2)):(i+self.train_days+int(self.predict_days/2))])
                a1 = -9
                a2 = -6
                if (np.max(pre2) > self.z2):
                    a3 = self.os_h
                    max_index = np.argmax(pre1)
                a5 = self.iterval
                if (jj < self.opt.train_volume*(self.oversampling/100)) and (np.max(pre2) > self.z2) and ( not np.isnan(self.sensor_data_norm1[i:i+self.lens]).any()) and (self.tag[i+self.train_days] <= a1 or a2 < self.tag[i+self.train_days] < 0):
                    if a3 > 0:
                        i = i + max_index -1
                        i = i - int(a3*a5/2)
                    for kk in range(a5): 
                        i = i + a3 
                        if i > len(self.data)-6*self.predict_days-1 or i < 6*self.predict_days :
                            continue
                        if ( not np.isnan(self.sensor_data_norm1[i:i+self.lens]).any() and self.tag[i+self.train_days] != 2 and self.tag[i+self.train_days] != 3 ): 
                
                            data0 = np.array(self.sensor_data_norm1[i:(i+self.train_days)]).reshape(self.train_days,-1)
                            label00 = np.array(self.sensor_data_norm[(i+self.train_days):(i+self.train_days+self.predict_days)]) 
                            label01 = label00
                            label0 =[[ff] for ff in label01]

                            b = i+self.train_days
                            e = i+self.train_days+self.predict_days

                            label2 = cos_date(self.month[b:e], self.day[b:e], self.hour[b:e]) # represent cos(int(data)) here
                            label2 = [[ff] for ff in label2]
                            
                            label3 = sin_date(self.month[b:e], self.day[b:e], self.hour[b:e]) # represent sin(int(data)) here
                            label3 = [[ff] for ff in label3]
                            
                            label = np.concatenate((label0,label2),1)
                            label = np.concatenate((label,label3),1)
                            
                            self.tag[i+self.train_days] = 4     
                            jj = jj + 1
                            DATA.append(data0)
                            Label.append(label)  
                    
                elif ii < self.opt.train_volume*(1 - self.oversampling/100) and ( not np.isnan(self.sensor_data_norm1[i:i+self.lens]).any()) and (self.tag[i+self.train_days] <= a1 or a2 < self.tag[i+self.train_days] < 0) : 

                    if (1==1):                     
                        data0 = np.array(self.sensor_data_norm1[i:(i+self.train_days)]).reshape(self.train_days,-1)
                        label00 = np.array(self.sensor_data_norm[(i+self.train_days):(i+self.train_days+self.predict_days)]) 
                        label0 =[[ff] for ff in label00]

                        b = i+self.train_days
                        e = i+self.train_days+self.predict_days

                        label2 = cos_date(self.month[b:e], self.day[b:e], self.hour[b:e]) # represent cos(int(data)) here
                        label2 = [[ff] for ff in label2]

                        label3 = sin_date(self.month[b:e], self.day[b:e], self.hour[b:e]) # represent sin(int(data)) here
                        label3 = [[ff] for ff in label3]

                        label = np.concatenate((label0,label2),1)
                        label = np.concatenate((label,label3),1)

                        DATA.append(data0)
                        Label.append(label)                         
                        self.tag[i+self.train_days] = 4
                        ii = ii + 1                                                                  
                
        dataset1=RnnDataset(DATA,Label)
        self.train_data_loader = DataLoader(dataset1, 
                             self.batch_size,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True,
                             collate_fn=lambda x: x)
    
    def refresh_dataset(self, trainX, R_X):
        self.trainX = trainX
        self.R_X = R_X
        # read sensor data to vector
        start_num = self.trainX[self.trainX["datetime"]==self.opt.start_point].index.values[0]
        print("for sensor ", self.opt.stream_sensor, "start_num is: ", start_num)
        idx_num = 0
        #foot label of train_end
        train_end = self.trainX[self.trainX["datetime"]==self.opt.train_point].index.values[0] - start_num 
        print("train set length is : ", train_end)
        
        #the whole dataset
        k = self.trainX[self.trainX["datetime"]==self.test_end_time].index.values[0]
        f = self.trainX[self.trainX["datetime"]==self.test_start_time].index.values[0]
        self.sensor_data = self.trainX[start_num:k] 
        self.data = np.array(self.sensor_data["value"].fillna(np.nan))  
        self.diff_data = diff_order_1(self.data)
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan)) 
        self.sensor_data_norm = log_std_normalization_1(self.data, self.mean, self.std)  # use old mean & std  
        self.sensor_data_norm1 = [[ff] for ff in self.sensor_data_norm] 
        
        if  (self.opt_hinter_dim >= 1):
            # read Rain data to vector      
            R_start_num = self.R_X[self.R_X["datetime"]==self.opt.start_point].index.values[0]
            print("for sensor ", self.opt.rain_sensor, "start_num is: ", R_start_num)
            R_idx_num = 0
            R_test_end = self.R_X[self.R_X["datetime"]==self.opt.test_end].index.values[0] - R_start_num 
            print("R_X set length is : ", R_test_end)        
            self.R_sensor_data = self.R_X[R_start_num:R_test_end+R_start_num] # e.g. 2011/7/1  22:30:00 - 2020/6/22  23:30:00 
            self.R_data = np.array(self.R_sensor_data["value"].fillna(np.nan))    
            self.R_data_time = np.array(self.R_sensor_data["datetime"].fillna(np.nan)) 
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(self.R_data)   
            self.R_sensor_data_norm1 = [[ff] for ff in  self.R_sensor_data_norm] 
            self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, self.R_sensor_data_norm1), 1)
            RR = self.R_sensor_data_norm
            RR = RR.tolist()
            for iR in range(self.ind_dim):
                RR = [0] + RR[:-1]
                RRt = [[ff] for ff in  RR] 
                self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, np.array(RRt)), 1)    

        self.tag = gen_month_tag(self.sensor_data) # update        
        self.month, self.day, self.hour = gen_time_feature(self.sensor_data) # update
        
        cos_d = cos_date(self.month, self.day, self.hour)
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.month, self.day, self.hour)
        sin_d = [[x] for x in sin_d]
        

    def gen_test_data(self):

        self.test_points = []
        self.refresh_dataset(self.trainX, self.R_X)
        print("Begin to generate test_points!")

        start_num = self.trainX[
            self.trainX["datetime"] == self.opt.start_point
        ].index.values[0]

        begin_num = (
            self.trainX[self.trainX["datetime"] == self.test_start_time].index.values[0]
            - start_num
        )
        end_num = (
            self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0]
            - start_num
        )

        for i in range(int((end_num - begin_num - self.predict_days) / 16)):
            point = self.data_time[begin_num + i * 16]
            if not np.isnan(
                np.array(
                    self.data[
                        begin_num
                        + i * 16
                        - self.train_days: begin_num
                        + i * 16
                        + self.predict_days
                    ]
                )
            ).any():
                self.test_points.append([point])