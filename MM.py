import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time
import shutil
import os
import hyperopt
from datetime import datetime
from torch import optim
from sklearn import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose, STL, DecomposeResult
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, base,Trials
cuda = torch.device('cuda')
print(os.getcwd())
print('IMPORTS ARE DONE!!!')


class PARAMETERS():
    
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def DICT_TO_LIST(self):
        prev_out_ch = 0
        self.LIST = list()
        self.seq_len_left = self.DICT['OTHERS']['1']['windowlength']
        for tt, key in enumerate(list(self.DICT.keys())):
            if key == 'flatten':
                self.LIST.append([key, ['nothing','nothing']])
                prev_out_ch = prev_out_ch * self.seq_len_left
            elif key != 'OTHERS':
                for ttt, layer in enumerate(list(self.DICT[key].keys())):
                    act = False
                    p_list = list()
                    for tttt, param in enumerate(list(self.DICT[key][layer].keys())):
                        if param not in ['dropout','batchnorm','activation_function']:
                            if param == 'KER':
                                self.seq_len_left = self.seq_len_left - self.DICT[key][layer][param] + 1
                            if tt is 0 and ttt is 0:
                                if tttt is 0:
                                    p_list.append(self.featuresize)       
                                p_list.append(self.DICT[key][layer][param])
                            else:
                                if tttt is 0:
                                    p_list.append(prev_out_ch)       
                                p_list.append(self.DICT[key][layer][param])
                    self.LIST.append([key, p_list])
                    prev_out_ch = p_list[1]

                    if 'batchnorm' in list(self.DICT[key][layer].keys()):
                        if self.DICT[key][layer]['batchnorm'] is True:
                            self.LIST.append(['batchnorm',[prev_out_ch,True]])

                    if self.DICT[key][layer]['dropout'][0] is True:
                        self.LIST.append(['dropout', [False, False]])
                    if self.DICT[key][layer]['activation_function'][0] is True:
                        self.LIST.append([self.DICT[key][layer]['activation_function'][1],[False,False]])

           
    def GET_OTHERS(self,OTHERS=None):
        if OTHERS is None:
            OTHERS  =  {
                            'windowlength': 24,
                            'out_size': 4,
                            'period': 24,
                            'lrate': 0.003,
                            'batchsize': 32,
                            'epoch': 550
                            }
        return OTHERS
        
    def GET_DICT(self,DICT=None):
        #USES: GET_OTHERS


        #CREATES: self.DICT

        try:
            _ = list(DICT.keys())
        except:
    
    
    
            OTHERS = self.GET_OTHERS()
            self.DICT = {'CONV': {
                                    '1': {'FIL': 64, 
                                          'KER': 8,
                                          'dropout': [True, 0.5],
                                          'batchnorm': False,
                                          'activation_function': [True, 'relu']
                                        },
                                    '2': {'FIL': 32, 
                                          'KER': 8,
                                          'dropout': [True, 0.5],
                                          'batchnorm': False,
                                          'activation_function': [True, 'relu']
                                        }
                                  },
            
                      'flatten': {'1': {'nofilter':0 , 'nonothing':0 }},
            
                      'DENSE': {
                      
                                '1': {'FIL': 64,
                                      'dropout' : [True,0.6],
                                      'activation_function': [True, 'relu']
                                    },

                                '2': {'FIL':OTHERS['out_size'],
                                      'dropout' : [False,0],
                                      'activation_function': [False, '-']
                                      }
                              },
                        
                      'OTHERS': {'1':OTHERS}
            }
            
            
    def CREATE_SEARCH_SPACE(self,TO_CHNG= None):
        #USES: GET_PARAMS_TO_CHANGE()

        #CREATES: self.space

        space = {}
        self.PARAMS_TO_CHANGE = self.GET_PARAMS_TO_CHANGE()
        HYP_DICT ={}
        d_count = 0
        for TYPE in  list(self.PARAMS_TO_CHANGE.keys()):
            HYP_DICT_LAYER = {}
            for layernum in list(self.PARAMS_TO_CHANGE[TYPE].keys()):
                HYP_DICT_PARAMS = {}
                for PARAM in list(self.PARAMS_TO_CHANGE[TYPE][layernum].keys()):
                    if PARAM == 'dropout':
                        d_count = d_count + 1
                        HYP_DICT_PARAMS[PARAM + str(d_count)] = hp.uniform(PARAM + str(d_count),self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][0],self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][1])
                    else:
                        HYP_DICT_PARAMS[PARAM + layernum] = hp.uniform(PARAM + layernum,self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][0],self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][1])
                    HYP_DICT_LAYER[layernum] =  HYP_DICT_PARAMS
                HYP_DICT[TYPE] =  HYP_DICT_LAYER

        self.space = hp.choice('paramz', [HYP_DICT])

    def GET_PARAMS_TO_CHANGE(self,PARAMS_TO_CHANGE=None):
        if PARAMS_TO_CHANGE is None:
              
            PARAMS_TO_CHANGE = {'CONV': {
                                          '1': {
                                                'KER': (2,14),
                                                'dropout': (0.2, 0.8)
                                              },
                                          '2': {
                                                'KER': (2,11),
                                                'dropout': (0.2, 0.8)
                                              }
                                        },
                                
                              'DENSE': {

                                          '1': {'FIL': (32,128),
                                                'dropout' : (0.2,0.8)
                                                }
                                          },
                              'OTHERS':{'1':{'lrate':(0.01,0.0001)}}
                              }
        return PARAMS_TO_CHANGE

    #CREATE SUBDIR OF ABOVE NAMED WITH 
    #EXPERIMENT DATE AND START TIME
    def CREATE_DIR(self):
        #INPUT: self.DICT

        first_Con = True
        SAVE_DIR = ''
        repopath = os.getcwd()
        try:
            os.mkdir(repopath + '/storage')
        except:
            pass
        for KEY in list(self.DICT.keys()):
            if KEY != 'OTHERS':
                SAVE_DIR = SAVE_DIR + '_' +KEY + '-'
                for LAYER in list(self.DICT[KEY].keys()):
                    SAVE_DIR = SAVE_DIR + LAYER + '-'
        SAVE_DIR = repopath + '/storage/' + SAVE_DIR
        try: 
            os.mkdir(SAVE_DIR)
            self.SAVE_DIR = SAVE_DIR
        except:
            pass

        SAVE_DIR = SAVE_DIR + '/' +str(datetime.now())[:-10]
        self.SAVE_DIR = SAVE_DIR

        SAVE_DIR_PLOTS = SAVE_DIR + '/' + 'PLOTS'
        self.SAVE_DIR_PLOTS = SAVE_DIR_PLOTS

        SAVE_DIR_MODELS = SAVE_DIR + '/' + 'MODELS'
        self.SAVE_DIR_MODELS = SAVE_DIR_MODELS

        try:
            os.mkdir(SAVE_DIR)
        except:
            pass

        try:
            os.mkdir(SAVE_DIR_PLOTS)
            os.mkdir(SAVE_DIR_MODELS)
        except:
            pass

    #CREATES SAVE NAME FOR BOTH PLOTS
    #AND THE KEY FOR HIST PLOT
    #key_VAR = CHANGING VARS WITH VALUES
    def CREATE_SAVE_NAME(self,DDD):
        save_DIR = ''
        plot_header = ''
        for keynum, KEY in enumerate(list(DDD.keys())):

            save_DIR = save_DIR + '_' + KEY  
            plot_header = plot_header + '\n' + KEY + '--- ' 
            for LAYER in list(DDD[KEY].keys()):
                save_DIR = save_DIR + '-' + LAYER  
                plot_header = plot_header + LAYER + ': ' 
                for PARAM in list(DDD[KEY][LAYER].keys()):
                    save_DIR = save_DIR + PARAM + '-' + str(DDD[KEY][LAYER][PARAM])[:5] + '---'
                    plot_header = plot_header + PARAM + '=' + str(DDD[KEY][LAYER][PARAM])[:5] + '   '
            plot_header = plot_header + '\n'
        return save_DIR, plot_header
    
    #SAVE CONSTANT HYPERPARAMETERS OF EXPERIMENT AS TXT
    def WRITE_CONSTANTS(self):
        key_CONST = ''
        key_VAR = ''
        for KEY in list(self.DICT.keys()):
            exist = True
            if KEY is not 'flatten':
                if KEY not in list(self.PARAMS_TO_CHANGE.keys()):
                    exist = False
                    key_CONST = key_CONST + '\n\n TYPE:   {} \n \n'.format(KEY)
                    key_CONST =  key_CONST + 'LAYER:'
                    for LAYER in list(self.DICT[KEY].keys()):
                        key_CONST =  key_CONST + '\n{} \t---\t'.format(LAYER)
                        for PARAM in list(self.DICT[KEY][LAYER].keys()):
                            key_CONST =  key_CONST + '{}: {} \t\t'.format(PARAM,self.DICT[KEY][LAYER][PARAM])
                        key_CONST = key_CONST + '\n'
                else:
                    key_VAR = key_VAR + '\n\n TYPE:   {}\n\n'.format(KEY)
                    key_VAR = key_VAR + 'LAYER:'

                    key_CONST = key_CONST + '\n\n TYPE:   {} \n \n'.format(KEY)
                    key_CONST =  key_CONST + 'LAYER:'

                    for LAYER in list(self.DICT[KEY].keys()):
                        key_CONST =  key_CONST + '\n{} \t---\t'.format(LAYER)
                        if LAYER in list(self.PARAMS_TO_CHANGE[KEY].keys()):
                            key_VAR =  key_VAR + '\n{} \t---\t'.format(LAYER)
                            for PARAM in list(self.DICT[KEY][LAYER].keys()):
                                if PARAM in list(self.PARAMS_TO_CHANGE[KEY][LAYER].keys()):
                                    key_VAR =  key_VAR + '{}: {} \t\t'.format(PARAM,self.PARAMS_TO_CHANGE[KEY][LAYER][PARAM])
                                else:
                                    key_CONST =  key_CONST + '{}: {} \t\t'.format(PARAM,self.DICT[KEY][LAYER][PARAM])

                        else:
                            for PARAM in list(self.DICT[KEY][LAYER].keys()):
                                key_CONST =  key_CONST + '{}: {} \t\t'.format(PARAM,self.DICT[KEY][LAYER][PARAM])

                     
        with open(self.SAVE_DIR + '/CONSTANT_HYPERPARAMETERS.txt' , 'w') as file: 
            file.write(key_CONST)
        with open(self.SAVE_DIR + '/PARAMETERS_TO_TUNE.txt', 'w') as file:
            file.write(key_VAR) 
 

    #SAVES HIST AND PRED PLOTS
    def SAVE_PLOTS(self,DDD):
        save_NAME, plot_header = self.CREATE_SAVE_NAME(DDD)
        fig = plt.figure(figsize=(12,6))
        fig.suptitle(plot_header)
        plt.plot(self.hist)
        plt.plot(self.hist_valid)
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.2),'--r')
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.3),'--b')
        plt.ylim((0.1,0.5))
        plt.savefig( self.SAVE_DIR + '/PLOTS/' + save_NAME + '.png')
        
        fig2 = self.plotz()
        fig2.suptitle(plot_header)
        plt.savefig(self.SAVE_DIR + '/PLOTS/PRD_' + save_NAME + '.png')
        plt.close('all')

    def CONV_DICT_TO_INT(self,DDD):
        for KEY in list(DDD.keys()):
            for LAYER in list(DDD[KEY].keys()):
                for PARAM in list(DDD[KEY][LAYER].keys()):    
                    if LAYER != 'flatten':
                        if PARAM[:7] != 'dropout':
                            if PARAM[:9] != 'batchnorm':
                                if PARAM[:5] != 'lrate':
                                    DDD[KEY][LAYER][PARAM] = np.int(np.round(DDD[KEY][LAYER][PARAM]))
                                    print(PARAM)
        return DDD


    def preprocess(self,split):
        global SCALERR
        data = pd.read_excel('clean.xlsx').dropna()
        print('welldone')
        windowlength = self.DICT['OTHERS']['1']['windowlength']
        period = self.DICT['OTHERS']['1']['period']
        outsize = self.DICT['OTHERS']['1']['out_size']
        arr = np.asarray(data['sales'])
        vv =pd.read_csv('vix.csv',sep=',')

        vix_inv = np.array(vv['Price'])
        vix = np.zeros(len(vix_inv))
        for i in range(len(vix)):
            vix[len(vix_inv) - i-1] = vix_inv[i]

        dol =pd.read_csv('dollar.csv',sep=',')
        dollar_inv = np.array(dol['Price'])

        dollars = np.zeros(len(dollar_inv)-1)
        for i in range(1,len(dollars)):
            dollars[len(dollar_inv) - i-1] = dollar_inv[i]
            
        res = STL(arr,period = period ,seasonal = 23 , trend = 25).fit()

        dataz = np.swapaxes(np.array([res.seasonal,res.trend,res.resid,vix,dollars]),0,1)
        train = dataz[:split]
        test = dataz[split:]
        TR_OUT_VAL = arr[:split]
        TST_OUT_VAL = arr[split:]                 
        MAX_window = windowlength
        scaler = StandardScaler()
        sclr = scaler.fit(train)
        train =  scaler.transform(train)
        test =  scaler.transform(test)

        del scaler

        SCALERR = StandardScaler()
        SCALERR = SCALERR.fit(TR_OUT_VAL.reshape(-1,1))
        TR_OUT_NORM =  SCALERR.transform(TR_OUT_VAL.reshape(-1,1)).reshape(split)
        TST_OUT_NORM =  SCALERR.transform(TST_OUT_VAL.reshape(-1,1)).reshape(len(arr)-split)

        TR_OUT = np.asarray([[TR_OUT_NORM[i+k+windowlength] for i in range(outsize)] for k in range(split - outsize - MAX_window)])
        TST_OUT = np.asarray([[TST_OUT_NORM[i+k+windowlength] for i in range(outsize)] for k in range(len(arr) - split - outsize - windowlength)])

        for feat in range(train.shape[1]):
            if feat == 0:
                TR_INP = np.array([[[ np.array(train[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(split - outsize - MAX_window)])
            else:
                TR_new = np.array([[[ np.array(train[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(split - outsize - MAX_window)])
                TR_INP = np.concatenate((TR_INP,TR_new),axis=1)

        for feat in range(test.shape[1]):
            if feat == 0:
                TST_INP = np.array([[[ np.array(test[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(len(arr) - split - outsize - MAX_window)])
            else:
                TST_new = np.array([[[ np.array(test[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(len(arr) - split - outsize - MAX_window)])
                TST_INP = np.concatenate((TST_INP,TST_new),axis=1)

        TR_INP = torch.Tensor(TR_INP).to(device = cuda)
        TST_INP = torch.Tensor(TST_INP).to(device = cuda)
        TR_OUT = torch.Tensor(TR_OUT).to(device = cuda)
        TST_OUT = torch.Tensor(TST_OUT).to(device = cuda)


        TRA_DSet = TensorDataset(TR_INP, TR_OUT)
        VAL_DSet = TensorDataset(TST_INP, TST_OUT)
        self.featuresize = TR_INP.shape[1]
        self.TST_INP = TST_INP
        self.TST_OUT = TST_OUT
        self.TR_INP = TR_INP
        self.TR_OUT = TR_OUT

        self.train_DL = DataLoader(TRA_DSet, batch_size=self.DICT['OTHERS']['1']['batchsize'])
        self.val_DL = DataLoader(VAL_DSet, batch_size=self.DICT['OTHERS']['1']['batchsize']*2)

        
    def GET_MODEL(self,DD):
        print(DD)
        call_data_again = False
        DD = self.CONV_DICT_TO_INT(DD)    
        print(DD)
        for KEY in  list(DD.keys()):
            if KEY is 'OTHERS':
                key__ = list(DD['OTHERS']['1'].keys())
                for key_ in key__:
                    if key_ in ['windowlength','Period','batchsize','outsize']:
                        call_data_again = True
              
            for layernum in list(DD[KEY].keys()):
                for PARAM in list(DD[KEY][layernum].keys()):
                    PARAM_ = PARAM[:-1]
                    if PARAM_ == 'dropout' :
                        self.DICT[KEY][layernum][PARAM_] = [True,DD[KEY][layernum][PARAM]]
                    else:
                        self.DICT[KEY][layernum][PARAM_] = DD[KEY][layernum][PARAM]
        if call_data_again:
            self.Preprocess(split=220)
        self.DICT_TO_LIST()
        SAVE_NAME, __ = P_OBJ.CREATE_SAVE_NAME(DD)
        self.TRIAL_DIR = self.SAVE_DIR + '/MODELS/' + SAVE_NAME
        os.mkdir(self.TRIAL_DIR)

        model = Model(self.DICT,self.LIST, self.DICT['OTHERS']['1'], self.scaler, self.train_DL, self.val_DL,self.SAVE_DIR,self.EXPERIMENT_NUMBER,self.TRIAL_DIR)
        model.to(device = cuda)
        model.optimizer = optim.Adam(model.parameters(),lr=self.DICT['OTHERS']['1']['lrate'])
        
        minloss = model.fit()
        print(dir(model))
        model.SAVE_PLOTS(DD)
        torch.cuda.empty_cache()
        P_OBJ.EXPERIMENT_NUMBER = P_OBJ.EXPERIMENT_NUMBER + 1
        return {
            'loss': minloss,
            'status': STATUS_OK,
            'attachments':
                {'time_module': pickle.dumps(time.time)}
              }
  

    def plotz(self):

        #timereal = np.array([i for i in range(50)])
        timez = np.zeros((350,self.DICT['OTHERS']['1']['out_size']))
        for i in range(350):
            for j in range(self.DICT['OTHERS']['1']['out_size']):
                timez[i,j] = i + j
        inp,out = [self.TST_INP,self.TST_OUT]
        
        self.eval()
        with torch.no_grad():
           pred = self(inp)
        pred1 = pred.cpu()
        out1 = out.cpu()
        pred1 = SCALERR.inverse_transform(pred1)
        out1 = SCALERR.inverse_transform(out1)
        fig = plt.figure(figsize=(12, 6))
        ax1, ax2,  = fig.subplots(2, 1, )
        bisi = out.shape[0]
        for i in range(int(bisi/2)):
            ax1.plot(timez[i],pred1[i])
            ax1.plot(timez[i],out1[i],color='black')

        for i in range(int(bisi/2)+1,2*int(bisi/2)):

            ax2.plot(timez[i],pred1[i])
            ax2.plot(timez[i],out1[i],color='black')

        return fig

print('PARAMETERS DEFINED !!!!')


class Model(nn.Module, PARAMETERS):
    def __init__(self, DICT,LIST, OTHERS, SCLR, TRAIN, VAL,SAVE_DIR, EXPERIMENT_NUMBER,TRIAL_DIR):
        super().__init__()
        self.SAVE_DIR = SAVE_DIR
        self.TRIAL_DIR = TRIAL_DIR
        self.scaler = StandardScaler()
        self.LIST = LIST
        self.DICT = DICT
        self.OTHERS = OTHERS
        self.epoch = self.OTHERS['epoch']
        self.preprocess(split=216)
        self.layers = nn.ModuleList()
        self.Loss_FUNC = F.mse_loss
        self.EXPERIMENT_NUMBER = EXPERIMENT_NUMBER

        for elem in self.LIST:
            key = elem[0]
            args = elem[1]
            self.layer_add(key,*args)
            
        
    def layer_add(self,key,*args):
        self.layers.append(self.layer_set(key,*args))

        
    def layer_set(self,key,*args):
        ## push args into key layer type, return it
        ## push args into key layer type, return it
        if key == 'CONV':
            return nn.Conv1d(*args)
        elif key == 'LSTM':
            return nn.LSTM(*args)
        elif key == 'DENSE':
            return nn.Linear(*args)
        elif key == 'dropout':
            return nn.Dropout(*args)
        elif key == 'batchnorm':
            return nn.BatchNorm1d(*args)
        elif key == 'flatten':
            return nn.Flatten()
        elif key == 'relu':
            return nn.ReLU()
        elif key == 'OTHERS':
            pass

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
    def loss_batch(self, TR_INP, TR_OUT, opt=None):
        loss = self.Loss_FUNC(self(TR_INP), TR_OUT)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(TR_INP)

    def fit(self):
        self.hist = list()
        self.hist_valid = list()
        min_loss = 10
        BEST_LOSS = 5
        BEST_CHECKPOINT_LOSS = 5
        for epoch in range(self.epoch):
            self.train()
            batch_loss = 0
            count = 0
            for TR_INP, TR_OUT in self.train_DL:
                losses, nums = self.loss_batch(TR_INP, TR_OUT,opt = self.optimizer)
                batch_loss = batch_loss + losses
                count = count + 1
            train_loss = batch_loss / count
            self.hist.append(train_loss)

            self.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[self.loss_batch(TR_INP, TR_OUT) for TR_INP, TR_OUT in self.val_DL]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            print('Train Loss:  {}  and Validation Loss:  {}'.format(train_loss, val_loss))

            self.hist_valid.append(val_loss)
            if epoch % 20 == 0 and epoch != 0:
                is_best = val_loss < BEST_CHECKPOINT_LOSS
                BEST_CHECKPOINT_LOSS = min(val_loss,BEST_CHECKPOINT_LOSS)
                self.save_checkpoint({
                                  'epoch': epoch + 1,
                                  'state_dict': self.state_dict(),
                                  'BEST_CHECKPOINT_LOSS': BEST_CHECKPOINT_LOSS,
                                  'optimizer' : self.optimizer.state_dict(),
                                }, is_best,epoch)

            is_best = val_loss < BEST_LOSS
            BEST_LOSS = min(val_loss,BEST_LOSS)
            #print('{}:   {}'.format(epoch,val_loss))

        return BEST_LOSS
      
    def save_checkpoint(self, state, is_best, epoch):
        dirr = self.TRIAL_DIR + '/' + str(epoch) + 'pth.tar'
        best_dirr = self.TRIAL_DIR + '/BEST_IS_' + str(epoch) + 'pth.tar'

        torch.save(state, dirr)
        if is_best:
            torch.save(state, best_dirr)



def SET_EXPERIMENT(PARAMS_TO_CHANGE=None):
    print('hi')
    global P_OBJ
    P_OBJ = PARAMETERS()

    print('PARAM DEFINED')
    P_OBJ.EXPERIMENT_NUMBER = 1
    P_OBJ.GET_DICT()
    print('DICT DEFINED')
    P_OBJ.GET_PARAMS_TO_CHANGE()
    print('PARAM_TO_CHANE DEFINED')
    P_OBJ.CREATE_SEARCH_SPACE()
    print('SPACE DEFINED')
    P_OBJ.CREATE_DIR()
    print('DIR CREATED')
    P_OBJ.WRITE_CONSTANTS()
    print('CONST_WRITTEN')

    P_OBJ.preprocess(split=220)
    print('PREPROCESS run')
    
    for _ in range(500):
        DDICT = hyperopt.pyll.stochastic.sample(P_OBJ.space)
        print(DDICT)
        P_OBJ.GET_MODEL(DDICT)
SET_EXPERIMENT()
