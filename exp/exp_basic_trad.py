import os
import numpy as np
import torch.nn as nn
import time
from models.Stat_models import Naive, Random, GBRT, Arima, SArima, Naive_repeat, Naive_seasonal, Constant, Autoarima
from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import visual, result_print

class Exp_Basic_Traditional(Exp_Basic):
    def __init__(self, args):
        super(Exp_Basic_Traditional, self).__init__(args)
        self.model = self._build_model()
        
    def _build_model(self):
        model_dict= {
            'Random': Random,
            'Naive': Naive,
            'Naive_repeat': Naive_repeat,
            'Naive_seasonal': Naive_seasonal,
            'Constant': Constant,
            'AutoARIMA': Autoarima,
            'Arima': Arima,
            'SArima': SArima,
            'GBRT': GBRT}
        model = model_dict[self.args.model](self.args)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.args.logger)
        return data_set, data_loader

    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse = False):
        outputs, reshaped_y = self.model(batch_x, batch_y)
        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            reshaped_y = dataset_object.inverse_transform(reshaped_y)
        return outputs, reshaped_y
        
    def vali(self, vali_data, vali_loader):
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(vali_data, batch_x, batch_y)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = self.criterion(pred, true)
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.output_path+'/checkpoints/', setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        criterion = self._select_criterion()
        
        iter_count = 0
        train_loss_arr, vali_loss_arr, test_loss_arr = [], [], []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            pred, true = self._process_one_batch(train_data, batch_x, batch_y)
            loss = criterion(pred, true)

            if (i + 1) % 100 == 0:
                self.args.logger.info("\titers: {0} | loss: {2:.7f}".format(i + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                self.args.logger.info('\tspeed: {:.4f}s/iter'.format(speed))
                iter_count = 0
                time_now = time.time()
        
            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)
 
            train_loss_arr.append(loss)
            vali_loss_arr.append(vali_loss)
            test_loss_arr.append(test_loss)     
            self.args.logger.info("Batch: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(i, loss, vali_loss, test_loss))      
        
        return self.model


    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')  
        preds = []
        trues = []
        trains = []  
        folder_path = self.args.output_path+'test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(test_data, batch_x, batch_y) 
            if i % 20 == 0:
                aaa= np.array(batch_x)
                gt = np.concatenate((aaa[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((aaa[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf')) 
            train = batch_x[:,:,0].detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)  
            trains.append(train)        
        # trains.append(train)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # process nan
        preds = np.nan_to_num(preds)
        trues = np.nan_to_num(trues)
        # mae, mse, rmse, mape, mspe, rse, corr = metric(np.array(preds), np.array(trues))
        
        folder_path = self.args.output_path + 'results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  
        mae, mse, rmse, mape, mspe, rse, corr, rmsse, wrmspe, smape, wape, mase = metric(preds, trues, trains, seasonality=self.args.seasonality)
        print(f'test_results----------')
        result_print('mse:{}, mae:{}, rse:{}, rmse:{}, mape:{}, mspe:{}, rmsse:{}, wrmspe:{}, smape:{}, wape:{}, mase:{}'.format(mse, mae, rse, rmse, mape, mspe, rmsse, wrmspe, smape, wape, mase))
        # print(('mse:{}, mae:{}, rse:{}, rmse:{}, mape:{}, mspe:{}, corr:{}'.format(mse, mae, rse, rmse, mape, mspe, corr)))
        # logger.info('mse:{}, mae:{}, rse:{}, rmse:{}, mape:{}, mspe:{}, corr:{}'.format(mse, mae, rse, rmse, mape, mspe, corr))
        # self.args.logger.info('mse:{}, mae:{}, rse:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rse, rmse, mape, mspe))
        # self.args.logger.info(result_print('mse:{}, mae:{}, rse:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rse, rmse, mape, mspe)))
        f = open(self.args.output_path + "result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, rse:{}, rmse:{}, mape:{}, mspe:{}, corr:{}'.format(mse, mae, rse, rmse, mape, mspe, corr))
        f.write('mse:{}, mae:{}, rse:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rse, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()    
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        return
    
    