import argparse
from exp.exp_basic_trad import Exp_Basic_Traditional
import random
import numpy as np
from utils.log import Logger
import datetime


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Experiments')

    # basic config
    parser.add_argument('--task_name', type=str, default='statistic',
                        help='task name, options:[statistic, long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Naive',
                        help='model name, options: [Random, Arima]')

    # data loader
    parser.add_argument('--data', type=str, default='self', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/xinz/ar57_scratch/xinz/HTSFB_project/HTSFB_datasets/self/Wiki/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='train_1_people.csv', help='data file')
    parser.add_argument('--output_path', type=str, default='/home/xz/big/code/HTSFB_project/HTSFB_output/', help='output path')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='/home/xz/big/code/HTSFB_project/HTSFB_output/checkpoints/', help='location of model checkpoints')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=42, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=7, help='start token length')
    parser.add_argument('--pred_len', type=int, default=28, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Daily', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    
    #
    parser.add_argument('--distribution', type=str, default='Gaussian', help='[Uniform, Normal, Gaussian]')
    parser.add_argument('--season', type=int, default=14, help='the seasonality of the data')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
  

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.current_time = current_time
    args.logger = Logger(args.output_path+'logs/', '{}-{}-{}-in_len:{}-out_len:{}-time:{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.current_time))
    #
    # print('Args in experiment:')
    # print(args)
    args.logger.info('Args in experiment:')
    args.logger.info(str(args))
    
    Exp = Exp_Basic_Traditional

    if args.is_training:
        # setting record of experiments
        setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}'.format(
            args.task_name,
            args.model,
            args.model_id,
            args.distribution,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.current_time)

        exp = Exp(args)  # set experiments
        # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        args.logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        args.logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
    else:
        setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}'.format(
            args.task_name,
            args.model,
            args.model_id,
            args.distribution,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.current_time)

        exp = Exp(args)  # set experiments
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        args.logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
