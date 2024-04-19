from data_provider.data_loader import Dataset_Custom, Dataset_Crime, Dataset_Wiki, Dataset_Crime_Related_re, Dataset_Wiki_Related_re, Dataset_Crime_Related_nore, Dataset_Wiki_Related_nore, Dataset_Custom_re, Dataset_Electricity, Dataset_Electricity_re
from torch.utils.data import DataLoader
from utils.tools import print_with_timestamp

data_dict = {
    'custom': Dataset_Custom,
    # for the self dataset
    'crime': Dataset_Crime,
    'crime-reindex': Dataset_Crime_Related_re,
    'crime-reindex-unrelated': Dataset_Crime_Related_nore,
    'wiki': Dataset_Wiki,
    'wiki-reindex': Dataset_Wiki_Related_re,
    'wiki-reindex-unrelated': Dataset_Wiki_Related_nore,
    'traffic': Dataset_Custom,
    'traffic-reindex': Dataset_Custom_re,
    'electricity': Dataset_Electricity,
    'electricity-reindex':Dataset_Electricity_re,
}


def data_provider(args, flag, logger=None):
    Data = data_dict[args.data]
    if args.task_name != 'statistic':
        timeenc = 0 if args.embed != 'timeF' else 1
    else:
        timeenc = 0
        
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
    else:
        shuffle_flag = True
        drop_last = True
    batch_size = args.batch_size 
    freq = args.freq
    if 'reindex' not in args.data:
        data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns)
        
    else:
        data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        data_topk_path=args.data_topk_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        timeenc=timeenc,
        freq=freq,
        k=args.k,
        seasonal_patterns=args.seasonal_patterns)
    print_with_timestamp(flag, len(data_set))
    print_with_timestamp(f'{flag}: {len(data_set)}')
    logger.info(f'{flag}: {len(data_set)}')
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
