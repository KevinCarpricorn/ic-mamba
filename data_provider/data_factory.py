from torch.utils.data import DataLoader
from data_provider.data_loader import (
    Dataset_hour,
    Dataset_minute,
    Dataset_Custom,
    Dataset_SS,
    din_loader,
    ss_loader,
)
from data_provider.cls import collate_fn


def data_provider(args, flag):
    """Select and return the appropriate dataset and dataloader.

    Parameters
    ----------
    args : argparse.Namespace
        Experiment configuration.
    flag : str
        Data split indicator (e.g., 'train', 'val', 'test', 'TRAIN', 'TEST').
    """
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False if flag in ['test', 'TEST'] else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'classification':
        data_dict = {
            'din': din_loader,
            'ss': ss_loader,
        }
        Data = data_dict[args.data]
        data_set = Data(args=args, root_path=args.root_path, flag=flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
        )
    else:
        data_dict = {
            'ETTh1': Dataset_hour,
            'ETTh2': Dataset_hour,
            'ETTm1': Dataset_minute,
            'ETTm2': Dataset_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
            'ss': Dataset_SS,
        }
        Data = data_dict[args.data]
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=getattr(args, 'data_path', None),
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=getattr(args, 'seasonal_patterns', None),
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

    return data_set, data_loader
