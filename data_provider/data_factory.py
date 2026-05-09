from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom
}


def data_provider(args, flag, contrastive=False):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    train_ratio = args.train_ratio
    test_ratio = args.test_ratio

    if args.task_name == 'anomaly_detection':
        drop_last = True if flag == "train" or flag == "val" else False
        # data_set = Data(
        #     args = args,
        #     root_path=args.root_path,
        #     # win_size=args.seq_len,
        #     flag=flag,
        #     train_ratio = train_ratio,
        #     test_ratio = test_ratio
        # )
        data_set = Data(
            args = args,
            root_path=args.root_path,
            task_name=args.task_name,
            data_path=args.data_path,
            flag=flag,
            size=args.win_size,
            step=args.step,
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            train_ratio = train_ratio,
            test_ratio = test_ratio,
            contrastive = contrastive
        )
        print(flag, len(data_set))
        if contrastive:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn = contrastive_collate_fn
            )    
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
            train_ratio = train_ratio,
            test_ratio = test_ratio
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            task_name=args.task_name,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            train_ratio = train_ratio,
            test_ratio = test_ratio
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
def contrastive_collate_fn(batch):
    anchors = []
    positives = []
    negatives = []
    labels = []

    for (x_tuple, _, _, _) in batch:
        anchor, pos, neg, label = x_tuple
        anchors.append(torch.Tensor(anchor))
        positives.append(torch.Tensor(pos))
        negatives.append(torch.Tensor(neg))
        labels.append(torch.Tensor(label))

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    # concat thành 1 batch lớn
    all_samples = torch.cat([anchors, positives, negatives], dim=0)
    labels = torch.stack(labels)

    # index mapping
    batch_size = anchors.shape[0]
    
    idx = torch.arange(batch_size)
    pos_idx = idx + batch_size
    neg_idx = idx + 2 * batch_size

    return all_samples, idx, pos_idx, neg_idx, labels
