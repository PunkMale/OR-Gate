import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

sys.path.append("..")

from trainer.dataset import Train_Dataset, Test_Dataset, Dev_Dataset
from parser import args


def train_dataloader(dataset, args):
    frames_len = np.random.randint(args.min_frames, args.max_frames)
    print("\nChunk size is: ", frames_len)
    print("Augment Mode: ", args.augment)
    print("Learning rate is: ", args.learning_rate)
    train_dataset = dataset
    # train_sampler = Train_Sampler(train_dataset, args.nPerSpeaker,
    #                               args.max_seg_per_spk, args.batch_size, filter=args.filter)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # sampler=train_sampler,
        # 当计算机的内存充足的时候，可以设置pin_memory=True
        pin_memory=True,
        # 丢弃掉最后一组数量不足的batch
        drop_last=True,
    )
    return loader


def test_dataloader(args, trials):
    enroll_list = np.unique(trials.T[1])
    test_list = np.unique(trials.T[2])
    eval_list = np.unique(np.append(enroll_list, test_list))
    print("number of eval: ", len(eval_list))
    print("number of enroll: ", len(enroll_list))
    print("number of test: ", len(test_list))

    test_dataset = Test_Dataset(data_list=eval_list, eval_frames=args.eval_frames, num_eval=0)
    loader = torch.utils.data.DataLoader(test_dataset, num_workers=args.num_workers, batch_size=1)
    return loader


if __name__ == "__main__":
    train_loader = train_dataloader(args)
    i = 0
    for batch in tqdm(train_loader):
        i = i + 1
    print(i)
    trials = np.loadtxt(args.trials_path, dtype=str)
    test_loader = test_dataloader(args, trials)
    i = 0
    for batch in tqdm(test_loader):
        i = i + 1
    print(i)
