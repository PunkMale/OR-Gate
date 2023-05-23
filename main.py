import os
import warnings
import datetime
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from parser import args
from trainer.nnet import TDNN_Encoder, ResNet34_Encoder, ECAPA_TDNN
from trainer.loss import Softmax, AMSoftmax, AAMSoftmax
from trainer.dataloader import train_dataloader, test_dataloader
from trainer.dataset import Train_Dataset, divide_dataset
from trainer.metric import cosine_score
from utils import PreEmphasis
import os
 

print('start time:{}'.format(datetime.datetime.now()))

# load trials and data list
trials_list = []
df = None
speaker = None
if os.path.exists(args.train_list_path):
    df = pd.read_csv(args.train_list_path)
    speaker = np.unique(df["utt_spk_int_labels"].values)
    args.num_classes = len(speaker)

save_model_dir = 'v1_r={}_w={}_k={}'.format(args.noisy_rate, args.warm_up, args.topk)
save_model_path = os.path.join("exp", save_model_dir)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Acoustic Feature
        self.mel_trans = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=512, win_length=400,
                                                 hop_length=160, window_fn=torch.hamming_window, n_mels=args.n_mels))
        self.instancenorm = nn.InstanceNorm1d(args.n_mels)

        # 2. Speaker_Encoder
        if args.nnet_type == 'tdnn':
            self.speaker_encoder = TDNN_Encoder(embedding_dim=args.embedding_dim, pooling_type=args.pooling_type,
                                                n_mels=args.n_mels)
        elif args.nnet_type == 'resnet34':
            self.speaker_encoder = ResNet34_Encoder(embedding_dim=args.embedding_dim, pooling_type=args.pooling_type,
                                                    n_mels=args.n_mels)
        elif args.nnet_type == 'ecapa_tdnn':
            self.speaker_encoder = ECAPA_TDNN(embedding_dim=args.embedding_dim, pooling_type=args.pooling_type,
                                              n_mels=args.n_mels)
        else:
            print("invalid net type:{}".format(args.nnet_type))

        # 3. Loss / Classifier
        if not args.evaluate:
            if args.loss_type == 'softmax':
                self.loss = Softmax(embedding_dim=args.embedding_dim, num_classes=args.num_classes)
            elif args.loss_type == 'amsoftmax':
                self.loss = AMSoftmax(embedding_dim=args.embedding_dim, num_classes=args.num_classes,
                                      margin=args.margin, scale=args.scale)
            elif args.loss_type == 'aamsoftmax':
                self.loss = AAMSoftmax(embedding_dim=args.embedding_dim, num_classes=args.num_classes,
                                       margin=args.margin, scale=args.scale)
            else:
                print("invalid loss type:{}".format(args.loss_type))

    def forward(self, x, label):
        x = self.extract_speaker_embedding(x)
        x = x.reshape(-1, args.nPerSpeaker, args.embedding_dim)
        loss, acc, output = self.loss(x, label)
        return loss.mean(), acc, output

    def extract_speaker_embedding(self, data):
        x = data.reshape(-1, data.size()[-1])
        x = self.mel_trans(x) + 1e-6
        x = x.log()
        x = self.instancenorm(x)
        x = self.speaker_encoder(x)
        return x


# 忽略警告
warnings.simplefilter("ignore")

# pre-define
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
train_dataset = Train_Dataset(args.train_list_path, args.augment,
                              musan_list_path=args.musan_list_path,
                              rirs_list_path=args.rirs_list_path,
                              max_frames=args.min_frames)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
)
model = Model().to(args.device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
print("\ntrain_list_path:{}".format(args.train_list_path))
print("Number of Training Speaker classes is: {}".format(args.num_classes))


def train(trainloader, temp_dataset, bp=True):
    # train
    model.train()
    progress = tqdm(trainloader)
    total = 0
    correct = 0
    total_loss = 0.0
    for idx, batch in enumerate(progress):
        progress.set_description("train")
        data, label, pseudo_label, data_path, noisy_tag = batch
        if idx >= len(trainloader.dataset)//trainloader.batch_size:
            labels = label.to('cpu').detach()
            temp_dataset.add_data(data_path, labels, pseudo_label, noisy_tag)
            continue
        data, label = data.to(args.device), label.to(args.device)
        loss, acc, output = model(data, label)

        total_loss + loss.item()
        _, predicted = output.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        labels = label.to('cpu').detach()
        output = output.to('cpu').detach()
        for idx in range(len(output)):
            topk = pd.Series(output[idx]).sort_values(ascending=False).index[:args.topk]
            if label[idx] in topk:
                pseudo_label[idx] += 1
        temp_dataset.add_data(data_path, labels, pseudo_label, noisy_tag)

        if bp:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress.update()
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100. * correct / total:.4f}",
        )
    progress.close()

    return temp_dataset


def dev_evaluate(model,trails_path):
    model.eval()
    trials = np.loadtxt(trails_path, dtype=str)
    dev_loader = test_dataloader(args, trials)

    scores_path = "temp.foo"

    index_mapping = {}
    eval_vectors = [[] for _ in range(len(dev_loader))]
    print("start eval...")
    with torch.no_grad():
        for idx, (data, label) in enumerate(tqdm(dev_loader)):
            data = data.permute(1, 0, 2).to(args.device)
            label = list(label)[0]
            index_mapping[label] = idx
            # embedding = model.module.extract_speaker_embedding(data)
            embedding = model.extract_speaker_embedding(data)
            embedding = torch.mean(embedding, axis=0)
            embedding = embedding.cpu().detach().numpy()
            # print("{}:{}".format(idx, embedding))
            eval_vectors[idx] = embedding
    eval_vectors = np.array(eval_vectors)
    print("start cosine scoring...")
    eer, th, mindcf_e, mindcf_h = cosine_score(trials, scores_path, index_mapping, eval_vectors)
    print("Cosine EER: {:.3f}%  minDCF(0.01): {:.5f}  minDCF(0.001): {:.5f}".format(eer * 100, mindcf_e, mindcf_h))
    return eer


def print_setting():
    print('device:{}'.format(args.device))
    print('warm up:{}'.format(args.warm_up))
    print('top_k:{}'.format(args.topk))
    print('max epoch:{}'.format(args.max_epochs))
    print('model:{}'.format(args.nnet_type))
    print('loss:{}'.format(args.loss_type))
    print('pooling:{}'.format(args.pooling_type))


if __name__ == "__main__":
    print_setting()
    best_eer = dev_evaluate(model, args.val_list_path)
    train_start_time = datetime.datetime.now()
    for epoch in range(args.max_epochs):
        epoch_start_time = datetime.datetime.now()
        print('\nEpoch {}:'.format(epoch))
        temp_dataset = Train_Dataset()
        if epoch < args.warm_up:
            trainset = train(train_loader, temp_dataset)
            train_loader = DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            labeled_dataset, unlabeled_dataset = divide_dataset(trainset)
            print('\nlabeled_clean:{}#labeled_noisy:{}#unlabeled_clean:{}#unlabeled_noisy:{}#\n'.format(
                torch.eq(labeled_dataset.noisy_tag, 0).sum(),
                torch.eq(labeled_dataset.noisy_tag, 1).sum(),
                torch.eq(unlabeled_dataset.noisy_tag, 0).sum(),
                torch.eq(unlabeled_dataset.noisy_tag, 1).sum()
            ))
            labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=False)
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True, drop_last=False)
            temp_dataset = train(labeled_loader, temp_dataset)
            trainset = train(unlabeled_loader, temp_dataset, bp=False)
        # counts
        counts = np.zeros(args.max_epochs, dtype=np.uint32)
        for index in range(args.max_epochs):
            counts[index] += torch.eq(trainset.pseudo_label, index).sum().numpy()
        print('\n\nCount:{}End\n'.format(counts))
        eer = dev_evaluate(model,args.val_list_path)
        if eer<best_eer:
            best_eer=eer

        if 'vox2' in args.train_list_path and args.max_epochs-epoch<=5:
            print("\n\nevaluate Vox-H...")
            eer_h=dev_evaluate(model,args.vox_h_path)
            print("\n\nevaluate Vox-E...")
            eer_e=dev_evaluate(model,args.vox_e_path)

        print('model save...')
        model_path = "{}/epoch={}_eer={:.4f}.pth".format(save_model_path, epoch, eer * 100)
        # torch.save(model.state_dict(), model_path)
        # model.load_state_dict(torch.load(model_path))
        torch.save(model, model_path)
        # model = torch.load('model_name.pth')
        print(model_path)

        epoch_end_time = datetime.datetime.now()
        epoch_time = epoch_end_time - epoch_start_time
        print('\nthis epoch time:{}'.format(epoch_time))
    train_end_time = datetime.datetime.now()
    train_time = train_end_time - train_start_time
    print('\ntotal train time:{}'.format(train_time))
