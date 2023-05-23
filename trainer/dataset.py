#! /usr/bin/python
# -*- encoding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
import random
import datetime
import soundfile
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from threading import Thread


def loadWAV(filename, max_frames, evalmode=False, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    # sample_rate, audio = wavfile.read(filename)
    audio, sample_rate = soundfile.read(filename) 
    # print('audio.shape:{}'.format(audio.shape))

    audiosize = audio.shape[0]

    # padding
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        # (0, storage)表示左边补充0个，右边补充storage个
        # wrap表示，后面补充前面，前面补充后面，例如[4,5, 1,2,3,4,5, 1,2]
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    # np.linspace(start, end, num_points) 在start和end之间生成num_points个序列
    if evalmode:
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random() * (audiosize - max_audio))])
    # print("startframe:{}".format(startframe))

    feats = []
    if evalmode and num_eval == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
    # np.stack:把几个数组对叠起来，axis就是在哪个维度进行堆叠，在这里堆叠没有改变shape，只是将list转换为了numpy
    feat = np.stack(feats, axis=0).astype(float)
    return feat


class AugmentWAV(object):
    def __init__(self, musan_data_list_path, rirs_data_list_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_audio = max_frames * 160 + 240
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noiselist = {}

        df = pd.read_csv(musan_data_list_path)
        augment_files = df["utt_paths"].values
        augment_types = df["speaker_name"].values
        for idx, file in enumerate(augment_files):
            if not augment_types[idx] in self.noiselist:
                self.noiselist[augment_types[idx]] = []
            self.noiselist[augment_types[idx]].append(file)
        df = pd.read_csv(rirs_data_list_path)
        self.rirs_files = df["utt_paths"].values

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        audio = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio
        return audio.astype(np.int16).astype(float)

    def reverberate(self, audio):
        rirs_file = random.choice(self.rirs_files)
        fs, rirs = wavfile.read(rirs_file)
        rirs = np.expand_dims(rirs.astype(float), 0)
        rirs = rirs / np.sqrt(np.sum(rirs ** 2))
        if rirs.ndim == audio.ndim:
            audio = signal.convolve(audio, rirs, mode='full')[:, :self.max_audio]
        return audio.astype(np.int16).astype(float)


class Train_Dataset(Dataset):
    def __init__(self, data_list_path=None, augment=False, musan_list_path=None, rirs_list_path=None, max_frames=200):
        if data_list_path is None:
            print('create a empty dataset')
            self.data_list = []
            self.data_label = torch.tensor([], dtype=torch.uint8)
            self.pseudo_label = torch.tensor([], dtype=torch.float32)
            self.noisy_tag = torch.tensor([], dtype=torch.int16)
        else:
            # load data list
            self.data_list_path = data_list_path
            print('load {}'.format(self.data_list_path))
            df = pd.read_csv(data_list_path)
            # utt对应的说话人id
            self.data_label = df["utt_spk_int_labels"].values
            self.noisy_tag = df["noisy_tag"].values
            # 所有的utt路径
            data_list = df["utt_paths"].values
            self.data_list = []
            for path in data_list:
                if os.path.exists(path) and path[-3:] == 'wav':
                    self.data_list.append(path)
            # 集成标签
            self.pseudo_label = torch.zeros(len(self.data_list))
            # self.pseudo_label = torch.rand(len(self.data_list))
            print("Speaker:{}".format(len(np.unique(self.data_label))))
            print("Utterance:{}".format(len(self.data_list)))
            print('noisy number:{}'.format(np.sum(self.noisy_tag)))

        # 语音增强
        if augment:
            self.augment_wav = AugmentWAV(musan_list_path, rirs_list_path, max_frames=max_frames)
        self.augment = augment
        self.max_frames = max_frames

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames)
        if self.augment:
            augtype = random.randint(0, 4)
            if augtype == 1:
                audio = self.augment_wav.reverberate(audio)
            elif augtype == 2:
                audio = self.augment_wav.additive_noise('music', audio)
            elif augtype == 3:
                audio = self.augment_wav.additive_noise('speech', audio)
            elif augtype == 4:
                audio = self.augment_wav.additive_noise('noise', audio)
        # 返回数据和标签
        return torch.FloatTensor(audio), self.data_label[index], self.pseudo_label[index], self.data_list[index], self.noisy_tag[index]

    def __len__(self):
        return len(self.data_list)

    def add_data(self, data_path, label, pseudo_label, noisy_tag):
        self.data_list += data_path
        self.data_label = torch.cat((self.data_label, label), 0)
        self.pseudo_label = torch.cat((self.pseudo_label, pseudo_label), 0)
        self.noisy_tag = torch.cat((self.noisy_tag, noisy_tag), 0)

class Dev_Dataset(Dataset):
    def __init__(self, data_list_path, eval_frames, num_eval=0, **kwargs):
        self.data_list_path = data_list_path
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values
        print("Dev Dataset load {} speakers".format(len(np.unique(self.data_label))))
        print("Dev Dataset load {} utterance".format(len(self.data_list)))

        self.max_frames = eval_frames
        self.num_eval = num_eval

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class Test_Dataset(Dataset):
    def __init__(self, data_list, eval_frames, num_eval=0, **kwargs):
        # load data list
        self.data_list = data_list
        self.max_frames = eval_frames
        self.num_eval = num_eval

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.data_list[index]

    def __len__(self):
        return len(self.data_list)


def divide_dataset(dataset):
    print('divide dataset')

    print('start time:{}'.format(datetime.datetime.now()))
    labeled_dataset = Train_Dataset()
    unlabeled_dataset = Train_Dataset()

    label_index = torch.ge(dataset.pseudo_label, 1)
    unlabel_index = torch.lt(dataset.pseudo_label, 1)
    data = np.array(dataset.data_list)
    if not torch.is_tensor(dataset.data_label):
        dataset.data_label = torch.from_numpy(dataset.data_label)
    labeled_dataset.add_data(list(data[label_index]), dataset.data_label[label_index], dataset.pseudo_label[label_index], dataset.noisy_tag[label_index])
    unlabeled_dataset.add_data(list(data[unlabel_index]), dataset.data_label[unlabel_index], dataset.pseudo_label[unlabel_index], dataset.noisy_tag[unlabel_index])
    print('end time:{}'.format(datetime.datetime.now()))

    print('labeled dataset size:{}#'.format(len(labeled_dataset)))
    print('unlabeled dataset size:{}#'.format(len(unlabeled_dataset)))

    return labeled_dataset, unlabeled_dataset



