import os
import argparse
import pandas as pd
import numpy as np
import random
from tqdm import tqdm


def create_noisy_train_list(train_list_path, noisy_rate):
    print("\n============> noisy rate:{}\nstart adding noisy label to {}".format(noisy_rate, train_list_path))
    noisy_train_list_path = "{}_r={}_tag.csv".format(train_list_path[:-4], noisy_rate)
    if os.path.exists(train_list_path):
        df = pd.read_csv(train_list_path)
        del df["Unnamed: 0"]
        speaker = np.unique(df["utt_spk_int_labels"].values)
        num_classes = len(speaker)
        num_utt = len(df["utt_spk_int_labels"])
        num_noisy_label = int(num_utt*noisy_rate)
        noisy_tag = np.zeros(num_utt)
        print("number of speaker:{}".format(num_classes))
        print("number of utterance:{}".format(num_utt))
        print("number of noisy label:{}".format(num_noisy_label))
        res = random.sample(range(0, num_utt), num_noisy_label)
        res.sort()
        noisy_tag[res] = 1
        noisy_tag=noisy_tag.astype(int)
        modify_dict = {}
        for idx in res:
            speaker_id = df["utt_spk_int_labels"].values[idx]
            new_speaker_id = speaker[random.randint(0, num_classes - 1)]
            while new_speaker_id == speaker_id:
                new_speaker_id = speaker[random.randint(0, num_classes - 1)]
            modify_dict[idx] = new_speaker_id
        progress = tqdm(res)
        for idx in progress:
            progress.set_description("index:{}   {} --> {}".format(idx, df.iloc[idx, 2], modify_dict[idx]))
            df.iloc[idx, 2] = modify_dict[idx]
            progress.update()
        df['noisy_tag'] = noisy_tag
        df.to_csv(noisy_train_list_path)
        print("saved file:{}".format(noisy_train_list_path))
        df = pd.read_csv(noisy_train_list_path)
        new_num_speaker = len(np.unique(df["utt_spk_int_labels"].values))
        print("number of speaker with noisy label:{}".format(new_num_speaker))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list_path', help='list save path', type=str, default="vox1_train_list.csv")
    args = parser.parse_args()
    for rate in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        create_noisy_train_list(args.train_list_path, rate)
