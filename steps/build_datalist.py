#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import tqdm
import pandas as pd
import numpy as np

def findAllSeqs(dirName,
                extension='.wav',
                speaker_level=1):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers
        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index
        outSpeakers
        The speaker labels (in order)
    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension
    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label. Ex if speaker_level == 2 then the
    dataset should be organized in the following fashion
    \dirName
        \crappy_label
            \speaker_label
                \..
                    ...
                    seqName.extension
    Set speaker_label == 0 if no speaker label will be retrieved no matter the
    organization of the dataset.
    """
    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []
    print("finding {}, Waiting...".format(extension))
    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]
        # if len(root)>39 and int(root[39:44])>=10270 and int(root[39:44])<=10309:
        #     continue
        if len(filtered_files) > 0:
            speakerStr = (os.sep).join(
                root[prefixSize:].split(os.sep)[:speaker_level])
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            for filename in filtered_files:
                full_path = os.path.join(root, filename)
                outSequences.append((speaker, full_path))
    outSpeakers = [None for x in speakersTarget]

    for key, index in speakersTarget.items():
        outSpeakers[index] = key

    print("find {} speakers".format(len(outSpeakers)))
    print("find {} utterance".format(len(outSequences)))

    return outSequences, outSpeakers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', help='file extension name', type=str, default="wav")
    parser.add_argument('--data_dir', help='dataset dir', type=str, default="/home2/database/voxceleb/voxceleb2_16k/dev/aac/")
    parser.add_argument('--data_list_path', help='list save path', type=str, default="vox2_train_list.csv")
    parser.add_argument('--speaker_level', help='list save path', type=int, default=1)
    args = parser.parse_args()

    outSequences, outSpeakers = findAllSeqs(args.data_dir,
                extension=args.extension,
                speaker_level=1)

    outSequences = np.array(outSequences, dtype=str)
    utt_spk_int_labels = outSequences.T[0].astype(int)
    utt_paths = outSequences.T[1]
    utt_spk_str_labels = []
    for i in utt_spk_int_labels:
        utt_spk_str_labels.append(outSpeakers[i])

    print("Spker ID   : {}".format(utt_spk_str_labels[0]))
    print("Utter Path : {}".format(utt_paths[0]))
    print("Spker Label  : {}".format(utt_spk_int_labels[0]))

    csv_dict = {"speaker_name": utt_spk_str_labels,
            "utt_paths": utt_paths,
            "utt_spk_int_labels": utt_spk_int_labels
            }
    df = pd.DataFrame(data=csv_dict)

    try:
        df.to_csv(args.data_list_path)
        print(f'Saved data list file at {args.data_list_path}')
    except OSError as err:
        print(f'Ran in an error while saving {args.data_list_path}: {err}')


