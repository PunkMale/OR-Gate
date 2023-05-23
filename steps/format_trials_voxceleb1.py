#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str, default="/home2/database/voxceleb/voxceleb1/dev/wav/")
    parser.add_argument('--src_trl_path', help='src_trials_path', type=str, default="/home2/database/voxceleb/voxceleb1/voxceleb1_test_v2.txt")
    parser.add_argument('--voxceleb1_test', help='voxceleb1_test', type=str, default="/home2/database/voxceleb/voxceleb1/test/wav/")
    parser.add_argument('--dst_trl_path', help='dst_trials_path', type=str, default="vox1_test_trials.lst")
    parser.add_argument('--apply_vad', action='store_true', default=False)
    args = parser.parse_args()
    # voxceleb1_test = '/home2/database/voxceleb/voxceleb1/test/wav/'

    trials = np.loadtxt(args.src_trl_path, dtype=str)

    f = open(args.dst_trl_path, "a+")
    for item in trials:
        if item[1][:7]>="id10270" and item[1][:7]<="id10309":
            enroll_path = os.path.join(args.voxceleb1_test, item[1])
        else:
            enroll_path = os.path.join(args.voxceleb1_root, item[1])
        if item[2][:7]>="id10270" and item[2][:7]<="id10309":
            test_path = os.path.join(args.voxceleb1_test, item[2])
        else:
            test_path = os.path.join(args.voxceleb1_root, item[2])
        if args.apply_vad:
            enroll_path = enroll_path.strip("*.wav") + ".vad"
            test_path = test_path.strip("*.wav") + ".vad"
        f.write("{} {} {}\n".format(item[0], enroll_path, test_path))

