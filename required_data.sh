#!/bin/bash

SPEAKER_TRAINER_ROOT=./
voxceleb1_root=/home2/database/voxceleb/voxceleb1/dev/wav/
voxceleb1_test=/home2/database/voxceleb/voxceleb1/test/wav/
voxceleb2_root=/home2/database/voxceleb/voxceleb2_16k/dev/aac/


# prepare data for model training
# mkdir -p data
# mkdir -p save_model
# echo Build $voxceleb1_root list
# python3 $SPEAKER_TRAINER_ROOT/steps/build_datalist.py \
#         --data_dir $voxceleb1_root \
#         --extension wav \
#         --speaker_level 1 \
#         --data_list_path data/vox1_train_list.csv

# # add noisy label to data
# python3 $SPEAKER_TRAINER_ROOT/steps/add_noisy_label_in_datalist.py \
#         --train_list_path data/vox1_train_list.csv

# echo Build $voxceleb2_root list
# python3 $SPEAKER_TRAINER_ROOT/steps/build_datalist.py \
#        --data_dir $voxceleb2_root \
#        --extension wav \
#        --speaker_level 1 \
#        --data_list_path data/vox2_train_list.csv

# add noisy label to data
python3 $SPEAKER_TRAINER_ROOT/steps/add_noisy_label_in_datalist.py \
        --train_list_path data/vox2_train_list.csv

# # prepare test trials for evaluation
python3 steps/format_trials_voxceleb1.py \
        --voxceleb1_root $voxceleb1_root \
        --src_trl_path /home/fangzh21/data/voxceleb/veri_test2.txt \
        --dst_trl_path data/VoxCeleb1-Clean.lst \
        --voxceleb1_test $voxceleb1_test

python3 steps/format_trials_voxceleb1.py \
        --voxceleb1_root $voxceleb1_root \
        --src_trl_path /home/fangzh21/data/voxceleb/list_test_all2.txt \
        --dst_trl_path data/VoxCeleb1-E.lst \
        --voxceleb1_test $voxceleb1_test

python3 steps/format_trials_voxceleb1.py \
        --voxceleb1_root $voxceleb1_root \
        --src_trl_path /home/fangzh21/data/voxceleb/list_test_hard2.txt \
        --dst_trl_path data/VoxCeleb1-H.lst \
        --voxceleb1_test $voxceleb1_test
