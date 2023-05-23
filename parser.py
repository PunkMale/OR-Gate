from argparse import ArgumentParser


########################
noisy_rate = 0.0
train_list_path = 'data/vox1_train_list_r={}_tag.csv'.format(noisy_rate)
device = "cuda:0"
warm_up = 5      # when the warm_up = 80, it is equivalent to baseline
topk = 90
max_epochs = 80
batch_size = 128
########################

val_list_path = 'data/VoxCeleb1-Clean.lst'
vox_h_path = "data/VoxCeleb1-H.lst"
vox_e_path = "data/VoxCeleb1-E.lst"

musan_list_path = None
rirs_list_path = None
nnet_type = "resnet34"
loss_type = "amsoftmax"
pooling_type = "ASP"
alpha = 0.5
start_epoch = 0
evaluate = False
filter = False
augment = False
n_mels = 80
max_frames = 201
min_frames = 200
nPerSpeaker = 1
max_seg_per_spk = 500
num_workers = 20
embedding_dim = 512
learning_rate = 0.0002
lr_step_size = 5
lr_gamma = 0.40
margin = 0.2
scale = 30.0
eval_interval = 5
eval_frames = 0

parser = ArgumentParser()

# Data Loader
parser.add_argument('--max_frames', type=int, default=max_frames)
parser.add_argument('--min_frames', type=int, default=min_frames)
parser.add_argument('--eval_frames', type=int, default=eval_frames)
parser.add_argument('--batch_size', type=int, default=batch_size)
# Maximum number of utterances per speaker per epoch
parser.add_argument('--max_seg_per_spk', type=int, default=max_seg_per_spk, help='')
# Number of utterances per speaker per batch, only for metric learning based losses
parser.add_argument('--nPerSpeaker', type=int, default=nPerSpeaker, help='')
parser.add_argument('--num_workers', type=int, default=num_workers)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--augment', action='store_true', default=augment)

# Training details
parser.add_argument('--max_epochs', type=int, default=max_epochs, help='Maximum number of epochs')
parser.add_argument('--start_epoch', type=str, default=start_epoch)
parser.add_argument('--loss_type', type=str, default=loss_type)
parser.add_argument('--nnet_type', type=str, default=nnet_type)
parser.add_argument('--pooling_type', type=str, default=pooling_type)
parser.add_argument('--eval_interval', type=int, default=eval_interval)
parser.add_argument('--keep_loss_weight', action='store_true', default=False)
parser.add_argument('--noisy_rate', type=float, default=noisy_rate)
parser.add_argument('--alpha', default=alpha)
parser.add_argument('--filter', default=filter)
parser.add_argument('--warm_up', default=warm_up)
parser.add_argument('--topk', default=topk)

# Optimizer
parser.add_argument('--learning_rate', type=float, default=learning_rate)
parser.add_argument('--lr_step_size', type=int, default=lr_step_size)
parser.add_argument('--lr_gamma', type=float, default=lr_gamma)
parser.add_argument('--auto_lr', action='store_true', default=False)

# Loss functions
parser.add_argument('--margin', type=float, default=margin)
parser.add_argument('--scale', type=float, default=scale)

# Training and test data
parser.add_argument('--train_list_path', type=str, default=train_list_path)
parser.add_argument('--val_list_path', type=str, default=val_list_path)
parser.add_argument('--scores_path', type=str, default='scores.foo')
parser.add_argument('--apply_metric', action='store_true', default=False)
parser.add_argument('--musan_list_path', type=str, default=musan_list_path)
parser.add_argument('--rirs_list_path', type=str, default=rirs_list_path)
parser.add_argument('--vox_h_path', type=str, default=vox_h_path)
parser.add_argument('--vox_e_path', type=str, default=vox_e_path)

# Load and save
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--save_top_k', type=int, default=15)
parser.add_argument('--suffix', type=str, default='')

# Model definition
parser.add_argument('--n_mels', type=int, default=n_mels)
parser.add_argument('--embedding_dim', type=int, default=embedding_dim)
parser.add_argument('--apply_plda', action='store_true', default=False)
parser.add_argument('--plda_dim', type=int, default=128)

# Test mode
parser.add_argument('--evaluate', action='store_true', default=evaluate)

# Device
parser.add_argument('--device', default=device)

args = parser.parse_args()
