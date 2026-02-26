import argparse

parser = argparse.ArgumentParser(description='EPGDUN')

# ------------------ Checkpoint ------------------
parser.add_argument('--load', type=str, default='.', help='load model from specific checkpoint')
parser.add_argument('--save', type=str, default='EPGDUN', help='file name to save')
parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')
parser.add_argument('--resume', type=int, default=0, help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true', help='print model')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=50, help='batches to wait before logging')
parser.add_argument('--save_results', type=bool, default=True, help='save output results')

# ------------------ Hardware ------------------
parser.add_argument('--n_threads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# ------------------ Data ------------------
parser.add_argument('--dir_data', type=str, default='../dataset', help='dataset directory')
parser.add_argument('--data_train', type=str, default='DIV2K', help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K', help='test dataset name')
parser.add_argument('--scale', type=int, nargs='+', default=[2], help='super resolution scale')
parser.add_argument('--ext', type=str, default='img', help='dataset file extension')
parser.add_argument('--patch_size', type=int, default=48, help='LR patch size for training')
parser.add_argument('--benchmark', action='store_true', help='use benchmark dataset')

# ------------------ Dataset size / offsets ------------------
parser.add_argument('--n_train', type=int, default=800, help='number of training images')
parser.add_argument('--n_val', type=int, default=100, help='number of validation images')
parser.add_argument('--offset_train', type=int, default=0, help='start index for training images')
parser.add_argument('--offset_val', type=int, default=800, help='start index for validation images')

# ------------------ Model ------------------
parser.add_argument('--model', type=str, default='EPGDUN', help='model name')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels')
parser.add_argument('--self_ensemble', action='store_true', help='use self-ensemble method for test')
parser.add_argument('--chop', action='store_true', help='enable memory-efficient forward')
parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test')
parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--testset', type=str, default='DIV2K', help='dataset name for testing')
parser.add_argument('--gan_k', type=int, default=1, help='k value for adversarial loss')

# ------------------ Training ------------------
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--split_batch', type=int, default=1, help='split the batch into smaller chunks')

# ------------------ Optimization ------------------
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--lr_decay', type=int, default=400, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='Mstep_300_450_600', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'), help='optimizer to use')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')

# ------------------ Loss ------------------
parser.add_argument('--loss', type=str, default='1*L1', help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default=1e10, help='skip batch with large error')

# ------------------ Patch / Noise ------------------
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--noise', type=str, default='none', help='Gaussian noise std.')

# ------------------ Parse ------------------
args = parser.parse_args()
