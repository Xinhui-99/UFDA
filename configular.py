import yaml
import easydict
from os.path import join
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from wheel import *
from torchvision.transforms.transforms import *
from PIL import Image
from os import path
import warnings
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='3'
# disable warning of imread
warnings.filterwarnings('ignore', message='.*', category=Warning)

class Dataset:
    def __init__(self, path, domains):
        self.path = path
        self.domains = domains

parser = argparse.ArgumentParser(description='Code for *UFDA: Universal Federated Domain Adaptation with Practical Assumptions*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-dir', default='experiment/UFDA', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('--config', type=str, default='train-config-office311.yaml', help='/path/to/config/file')
parser.add_argument('--configs', type=str, default='train-config-office30.yamlv', help='/path/to/config/file')
parser.add_argument('-lr_decay_epochs', type=str, default='30,60,80', help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('-tem', type=float, default=0.05, help='temperature for soft max')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:18599', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or multi node data parallel training')
parser.add_argument('--low-dim', default=128, type=int, help='embedding dimension')
parser.add_argument('--moco_queue', default=256, type=int, help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float, help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the moving average of prototypes')
parser.add_argument('--re_pseu', type=float, default=0.4, help='parameter for label_set select')
parser.add_argument('--threthod_scores', type=float, default=0.11, help='parameter for time integrated')

parser.add_argument('--loss_weight', default=0.0, type=float, help='contrastive loss weight: default: 0.01')
parser.add_argument('--loss_penalty', default=0, type=float, help='contrastive loss weight of penalty: default: 0.01')
parser.add_argument('--conf_ema_range_start', type=float, default=0.85, help='pseudo target updating coefficient (phi)')
parser.add_argument('--conf_ema_range_end', type=float, default=0.65, help='pseudo target updating coefficient (phi)')

parser.add_argument('--prot_start', default=100, type=int, help='Start Prototype Updating')
parser.add_argument('--hierarchical', action='store_true', help='for dataset fine-grained training')
parser.add_argument('--alpha_z', type=float, default=0.6, help='parameter for time integrated')
parser.add_argument('--test', action='store_true', default=False, help='test')

parser.add_argument('--cosine', action='store_true', default=True, help='use cosine lr schedule')
parser.add_argument('--soft_source_label', action='store_true', default=False,
                    help='use soft outputs of source domains')
parser.add_argument('-t', '--train-time', default=1, type=str, metavar='N', help='the x-th time of training')
parser.add_argument('-bm', '--bn-momentum', type=float, default=0.1, help="the batchnorm momentum parameter")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pre_epochs', default=2, type=int, help='number of total epochs to run')

# Optimizer Parameters2
parser.add_argument('--optimizer', default="SGD", type=str, metavar="Optimizer Name")
parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')

argss = parser.parse_args()

config_file = argss.config
args = yaml.load(open(config_file), Loader=yaml.FullLoader)
save_config = yaml.load(open(config_file), Loader=yaml.FullLoader)
args = easydict.EasyDict(args)

dataset = None
if args.data.dataset.name == 'office':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['amazon', 'dslr', 'webcam'])
elif args.data.dataset.name == 'officehome':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['Art', 'Clipart', 'Product', 'Real_World'])
elif args.data.dataset.name == 'VisDA+ImageCLEF-DA':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['S', 'R', 'C', 'I', 'P'])
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')
domains = args.data.dataset.domains
domains.remove(args.data.dataset.target)
source1_domain_name = dataset.domains[0]
source1_file = path.join(args.data.dataset.root_path, domains[0])
target_domain_name = args.data.dataset.target
target_file =path.join(args.data.dataset.root_path, args.data.dataset.target)
if len(args.data.dataset.domains) >= 2:
     source2_domain_name = dataset.domains[1]
     source2_file = path.join(args.data.dataset.root_path, domains[1])

if len(args.data.dataset.domains) >= 3 :
     source3_domain_name = dataset.domains[2]
     source3_file = path.join(args.data.dataset.root_path, domains[2])

if len(args.data.dataset.domains) >= 4:
     source4_domain_name = dataset.domains[3]
     source4_file = path.join(args.data.dataset.root_path, domains[3])

