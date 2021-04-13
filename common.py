import argparse
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', default=1, type=int, metavar='N')  # 服务器数目
parser.add_argument('--n_gpu', default=1, type=int, help='number of gpus per node')  # 每台服务器上有多少个gpu
parser.add_argument('--rank', default=0, help='ranking within the nodes')  # 服务器优先级
parser.add_argument("--local_rank", type=int, default=0, help="local_rank for distributed training on gpus")  # 使用launch的参数，launch会自己传参
parser.add_argument('--init_method', default='tcp://127.0.0.1:23456', help="init-method")

args = parser.parse_args()

args.use_cuda = torch.cuda.is_available()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.distributed = False  # 是否使用分布式训练
# 如果使用 torch.distributed.launch 运行程序，将自动生成'WORLD_SIZE'环境变量。
# if 'WORLD_SIZE' in os.environ:
#     args.distributed = int(os.environ['WORLD_SIZE']) > 1
# 如果不使用 torch.distributed.launch 运行程序，计算word_size，大于1则进行分布式训练
args.world_size = args.n_gpu * args.nodes
if args.use_cuda and args.world_size > 1:
    args.distributed = True
# 如果只有一个gpu，运行在指定gpu上
if args.use_cuda and not args.distributed:
    args.device = torch.device('cuda:{}'.format(args.local_rank))

if args.use_cuda:
    logger.info("device %s n_gpu %d distributed training %r", args.device, args.n_gpu, bool(args.local_rank != -1))
else:
    logger.info("cpu")


























