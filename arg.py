from common import *

args.kernel_num = 4  # output chanel size
args.kernel_size = [2, 3, 5]
args.vocab_size = 50
args.embed_dim = 64
args.dropout = 0.9
args.class_num = 2
args.batch_size = 2
args.file_train = "./data_train.txt"
args.file_dev = "./data_dev.txt"
args.sentence_length = 30
args.epoch = 50
args.save_dir = "./model"
args.weight_file = "weight.pkl"
args.model_file = "model.pkl"
args.save_model = False
args.sdt_decay_step = 0  # 间隔多少epoch调整学习率，0表示不调整
args.gamma = 0.1  # StepLR参数
args.learning_rate = 0.03
