from torch.utils.data import Dataset, DataLoader
from arg import *
from vocab import get_worddict, get_input


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, file_name):
        data_load = get_worddict(file_name)
        word2ind, ind2word, label2ind, ind2label, datas = data_load
        x, y = get_input(word2ind, label2ind, datas)
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


if __name__ == "__main__":
    file = args.file_train
    dataset = DealDataset(file)
    cla, sen = dataset.__getitem__(0)

    print(cla)
    print(sen)
    # 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。

    newLoader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
