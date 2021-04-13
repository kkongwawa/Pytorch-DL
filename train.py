import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
from model import *
import os
from arg import *
from data import DealDataset, DataLoader
import time


def evaluate_step(sentences, label, model, criterion):
    out, p = model(sentences)
    loss = criterion(out, label)
    return loss, p


def train_step(sentences, label, model, criterion, scaler, optimizer):
    if args.use_cuda:
        with autocast():  # 只有cuda设备下可以用自动混合精度
            out, p = model(sentences)
            loss = criterion(out, label)
        # Scales loss. 为了梯度放大.
        scaler.scale(loss).backward()
        # scaler.step() 首先把梯度的值unscale回来.
        # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
        # 否则，忽略step调用，从而保证权重不更新（不被破坏）
        scaler.step(optimizer)
        # 准备着，看是否要增大scaler
        scaler.update()
    else:
        out, p = model(sentences)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
    return loss, p


def cls_report(y_true, y_pred):
    target_names = ["class" + str(i) for i in range(args.class_num)]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n" + report + "\n" + "*"*50)
    time.sleep(0.05)


def evaluate(dev_loader, model, criterion):
    loss_total, cnt = 0, 0
    y_true, y_pred = [], []
    for i, (sentences, label) in enumerate(dev_loader):
        sentences = sentences.type(torch.LongTensor).to(args.device)
        label = label.type(torch.LongTensor).to(args.device)
        loss, p = evaluate_step(sentences, label, model, criterion)
        loss_total += loss
        cnt += 1
        y_true.extend(label.tolist())
        y_pred.extend(p.max(axis=1)[1].tolist())
    cls_report(y_true, y_pred)
    loss = loss_total/cnt

    return loss


def train(train_loader, dev_loader, model, criterion, optimizer, scaler, scheduler):
    # TODO 训练集准确率
    for epoch in range(args.epoch):
        for i, (sentences, label) in enumerate(train_loader):
            optimizer.zero_grad()
            sentences = sentences.type(torch.LongTensor).to(args.device)
            label = label.type(torch.LongTensor).to(args.device)
            loss, p = train_step(sentences, label, model, criterion, scaler, optimizer)

            if scheduler:
                data = "train: epoch {} step {} learning_rate {} --> loss {}\n".format(
                    str(epoch + 1), str(i + 1), scheduler.get_last_lr()[0], str(loss.item()))
            else:
                data = "train: epoch {} step {} --> loss {}\n".format(
                    str(epoch + 1), str(i + 1), str(loss.item()))
            logger.info(data)
        if scheduler:
            scheduler.step()
        if args.rank == 0:
            logger.info('start dev: epoch {}...'.format(epoch + 1))  # 每训练一代计算一次验证集准确率和loss
            evaluate_loss = evaluate(dev_loader, model, criterion)
            data = "dev: epoch {} --> loss {}\n".format(
                str(epoch + 1), str(evaluate_loss.item()))
            logger.info(data)
            logger.info("save model...")
            if args.save_model:
                torch.save(model, args.model_file)
                torch.save(model, "model\{}_model_iter_{}_loss_{:.2f}.pkl".format(
                    time.strftime('%y%m%d%H'), epoch, loss.item())
                           )
            else:
                torch.save(model.state_dict(), args.weight_file)
                torch.save(model.state_dict(), "model\{}_weight_iter_{}_loss_{:.2f}.pkl".format(
                    time.strftime('%y%m%d%H'), epoch, loss.item())
                           )


def main(gpu_num, args):
    # 每台主机有args.gpus个gpu,当前是第args.nodes台主机上的第gpu_num个gpu，由此计算rank
    rank = args.nodes * args.n_gpu + gpu_num
    if args.distributed:
        dist.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.world_size,
            rank=rank
        )
    torch.manual_seed(rank)
    logger.info('init model...')
    model = TextCNN(args).train()
    model.init_parameters()
    if os.path.exists(args.weight_file):
        logger.info('load weight...')
        model.load_state_dict(torch.load(args.weight_file))
    if os.path.exists(args.model_file):
        logger.info('load model...')
        model = torch.load(args.model_file)
    if args.distributed:
        torch.cuda.set_device(gpu_num)
        model.cuda(gpu_num)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_num])
    else:
        model.to(args.device)

    logger.info('init dataset...')
    dataset_train = DealDataset(args.file_train)
    dataset_dev = DealDataset(args.file_dev)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=args.world_size,
            rank=rank
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler)
    else:
        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
    dev_loader = None
    if args.rank == 0:
        dev_loader = DataLoader(dataset=dataset_dev, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.sdt_decay_step > 0:
        scheduler = StepLR(optimizer, step_size=args.sdt_decay_step, gamma=args.gamma)
    scaler = GradScaler()

    logger.info('start dev epoch 0...')  # 在未训练前，计算验证集准确率和loss
    if args.rank == 0:
        loss = evaluate(dev_loader, model, criterion)
        data = "dev: epoch 0 --> loss {}\n".format(str(loss.item()))
        logger.info(data)
    logger.info('start train...')  # 开始训练
    train(train_loader, dev_loader, model, criterion, optimizer, scaler, scheduler)


if __name__ == "__main__":

    if args.distributed:
        # mp.spawn会告诉train此时是当前主机的第i个进程，每个进程管理一个gpu，i从0到args.gpus-1
        mp.spawn(main, nprocs=args.gpus, args=(args,))
    else:
        main(-1, args)














