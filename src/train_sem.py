import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import SemNet
from datasets.a2d2 import A2D2
from utils.utils import multi_acc, mkdir, denormalize
from torch.utils.tensorboard import SummaryWriter
from configs.cfg import args


def main(
    need_val=True,
    save_net=True,
    resume_train=False
):
    """
    single task of semantic segementation.
    Net: SemNet(default, ResUNet)
    Dataset: a2d2(default);
    Optimizer: Adam(default)
    :param need_val: bool, need validation process or not
    :param save_net: bool, save net.pth or not
    :param resume_train: bool, load the net.pth and continue to train or not
    :return: none
    """
    # create a new folder to save net.pth and tensorboard documents
    save_path = mkdir(args['save_path'])

    # create a log writer
    writer = SummaryWriter(save_path)

    # set device cuda or cpu
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # set network
    sem_net = SemNet(
        args['net'],
        sem_num_classes=args['sem_num_classes'],
        pretrained=args['pretrained']
    )

    # resume a train with the loaded net.pth
    if resume_train:
        sem_net.load_state_dict(
            torch.load(
                os.path.join(args['root'],
                             '2020-09-02-03-49-00_net_parameters_final.pth')
            )
        )

    sem_net.to(device).train()

    # set optimizer as Adam(default) or SGD
    if args['optim'] == 'Adam':
        optimizer = optim.Adam(
            sem_net.parameters(),
            lr=args['lr']
        )

    elif args['optim'] == 'SGD':
        optimizer = optim.SGD(
            sem_net.parameters(),
            lr=args['lr'],
            momentum=args['momentum']
        )
        # shrink learning rate by step size
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args['step_size'],
            gamma=args['gamma']
        )

    # set loss function, the loss weight is set as [1: 2: 4]
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(args['seg_loss_weight']).to(device)
    )

    # set dataset a2d2
    # partial is the amount of data, partial=None for the whole dataset
    data_set = A2D2(task='sem_seg', partial=args['partial'])

    # train:val:test = 6:2:2
    train_data_len = int(0.6 * len(data_set))
    val_data_len = int(0.2 * len(data_set))
    test_data_len = len(data_set) - train_data_len - val_data_len

    train_set, val_set, test_set = torch.utils.data.random_split(
        data_set, [train_data_len, val_data_len, test_data_len]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args['train_batch_size'],
        num_workers=args['num_workers'],
        shuffle=args['shuffle'],
    )

    # create validation dataloader only if need_val
    if need_val:
        val_loader = DataLoader(
            val_set,
            batch_size=args['val_batch_size'],
            num_workers=args['num_workers'],
            shuffle=args['shuffle'],
        )

    # write all the config parameters into tensorboard
    for key, value in args.items():
        writer.add_text(str(key), str(value))

    # start training
    for epoch in range(args['epoch']):
        train(
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            sem_net=sem_net,
            criterion=criterion,
            epoch=epoch,
            writer=writer,
        )

        if need_val:
            val(
                val_loader=val_loader,
                device=device,
                sem_net=sem_net,
                criterion=criterion,
                epoch=epoch,
                writer=writer,
            )

        if save_net:
            net_path = os.path.join(
                save_path, 'net_parameters' + '.pth'
            )
            torch.save(sem_net.state_dict(), net_path)

        # update learning rate in SGD
        if args['optim'] == 'SGD':
            scheduler.step()
    writer.close()


def train(
    train_loader,
    device,
    optimizer,
    sem_net,
    criterion,
    epoch,
    writer,
):
    """
    training process of semantic segmentation
    :param train_loader: DataLoader
    :param device: torch.device
    :param optimizer:Adam
    :param sem_net: Sem_Net
    :param criterion: CrossEntropy
    :param epoch: the current epoch number
    :param writer: tensorboard
    :return: none
    """
    with tqdm(total=len(train_loader)) as epoch_bar_train:
        # reset the loss and accuracy values at the beginning of each epoch
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0

        for i, data in enumerate(train_loader):
            # get the input and grundtruth label
            inputs_train = data[0].to(device)
            labels_train = data[1].to(device)

            # reset the gradient of the optimizer
            optimizer.zero_grad()

            outputs_train = sem_net(inputs_train)[0]

            # calculate loss value and accuracy by pixel-pixel comparison
            loss_train = criterion(outputs_train, labels_train)
            accuracy_train = multi_acc(outputs_train, labels_train)

            train_epoch_loss += loss_train.detach().item()
            train_epoch_acc += accuracy_train

            loss_train.backward()
            optimizer.step()

            # write the loss value each iteration: fine loss
            writer_index_train = epoch * len(train_loader) + i
            writer.add_scalar(
                '(fine)Train Loss',
                loss_train.item(),
                writer_index_train
            )

            # demonstrate the epoch bar with tqdm
            epoch_bar_train.set_description(
                f'Train Epoch: {epoch} |\
                     Loss: {loss_train.item()} |\
                          Acc: {accuracy_train}'
            )
            epoch_bar_train.update(1)

        # write information at each end of the epoch
        img_denormalized_a2d2 = denormalize(inputs_train[0],  # convert the normalized image to original
                                            mean=[0.4499, 0.4365, 0.4364],
                                            std=[0.1902, 0.1972, 0.2085])

        writer.add_scalar('Train Loss', train_epoch_loss/len(train_loader),
                          epoch)
        writer.add_scalar('Train Accuracy', train_epoch_acc/len(train_loader),
                          epoch)
        writer.add_image('Train Image',
                         img_denormalized_a2d2,
                         epoch,
                         dataformats='CHW')
        writer.add_image('Train Label',
                         labels_train[0].float() /
                         args['sem_num_classes'],
                         epoch,
                         dataformats='HW')
        writer.add_image('Train Prediction',
                         outputs_train[0].argmax(0).float() /
                         args['sem_num_classes'],
                         epoch,
                         dataformats='HW')


def val(
    val_loader,
    device,
    sem_net,
    criterion,
    epoch,
    writer,
):
    """
    validation process of semantic segmentation, compare with train loop, validation don't need optimizer
    and don't need write the loss value each iteration
    :param val_loader: DataLoader
    :param device: torch.device
    :param sem_net: SemNet
    :param criterion:
    :param epoch: the current epoch number
    :param writer: tensorboard
    :return:
    """
    with tqdm(total=len(val_loader)) as epoch_bar_val:
        # reset the loss and accuracy value at the beginning of each epoch
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        with torch.no_grad():
            for i, data_val in enumerate(val_loader):
                # get the input and grundtruth label
                inputs_val = data_val[0].to(device)
                labels_val = data_val[1].to(device)

                outputs_val = sem_net(inputs_val)[0]

                # calculate loss value and accuracy by pixel-pixel comparison
                loss_val = criterion(outputs_val, labels_val)
                accuracy_val = multi_acc(outputs_val, labels_val)

                val_epoch_loss += loss_val.detach().item()
                val_epoch_acc += accuracy_val

                # demonstrate the epoch bar with tqdm
                epoch_bar_val.set_description(
                    f'Validation Epoch: {epoch+1} | \
                        Loss: {loss_val.item()} | \
                            Acc: {accuracy_val}')
                epoch_bar_val.update(1)

            # write information at each end of the epoch
            img_denormalized_a2d2 = denormalize(inputs_val[0],  # convert the normalized image to original
                                                mean=[0.4499, 0.4365, 0.4364],
                                                std=[0.1902, 0.1972, 0.2085])

            writer.add_scalar('Val Loss', val_epoch_loss/len(val_loader),
                              epoch)
            writer.add_scalar('Val Accuracy', val_epoch_acc/len(val_loader),
                              epoch)
            writer.add_image('Val Image',
                             img_denormalized_a2d2,
                             epoch,
                             dataformats='CHW')
            writer.add_image('Val Lable',
                             labels_val[0].float() /
                             args['sem_num_classes'],
                             epoch,
                             dataformats='HW')
            writer.add_image('Val Prediction',
                             outputs_val[0].argmax(0).float() /
                             args['sem_num_classes'],
                             epoch,
                             dataformats='HW')


if __name__ == '__main__':
    main()
