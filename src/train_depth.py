import os
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import DepthUNet
from models.depth_decode import discretization_decode
from models.losses import *
from datasets.cityscapes import Cityscapes
from torch.utils.tensorboard import SummaryWriter
from configs.cfg import args
from utils.utils import denormalize, mkdir, apply_color_map


def main(
    need_val=True,
    save_net=True,
    resume_train=False
):
    """
    single task of depth estimation
    Net: DepthUNet(default, ResUNet)
    Dataset: a2d2(default);
    Optimizer: Adam(default)
    :param need_val: bool, need validation process or not
    :param save_net: bool, save net.pth or not
    :param resume_train: bool, load the net.pth and continue to train or not
    :return:  none
    """
    # get the save path
    save_path = mkdir(args['save_path'])

    # create a log writer
    writer = SummaryWriter(save_path)

    # set device cuda or cpu
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # set network
    depth_net = DepthUNet(args['net'])

    # resume a train with the loaded net.pth
    if resume_train:
        depth_net.load_state_dict(
            torch.load(
                os.path.join(args['root'])
            )
        )

    depth_net.to(device).train()

    # set optimizer as Adam(default) or SGD
    if args['optim'] == 'Adam':
        optimizer = optim.Adam(
            depth_net.parameters(),
            lr=args['lr']
        )

    elif args['optim'] == 'SGD':
        optimizer = optim.SGD(
            depth_net.parameters(),
            lr=args['lr'],
            momentum=args['momentum']
        )

        # shrink learning rate by step size
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args['step_size'],
            gamma=args['gamma']
        )

    # set loss function
    criterion = ScaleInvariantLogLoss()

    # set dataset Cityscapes
    # partial is the amount of data, partial=None for the whole dataset
    # the dataset of Cityscapes is already split by official
    train_set = Cityscapes(mode='train', task='depth',
                           train_extra=args['train_extra'], partial=None)

    # create validation dataloader only if need_val
    if need_val:
        val_set = Cityscapes(mode='val', task='depth', train_extra=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args['train_batch_size'],
        num_workers=args['num_workers'],
        shuffle=args['shuffle']
    )

    if need_val:
        val_loader = DataLoader(
            val_set,
            batch_size=args['train_batch_size'],
            num_workers=args['num_workers'],
            shuffle=False
        )

    # write all teh config parameters into tensorboard
    for key, value in args.items():
        writer.add_text(str(key), str(value))

    # start training
    for epoch in range(args['epoch']):
        train(
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            depth_net=depth_net,
            criterion=criterion,
            epoch=epoch,
            writer=writer,
        )

        if need_val:
            val(
                val_loader=val_loader,
                device=device,
                depth_net=depth_net,
                criterion=criterion,
                epoch=epoch,
                writer=writer,
            )

        if save_net:
            net_path = os.path.join(
                save_path, 'net_parameters' + '.pth'
            )
            torch.save(depth_net.state_dict(), net_path)

        # update learning rate in SGD
        if args['optim'] == 'SGD':
            scheduler.step()
    writer.close()


def train(
    train_loader,
    device,
    optimizer,
    depth_net,
    criterion,
    epoch,
    writer,
):
    """
    training process of depth estimation
    :param train_loader: DataLoader
    :param device: torch.device
    :param optimizer: Adam
    :param depth_net: DepthUNet
    :param criterion: ScaleInvarianteLogLoss
    :param epoch: the current epoch number
    :param writer: tensorboard
    :return: none
    """
    with tqdm(total=len(train_loader)) as epoch_bar_train:
        # reset the loss value at the beginning of each epoch
        train_epoch_loss = 0.0

        for i, data in enumerate(train_loader):
            # get the input and grundtruth label
            inputs_train = data[0].to(device)
            labels_train = data[1].to(device)

            # reset the gradient of the optimizer
            optimizer.zero_grad()

            outputs_train = depth_net(inputs_train)[0].squeeze()

            # calculate loss value
            loss_train = criterion(outputs_train, labels_train)

            train_epoch_loss += loss_train.item()

            loss_train.backward()
            optimizer.step()

            # writer the loss value each iteration: fine loss
            writer_index_train = epoch * len(train_loader) + i
            writer.add_scalar(
                '(fine)Train Loss',
                loss_train.item(),
                writer_index_train
            )

            # demonstrate the epoch bar with tqdm
            epoch_bar_train.set_description(
                f'Train Epoch: {epoch} |\
                     Loss: {loss_train.item()}'
            )
            epoch_bar_train.update(1)

        # writer information at each end of the epoch
        img_denormalized = denormalize(  # convert the normalized image to original
            inputs_train[0],
            mean=[0.3160, 0.3553, 0.3110],
            std=[0.1927, 0.1964, 0.1965]
        )

        # apply a better color appearance for disparity image
        label_show = apply_color_map(labels_train[0])
        pred_show = apply_color_map(outputs_train[0])

        writer.add_scalar('Train Loss',
                          train_epoch_loss/len(train_loader),
                          epoch)
        writer.add_image('Train Image',
                         img_denormalized,
                         epoch,
                         dataformats='CHW')
        writer.add_image('Train Label',
                         label_show,
                         epoch,
                         dataformats='HWC')
        writer.add_image('Train Prediction',
                         pred_show,
                         epoch,
                         dataformats='HWC')


def val(
    val_loader,
    device,
    depth_net,
    criterion,
    epoch,
    writer,
):
    """
    validation process of semantic segmentation, compare with train loop, validation don't need optimizer
    and don't need write the loss value each iteration
    :param val_loader: DataLoader
    :param device: torch.device
    :param depth_net: DepthUNet
    :param criterion:
    :param epoch: the current epoch number
    :param writer: tensorboard
    :return: none
    """
    with tqdm(total=len(val_loader)) as epoch_bar_val:
        # reset the loss value at the beginning of each epoch
        val_epoch_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # get the input and grundtruth label
                inputs_val = data[0].to(device)
                labels_val = data[1].to(device)

                outputs_val = depth_net(inputs_val)[0].squeeze()

                # calculate loss value
                loss_val = criterion(outputs_val, labels_val)

                val_epoch_loss += loss_val.item()

                # writer loss value each iteration
                writer_index_val = epoch * len(val_loader) + i
                writer.add_scalar(
                    '(fine)Val Loss',
                    loss_val.item(),
                    writer_index_val
                )

                # demonstrate the epoch bar
                epoch_bar_val.set_description(
                    f'Val Epoch: {epoch} |\
                        Loss: {loss_val.item()}'
                )
                epoch_bar_val.update(1)

            # writer information at each end of the epoch
            img_denormalized = denormalize(
                inputs_val[0],
                mean=[0.3160, 0.3553, 0.3110],
                std=[0.1927, 0.1964, 0.1965])
            label_show = apply_color_map(labels_val[0])
            pred_show = apply_color_map(outputs_val[0])
            writer.add_scalar('Val Loss',
                              val_epoch_loss/len(val_loader),
                              epoch)
            writer.add_image('Train Image',
                             img_denormalized,
                             epoch,
                             dataformats='CHW')
            writer.add_image('Train Label',
                             label_show,
                             epoch,
                             dataformats='HWC')
            writer.add_image('Train Prediction',
                             pred_show,
                             epoch,
                             dataformats='HWC')


if __name__ == '__main__':
    main()
