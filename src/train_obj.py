import os
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import DetNet
from models.losses import DetLoss
from models.heatmap_decode import decode
from datasets.a2d2 import A2D2
from torch.utils.tensorboard import SummaryWriter
from configs.cfg import args
from utils.utils import denormalize, draw_pred_bboxes, draw_gt_bboxes, mean_average_precision, mkdir


def main(
    need_val=True,
    save_net=True,
    resume_train=False
):
    """
    single task of object detection.
    Net: CenterNet(default)
    Dataset: a2d2(default)
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

    # seg device cuda of cpu
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # set network
    det_net = DetNet(
        args['net'],
        det_num_classes=args['det_num_classes'],
        pretrained=args['pretrained']
    )
    # resume a train with the loaded net.pth
    if resume_train:
        det_net.load_state_dict(
            torch.load(
                os.path.join(args['root'],
                             '/home/wuyin/Documents/ZhuBangyu/src/2020-09-01-14-20-06_net_parameters_final.pth')))

    det_net.to(device).train()

    # set optimizer as Adam(default) of SGD
    if args['optim'] == 'Adam':
        optimizer = optim.Adam(
            det_net.parameters(),
            lr=args['lr'],
        )

    elif args['optim'] == 'SGD':
        optimizer = optim.SGD(
            det_net.parameters(),
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
    criterion = DetLoss(weights=args['det_loss_weight'])

    # set dataset a2d2
    # partial is the amount of data, partial=None for the whole dataset
    data_set = A2D2(partial=args['partial'], task='obj_det')

    # train:val:test = 6:2:2
    train_data_len = int(0.6 * len(data_set))
    val_data_len = int(0.6 * len(data_set))
    test_data_len = len(data_set) - train_data_len - val_data_len

    train_set, val_set, test_set = torch.utils.data.random_split(
        data_set, [train_data_len, val_data_len, test_data_len])

    train_loader = DataLoader(
        train_set,
        batch_size=args['train_batch_size'],
        num_workers=args['num_workers'],
        shuffle=args['shuffle']
    )
    # create validation dataloader only if need_val
    if need_val:
        val_loader = DataLoader(
            val_set,
            batch_size=args['val_batch_size'],
            num_workers=args['num_workers'],
            shuffle=args['shuffle']
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
            det_net=det_net,
            criterion=criterion,
            epoch=epoch,
            writer=writer,
        )

        if need_val:
            val(
                val_loader=val_loader,
                device=device,
                det_net=det_net,
                criterion=criterion,
                epoch=epoch,
                writer=writer,
            )

        if save_net:
            net_path = os.path.join(
                save_path, 'net_parameters' + '.pth'
            )
            torch.save(det_net.state_dict(), net_path)

        # update learning rate for SGD
        if args['optim'] == 'SGD':
            scheduler.step()
    writer.close()


def train(
    train_loader,
    device,
    optimizer,
    det_net,
    criterion,
    epoch,
    writer,
):
    """
    training process of object detection
    :param train_loader: DataLoader
    :param device: torch.device
    :param optimizer: Adam
    :param det_net: Det_net
    :param criterion: DetLoss
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
            labels_train = data[1]
            labels_train_map = data[2]

            # reset the gradient of the optimizer
            optimizer.zero_grad()

            outputs_train = det_net(inputs_train)[0]

            # caululate the loss value with DetLoss
            loss_train = criterion(outputs_train, labels_train)

            train_epoch_loss += loss_train.detach().item()

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
                f'Train Epoch: {epoch + 1} |\
                     Loss: {loss_train.item()}'
            )
            epoch_bar_train.update(1)

        # write information at each end of the epoch
        img_denormalized = denormalize(inputs_train[0],  # convert the normalized image to original
                                       mean=[0.4499, 0.4365, 0.4364],
                                       std=[0.1902, 0.1972, 0.2085])

        # get the bounding box, class, probability scores from outputs_train
        bboxes, cls, scores = decode(outputs_train['cls'],
                                     outputs_train['wh'],
                                     outputs_train['reg'], 40)

        # draw prediction bounding box
        img_pred_bbox = draw_pred_bboxes(img_denormalized,
                                         bboxes[0], cls[0], scores[0])

        # draw ground truth bounding box
        img_gt_bbox = draw_gt_bboxes(img_denormalized,
                                     labels_train_map['bboxes'][0],
                                     labels_train_map['lbls'][0],
                                     labels_train_map['num_bboxes'][0])

        writer.add_scalar('Train Loss', train_epoch_loss/len(train_loader),
                          epoch)
        writer.add_image('Preds_train',
                         img_pred_bbox,
                         epoch,
                         dataformats='HWC')
        writer.add_image('Gt_train',
                         img_gt_bbox,
                         epoch,
                         dataformats='HWC')


def val(
    val_loader,
    device,
    det_net,
    criterion,
    epoch,
    writer,
):
    """
    validation process of object detection, compare with train loop, validation don't need optimizer
    and don't need write the loss value each iteration
    :param val_loader: DataLoader
    :param device: torch.device
    :param det_net: Det_Net
    :param criterion: DetLoss
    :param epoch: the current epoch number
    :param writer:
    :return:
    """
    with tqdm(total=len(val_loader)) as epoch_bar_val:
        # reset the loss value at the beginning of each epoch
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        with torch.no_grad():
            for i, data_val in enumerate(val_loader):
                # get the input and grundtruth label
                inputs_val = data_val[0].to(device)
                labels_val = data_val[1]
                labels_val_map = data_val[2]

                outputs_val = det_net(inputs_val)[0]

                # calculate the mean average precision and loss value
                bboxes, cls, scores = decode(outputs_val['cls'],
                                             outputs_val['wh'],
                                             outputs_val['reg'], 40)
                loss_val = criterion(outputs_val, labels_val)
                map = mean_average_precision(
                    (bboxes, cls, scores), labels_val_map)

                img_denormalized_val = denormalize(inputs_val[0],
                                                   mean=[0.4499, 0.4365, 0.4364],
                                                   std=[0.1902, 0.1972, 0.2085])

                img_pred_bbox_val = draw_pred_bboxes(img_denormalized_val,
                                                     bboxes[0], cls[0], scores[0])

                img_gt_bbox_val = draw_gt_bboxes(img_denormalized_val,
                                                 labels_val_map['bboxes'][0],
                                                 labels_val_map['lbls'][0],
                                                 labels_val_map['num_bboxes'][0])

                val_epoch_loss += loss_val.item()
                val_epoch_acc += map

                # write information at each end of the iteration
                writer_index_val = epoch * len(val_loader) + i

                writer.add_scalar(
                    '(fine)Val Loss',
                    loss_val.item(),
                    writer_index_val
                )
                writer.add_scalar(
                    '(fine)Val ACC',
                    map,
                    writer_index_val
                )

                # demonstrate the epoch bar
                epoch_bar_val.set_description(
                    f'Val Epoch: {epoch + 1} |\
                     Loss: {loss_val.item()}'
                )
                epoch_bar_val.update(1)

            # writer the information at each end of the epoch
            writer.add_image('Preds_val',
                             img_pred_bbox_val,
                             epoch,
                             dataformats='HWC')
            writer.add_image('Gt_val',
                             img_gt_bbox_val,
                             epoch,
                             dataformats='HWC')
            writer.add_scalar('Val Loss', val_epoch_loss/len(val_loader),
                              epoch)
            writer.add_scalar('Val Acc', val_epoch_acc/len(val_loader),
                              epoch)


if __name__ == '__main__':
    """Test train_sem.py
    """
    main()
