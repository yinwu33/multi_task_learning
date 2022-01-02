import os
import cv2
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import ShareNet
from models.losses import DetLoss, ScaleInvariantLogLoss, L1MaskedLoss, dynamic_weight_average, dynamic_weight_average_loss_weights
from models.depth_decode import discretization_decode
from models.heatmap_decode import decode
from datasets.a2d2 import A2D2
from datasets.cityscapes import Cityscapes
from datasets.merge_datasets import MergeDataset, BatchSchedulerSampler
from torch.utils.tensorboard import SummaryWriter
from configs.cfg import args
from utils.utils import denormalize, draw_pred_bboxes, draw_gt_bboxes, mean_average_precision, apply_color_map, mkdir


def main(args=args, need_val=True, early_stop=False, resume_train=False, save_epochwise=True, save_net=True):
    assert need_val == early_stop or (need_val is True and
                                      early_stop is False), 'early stop error'
    PATH = mkdir(args["save_path"])
    # # create a log writer
    writer = SummaryWriter(PATH)

    # set device cuda or cpu
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # set network
    net = ShareNet(args['net'],
                     sem_num_classes=args['sem_num_classes'],
                     det_num_classes=args['det_num_classes'],
                     pretrained=args['pretrained'])

    # set optimizer
    optimizer = optim.Adam(net.parameters(),
                           lr=args['lr'])

    if resume_train:
        net_path = "/root/project/save/2020-09-05-15-22-10/net_parameters_epoch_0.pth"
        checkpoint = torch.load(net_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    net.to(device).train()


    # set learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                       step_size=args['step_size'],
    #                                       gamma=args['gamma'])

    # set loss function
    sem_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(args['seg_loss_weight']).to(device))
    det_criterion = DetLoss(weights=args['det_loss_weight'])
    depth_criterion = ScaleInvariantLogLoss()

    # a2d2 dataset
    data_set = A2D2(partial=args['partial'])
    train_data_len = int(0.9832 * len(data_set))
    val_data_len = int(0.0168 * len(data_set))

    train_set_a2d2, val_set_a2d2 = torch.utils.data.random_split(data_set, [train_data_len, val_data_len + 1])

    train_set_cs = Cityscapes(mode='train', task='depth', train_extra=args['train_extra'])
    val_set_cs = Cityscapes(mode='val', task='depth', train_extra=False)

    train_loader_a2d2 = DataLoader(train_set_a2d2,
                                   batch_size=args['train_batch_size'],
                                   num_workers=args['num_workers'],
                                   shuffle=args['shuffle'])

    train_loader_cs = DataLoader(train_set_cs,
                                   batch_size=args['train_batch_size'],
                                   num_workers=args['num_workers'],
                                   shuffle=args['shuffle'])

    val_loader_a2d2 = DataLoader(val_set_a2d2,
                                   batch_size=args['val_batch_size'],
                                   num_workers=args['num_workers'],
                                   shuffle=args['shuffle'])

    val_loader_cs = DataLoader(val_set_cs,
                                 batch_size=args['val_batch_size'],
                                 num_workers=args['num_workers'],
                                 shuffle=args['shuffle'])

    # train_set = MergeDataset([train_set_a2d2, train_set_cs])
    # val_set = MergeDataset([val_set_a2d2, val_set_cs])
    #
    # train_sampler = BatchSchedulerSampler(dataset=train_set, batch_size=args['train_batch_size'] // 2)
    # val_sampler = BatchSchedulerSampler(dataset=train_set, batch_size=args['train_batch_size'] // 2)

    # # sampler option is mutually exclusive with shuffle
    # train_loader = DataLoader(train_set,
    #                           sampler=train_sampler,
    #                           batch_size=args['train_batch_size'] // 2,
    #                           num_workers=args['num_workers'])

    # early stop criterion
    # accuracy_val_pre = 0.0
    #
    # def stop_criterion(accuracy_val_pre, accuracy_val, stop_rate=0.05):
    #     return True if (accuracy_val_pre - accuracy_val) / accuracy_val > stop_rate \
    #         else False

    # writer net parameters
    # writer.add_text(
    #     'arg', 'lr: ' + str(args['lr']) + '    |    batch size: ' +
    #            str(args['train_batch_size']) + '    |    net: ' + args['net'])

    # train loop
    for key, value in args.items():
        writer.add_text(str(key), str(value))

    writer_index_train = 0
    writer_index_val = 0

    for epoch in range(args['epoch']):
        train(net, epoch, train_loader_cs, train_loader_a2d2,
              sem_criterion, det_criterion, depth_criterion,
              optimizer, writer, writer_index_train, device)

        writer_index_train += 1

        # save the network by each epoch

        # validation loop
        if need_val:
            val(net, epoch, val_loader_cs, val_loader_a2d2,
            sem_criterion, det_criterion, depth_criterion,
            writer, writer_index_val, device)
            writer_index_val += 1

        if save_epochwise and save_net:
            net_path = os.path.join(
                PATH, 'net_parameters_epoch_' + str(epoch) + '.pth')
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, net_path)

    # save the net at end of the train
    if save_net:
        net_path = os.path.join(PATH, 'net_parameters_final' + '.pth')
        torch.save(net.state_dict(), net_path)

    writer.close()


def train(net, epoch, train_loader_cs, train_loader_a2d2,
          sem_criterion, det_criterion, depth_criterion,
          optimizer, writer, writer_index_train, device):
    train_epoch_loss = 0.0
    train_epoch_loss_sem = 0.0
    train_epoch_loss_det = 0.0
    train_epoch_loss_depth = 0.0

    len_train_loader = len(train_loader_cs)
    with tqdm(total=len_train_loader) as epoch_bar_train:
        for i, data1 in enumerate(iter(train_loader_cs)):
            # two small-batches at a time, so that each batch contains data from both datasets
            data2 = next(iter(train_loader_a2d2))
            optimizer.zero_grad()

            inputs1 = data1[0].to(device)
            inputs2 = data2[0].to(device)

            labels_sem = data2[1].to(device)
            labels_det = data2[2]
            labels_depth = data1[1].to(device)

            outputs_sem_a2d2 = net(inputs2)[0]
            outputs_det_a2d2 = net(inputs2)[1]
            outputs_depth_cs = net(inputs1)[2]

            # we don't use those preds for loss calculation, because there are no grundtruth labels
            outputs_sem_cs = net(inputs1)[0]
            outputs_det_cs = net(inputs1)[1]
            outputs_depth_a2d2 = net(inputs2)[2]

            loss_sem = sem_criterion(outputs_sem_a2d2, labels_sem) * 5
            loss_det = det_criterion(outputs_det_a2d2, labels_det)
            loss_depth = depth_criterion(outputs_depth_cs, labels_depth)

            # if i != 0 and args['loss strategy'] == 'dynamic weight averaging':
            #     weights_slope = dynamic_weight_average([loss_sem, loss_det, loss_depth], prev_losses,
            #                                            num_task=3)
            #
            #     loss_weights = dynamic_weight_average_loss_weights(weights_slope, num_task=3, epochs=1)

            if args['loss strategy'] == 'handcrafted':
                loss_train = loss_sem * args['total_loss_weight'][0] \
                             + loss_det * args['total_loss_weight'][1] \
                             + loss_depth * args['total_loss_weight'][2]

            if args['loss strategy'] == 'geometric loss strategy':
                loss_train = torch.pow(loss_sem * loss_det * loss_depth, 1 / 3)

            if args['loss strategy'] == 'focused loss strategy':
                loss_train = torch.pow(loss_sem * loss_det * loss_depth, 1 / 3) + \
                             torch.pow(loss_sem * loss_depth, 1 / 2)

            # if args['loss strategy'] == 'dynamic weight avg':
            #     try:
            #         loss_train = loss_sem * loss_weights[0] \
            #                      + loss_det * loss_weights[1] \
            #                      + loss_depth * loss_weights[2]
            #     except:
            #         loss_train = loss_sem * args['total_loss_weight'][0] \
            #                      + loss_det * args['total_loss_weight'][1] \
            #                      + loss_depth * args['total_loss_weight'][2]

            train_epoch_loss += loss_train.item()
            train_epoch_loss_sem += loss_sem.item()
            train_epoch_loss_det += loss_det.item()
            train_epoch_loss_depth += loss_depth.item()


            # TODO acc

            # demonstrate the epoch bar
            epoch_bar_train.set_description(
                f'Train Epoch: {epoch + 1} | Loss: {loss_train.item()}'
            )
            epoch_bar_train.update(1)

            # calculate gradient of loss
            loss_train.backward()
            optimizer.step()
            # update learning rate
            # write logger
            writer_index_train = epoch * len_train_loader + i
            # training loss in each iteration
            writer.add_scalar('(fine)Train Loss', loss_train.item(),
                          writer_index_train)
            writer.add_scalar('(fine)Train Loss Sem', loss_sem.item(),
                          writer_index_train)
            writer.add_scalar('(fine)Train Loss Det', loss_det.item(),
                          writer_index_train)
            writer.add_scalar('(fine)Train Loss Depth', loss_depth.item(),
                          writer_index_train)

        img_denormalized_a2d2 = denormalize(inputs2[0],
                                            mean=[0.4499, 0.4365, 0.4364],
                                            std=[0.1902, 0.1972, 0.2085])

        bboxes_a2d2, cls_a2d2, scores_a2d2 = decode(outputs_det_a2d2['cls'],
                                                    outputs_det_a2d2['wh'],
                                                    outputs_det_a2d2['reg'], 40)

        img_pred_bbox_a2d2 = draw_pred_bboxes(img_denormalized_a2d2,
                                              bboxes_a2d2[0], cls_a2d2[0], scores_a2d2[0])

        img_denormalized_cs = denormalize(inputs1[0],
                                          mean=[0.3160, 0.3553, 0.3110],
                                          std=[0.1927, 0.1964, 0.1965])

        bboxes_cs, cls_cs, scores_cs = decode(outputs_det_cs['cls'],
                                              outputs_det_cs['wh'],
                                              outputs_det_cs['reg'], 40)

        img_pred_bbox_cs = draw_pred_bboxes(img_denormalized_cs,
                                            bboxes_cs[0], cls_cs[0], scores_cs[0])

        # img_gt_bbox = draw_gt_bboxes(img_denormalized,
        #                              labels_train_map['bboxes'][0],
        #                              labels_train_map['lbls'][0],
        #                              labels_train_map['num_bboxes'][0])

        depth_map_a2d2 = apply_color_map(outputs_depth_a2d2[0])

        depth_map_cs = apply_color_map(outputs_depth_cs[0])

        writer.add_scalar('Train Loss', train_epoch_loss / len_train_loader,
                          epoch)
        writer.add_scalar('Train Loss Sem', train_epoch_loss_sem / len_train_loader,
                          epoch)
        writer.add_scalar('Train Loss Det', train_epoch_loss_det / len_train_loader,
                          epoch)
        writer.add_scalar('Train Loss Depth', train_epoch_loss_depth / len_train_loader,
                          epoch)

        writer.add_image('Preds Train Sem A2D2',
                         outputs_sem_a2d2[0].argmax(0).float() /
                         args['sem_num_classes'],
                         epoch,
                         dataformats='HW')
        writer.add_image('Preds Train Det A2D2',
                         img_pred_bbox_a2d2,
                         epoch,
                         dataformats='HWC')
        writer.add_image('Preds Train Depth A2D2',
                         depth_map_a2d2,
                         epoch,
                         dataformats='HWC')
        writer.add_image('Preds Train Sem CS',
                         outputs_sem_cs[0].argmax(0).float() /
                         args['sem_num_classes'],
                         epoch,
                         dataformats='HW')
        writer.add_image('Preds Train Det CS',
                         img_pred_bbox_cs,
                         epoch,
                         dataformats='HWC')
        writer.add_image('Preds Train Depth CS',
                         depth_map_cs,
                         epoch,
                         dataformats='HWC')

        # writer.add_image('Gt_train',
        #                  img_gt_bbox,
        #                  writer_index_train,
        #                  dataformats='HWC')


def val(net, epoch, val_loader_cs, val_loader_a2d2,
        sem_criterion, det_criterion, depth_criterion,
        writer, writer_index_val, device):
    val_epoch_loss = 0.0
    val_epoch_loss_sem = 0.0
    val_epoch_loss_det = 0.0
    val_epoch_loss_depth = 0.0

    len_val_loader = len(val_loader_cs)
    with tqdm(total=len_val_loader) as epoch_bar_val:
        
        for i, data1 in enumerate(iter(val_loader_cs)):
        # data1 = next(iter(val_loader_cs))
            data2 = next(iter(val_loader_a2d2))
            with torch.no_grad():
                inputs1 = data1[0].to(device)
                inputs2 = data2[0].to(device)

                labels_sem = data2[1].to(device)
                labels_det = data2[2]
                labels_depth = data1[1].to(device)

                outputs_sem_a2d2 = net(inputs2)[0]
                outputs_det_a2d2 = net(inputs2)[1]
                outputs_depth_cs = net(inputs1)[2]

                # we don't use those preds for loss calculation, because there are no grundtruth labels
                outputs_sem_cs = net(inputs1)[0]
                outputs_det_cs = net(inputs1)[1]
                outputs_depth_a2d2 = net(inputs2)[2]

                loss_sem = sem_criterion(outputs_sem_a2d2, labels_sem) * 5
                loss_det = det_criterion(outputs_det_a2d2, labels_det)
                loss_depth = depth_criterion(outputs_depth_cs, labels_depth)

                if args['loss strategy'] == 'handcrafted':
                    loss_val = loss_sem * args['total_loss_weight'][0] \
                            + loss_det * args['total_loss_weight'][1] \
                            + loss_depth * args['total_loss_weight'][2]

                if args['loss strategy'] == 'geometric loss strategy':
                    loss_val = torch.pow(loss_sem * loss_det * loss_depth, 1 / 3)

                if args['loss strategy'] == 'focused loss strategy':
                    loss_val = torch.pow(loss_sem * loss_det * loss_depth, 1 / 3) + \
                               torch.pow(loss_sem * loss_depth, 1 / 2)

                val_epoch_loss += loss_val.item()
                val_epoch_loss_sem += loss_sem.item()
                val_epoch_loss_det += loss_det.item()
                val_epoch_loss_depth += loss_depth.item()

            # demonstrate the epoch bar
            epoch_bar_val.set_description(
                f'Val Epoch: {epoch + 1} | Loss: {loss_val.item()}'
            )
            epoch_bar_val.update(1)

        img_denormalized_a2d2 = denormalize(inputs2[0],
                                            mean=[0.4499, 0.4365, 0.4364],
                                            std=[0.1902, 0.1972, 0.2085])

        bboxes, cls, scores = decode(outputs_det_a2d2['cls'],
                                    outputs_det_a2d2['wh'],
                                    outputs_det_a2d2['reg'], 40)

        img_pred_bbox_a2d2 = draw_pred_bboxes(img_denormalized_a2d2,
                                            bboxes[0], cls[0], scores[0])

        img_denormalized_cs = denormalize(inputs1[0],
                                        mean=[0.3160, 0.3553, 0.3110],
                                        std=[0.1927, 0.1964, 0.1965])

        bboxes, cls, scores = decode(outputs_det_cs['cls'],
                                    outputs_det_cs['wh'],
                                    outputs_det_cs['reg'], 40)

        img_pred_bbox_cs = draw_pred_bboxes(img_denormalized_cs,
                                            bboxes[0], cls[0], scores[0])

        # img_gt_bbox = draw_gt_bboxes(img_denormalized,
        #                              labels_train_map['bboxes'][0],
        #                              labels_train_map['lbls'][0],
        #                              labels_train_map['num_bboxes'][0])

        depth_map_a2d2 = apply_color_map(outputs_depth_a2d2[0])

        depth_map_cs = apply_color_map(outputs_depth_cs[0])

        writer.add_scalar('Val Loss', val_epoch_loss / len_val_loader,
                        epoch)
        writer.add_scalar('Val Loss Sem', val_epoch_loss_sem / len_val_loader,
                        epoch)
        writer.add_scalar('Val Loss Det', val_epoch_loss_det / len_val_loader,
                        epoch)
        writer.add_scalar('Val Loss Depth', val_epoch_loss_depth / len_val_loader,
                        epoch)

        writer.add_image('Preds Val Sem A2D2',
                        outputs_sem_a2d2[0].argmax(0).float() /
                        args['sem_num_classes'],
                        epoch,
                        dataformats='HW')
        writer.add_image('Preds Val Det A2D2',
                        img_pred_bbox_a2d2,
                        epoch,
                        dataformats='HWC')
        writer.add_image('Preds Val Depth A2D2',
                        depth_map_a2d2,
                        epoch,
                        dataformats='HWC')
        writer.add_image('Preds Val Sem CS',
                        outputs_sem_cs[0].argmax(0).float() /
                        args['sem_num_classes'],
                        epoch,
                        dataformats='HW')
        writer.add_image('Preds Val Det CS',
                        img_pred_bbox_cs,
                        epoch,
                        dataformats='HWC')
        writer.add_image('Preds Val Depth CS',
                        depth_map_cs,
                        epoch,
                        dataformats='HWC')


if __name__ == '__main__':
    main()