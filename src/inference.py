import torch
import time
from models.model import SemNet, DetNet, DepthUNet, ShareNet
from datasets.cityscapes import Cityscapes
from datasets.a2d2 import A2D2
from utils.utils import class_color_mapping, mkdir, draw_pred_bboxes, draw_gt_bboxes, apply_color_map, denormalize
from torch.utils.data import DataLoader
from configs.cfg import args
from models.heatmap_decode import decode
from torch.utils.tensorboard import SummaryWriter


def inference(number=3):
    """
    inference three samples from each A2D2 and Cityscapes. Using three single-task network and multi-task network.
    generate all these single network or multi network prediction images;
    generate groundtruth;
    generate original input images;
    all in tensorboard to browse
    :param number: number of the sample to generate images
    :return:
    """
    # create a folder for saving
    save_path = mkdir(args['save_path'])
    writer = SummaryWriter(save_path)

    # device setting
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')

    # set networks and load net parameters
    net_multi = ShareNet(args['net'],
                         sem_num_classes=args['sem_num_classes'],
                         det_num_classes=args['det_num_classes'],
                         pretrained=False).to(device).eval()
    net_sem = SemNet(args['net'],
                     sem_num_classes=args['sem_num_classes'],
                     pretrained=False).to(device).eval()
    net_det = DetNet(
        args['net'],
        det_num_classes=args['det_num_classes'],
        pretrained=False).to(device).eval()

    net_depth = DepthUNet(args['net']).to(device).eval()

    # load all four net.pth parameters
    try:
        net_multi.load_state_dict(torch.load(args['net_multi'])['model_state_dict'])
        net_sem.load_state_dict(torch.load(args['net_sem']))
        net_det.load_state_dict(torch.load(args['net_det']))
        net_depth.load_state_dict(torch.load(args['net_depth']))

    except:
        raise Exception("Please check the path of net.pth")

    # load A2D2 data
    data_set_a2d2 = A2D2(partial=number)
    loader_a2d2 = DataLoader(data_set_a2d2,
                             batch_size=1)

    # load Cityscapes data
    data_set_cs = Cityscapes(mode='val', task='multi_task', partial=number)
    loader_cs = DataLoader(data_set_cs,
                           batch_size=1)

    # initial time consume
    multi_time = 0.0
    sem_time = 0.0
    det_time = 0.0
    depth_time = 0.0

    # start inference loop
    for data_a2d2, data_cs in zip(iter(loader_a2d2), iter(loader_cs)):
        # A2D2
        input_a2d2 = data_a2d2[0].to(device)
        label_a2d2_sem = data_a2d2[1]
        label_a2d2_det = data_a2d2[3]

        # CS
        input_cs = data_cs[0].to(device)
        label_cs_sem = data_cs[1]
        label_cs_depth = data_cs[3]

        # test and mess time, multi-task network for A2D2
        torch.cuda.synchronize(device)
        start = time.time()
        output_a2d2_multi_sem, output_a2d2_multi_det, output_a2d2_multi_depth = net_multi(input_a2d2)
        end = time.time()
        torch.cuda.synchronize(device)
        multi_time += (end - start)

        # test and mess time, multi-task network for CS
        torch.cuda.synchronize(device)
        start = time.time()
        output_cs_multi_sem, output_cs_multi_det, output_cs_multi_depth = net_multi(input_cs)
        end = time.time()
        torch.cuda.synchronize(device)
        multi_time += (end - start)

        # write into tensorboard for multi task
        a2d2_image_denormalized = denormalize(input_a2d2[0],
                                              mean=[0.4499, 0.4365, 0.4364],
                                              std=[0.1902, 0.1972, 0.2085])
        cs_image_denormalized = denormalize(input_cs[0],
                                            mean=[0.3160, 0.3553, 0.3110],
                                            std=[0.1927, 0.1964, 0.1965])

        # groundtruth of a2d2 det
        label_gt_bbox = draw_gt_bboxes(a2d2_image_denormalized,
                                       label_a2d2_det['bboxes'][0],
                                       label_a2d2_det['lbls'][0],
                                       label_a2d2_det['num_bboxes'][0])

        # groundtruth of cs depth
        depth_map_cs_multi = apply_color_map(label_cs_depth[0])

        # write GT for A2D2 semantic segmentation
        writer.add_image('A2D2 GT SEM',
                         label_a2d2_sem[0].float() /
                         args['sem_num_classes'],
                         dataformats='HW')

        # write GT for A2D2 object detection
        writer.add_image('A2D2 GT DET',
                         label_gt_bbox,
                         dataformats='HWC')

        # write GT for CS semantic segmentation
        writer.add_image('CS GT SEM',
                         label_cs_sem[0].float() /
                         args['sem_num_classes'],
                         dataformats='HW')
        # write GT for CS depth estimation
        writer.add_image('CS GT DEPTH',
                         depth_map_cs_multi,
                         dataformats='HWC')

        # draw bounding boxes
        bboxes, cls, scores = decode(output_a2d2_multi_det['cls'],
                                     output_a2d2_multi_det['wh'],
                                     output_a2d2_multi_det['reg'], 40)

        a2d2_img_multi_bbox = draw_pred_bboxes(a2d2_image_denormalized,
                                               bboxes[0],
                                               cls[0],
                                               scores[0])
        bboxes, cls, scores = decode(output_cs_multi_det['cls'],
                                     output_cs_multi_det['wh'],
                                     output_cs_multi_det['reg'], 40)

        cs_img_multi_bbox = draw_pred_bboxes(cs_image_denormalized,
                                             bboxes[0],
                                             cls[0],
                                             scores[0])

        # apply a better color performance of depth estimation
        cs_depth_map_multi = apply_color_map(output_cs_multi_depth[0])
        a2d2_depth_map_multi = apply_color_map(output_a2d2_multi_depth[0])

        # write all information of Multi Task of A2D2 and Cityscapes
        writer.add_image('A2D2 image',
                         a2d2_image_denormalized)
        writer.add_image('A2D2-Multi SEM',
                         output_a2d2_multi_sem[0].argmax(0).float() /
                         args['sem_num_classes'],
                         dataformats='HW')
        writer.add_image('A2D2-Multi DET',
                         a2d2_img_multi_bbox,
                         dataformats='HWC')
        writer.add_image('A2D2-Multi DEPTH',
                         a2d2_depth_map_multi,
                         dataformats='HWC')

        writer.add_image('CS image',
                         cs_image_denormalized)
        writer.add_image('CS-Multi SEM',
                         output_cs_multi_sem[0].argmax(0).float() /
                         args['sem_num_classes'],
                         dataformats='HW')
        writer.add_image('CS-Multi DET',
                         cs_img_multi_bbox,
                         dataformats='HWC')
        writer.add_image('CS-Multi DEPTH',
                         cs_depth_map_multi,
                         dataformats='HWC')

        # Single Task: semantic segmentation
        # test and mess time, Semseg, A2D2
        torch.cuda.synchronize(device)
        start = time.time()
        output_single_sem = net_sem(input_a2d2)[0]
        torch.cuda.synchronize(device)
        end = time.time()
        sem_time += (end - start)

        writer.add_image('A2D2-Single sem',
                         output_single_sem[0].argmax(0).float() /
                         args['sem_num_classes'],
                         dataformats='HW')

        # test and mess time, Semseg, CS
        torch.cuda.synchronize(device)
        start = time.time()
        output_single_sem = net_sem(input_cs)[0]
        torch.cuda.synchronize(device)
        end = time.time()
        sem_time += (end - start)

        writer.add_image('CS-Single sem',
                         output_single_sem[0].argmax(0).float() /
                         args['sem_num_classes'],
                         dataformats='HW')

        # Single Task: object detection
        # test and mess for detection, A2D2
        torch.cuda.synchronize(device)
        start = time.time()
        output_single_det = net_det(input_a2d2)[0]
        torch.cuda.synchronize(device)
        end = time.time()
        det_time += (end - start)

        bboxes, cls, scores = decode(output_single_det['cls'],
                                     output_single_det['wh'],
                                     output_single_det['reg'], 40)

        img_single_bbox = draw_pred_bboxes(a2d2_image_denormalized,
                                           bboxes[0],
                                           cls[0],
                                           scores[0])
        writer.add_image('A2D2-Single det',
                         img_single_bbox,
                         dataformats='HWC')

        # test and mess time for detection, CS
        torch.cuda.synchronize(device)
        start = time.time()
        output_single_det = net_det(input_cs)[0]
        torch.cuda.synchronize(device)
        end = time.time()
        det_time += (end - start)

        bboxes, cls, scores = decode(output_single_det['cls'],
                                     output_single_det['wh'],
                                     output_single_det['reg'], 40)

        img_single_bbox = draw_pred_bboxes(cs_image_denormalized,
                                           bboxes[0],
                                           cls[0],
                                           scores[0])
        writer.add_image('CS-Single det',
                         img_single_bbox,
                         dataformats='HWC')

        # Single Task: depth estimation
        # test and mess time for depth estimation of A2D2
        torch.cuda.synchronize(device)
        start = time.time()
        output_single_depth = net_depth(input_a2d2)
        torch.cuda.synchronize(device)
        end = time.time()
        depth_time += (end - start)

        output_single_depth = output_single_depth[0].squeeze()
        depth_map_single = apply_color_map(output_single_depth)
        writer.add_image("A2D2-Single depth",
                         depth_map_single,
                         dataformats='HWC')

        # test and mess time of CS
        torch.cuda.synchronize(device)
        start = time.time()
        output_single_depth = net_depth(input_cs)
        torch.cuda.synchronize(device)
        end = time.time()
        depth_time += (end - start)

        output_single_depth = output_single_depth[0].squeeze()
        depth_map_single = apply_color_map(output_single_depth)
        writer.add_image("CS-Single depth",
                         depth_map_single,
                         dataformats='HWC')

    print("multi time: ", multi_time / number / 2)
    print("fps: ", 1 // (multi_time / number / 2))
    print("sem time: ", sem_time / number / 2)
    print("fps: ", 1 // (sem_time / number / 2))
    print("det time: ", det_time / number / 2)
    print("fps: ", 1 // (det_time / number / 2))
    print("depth time", depth_time / number / 2)
    print("fps: ", 1 // (depth_time / number / 2))

    writer.close()


if __name__ == '__main__':
    # Task 1: inference results and speed testing of 3 single-task network, 1 multi-task network
    inference(number=3)

    # Task 2: test the GPU memory usage of each network
    # 1. comment the inference(number=3) above
    # 2. cancel the comment of one net, which is to test the GPU usage
    # net = ShareNet(args['net'], sem_num_classes=args['sem_num_classes'], det_num_classes=args['det_num_classes'],
    #                pretrained=False)
    # net = SemNet(args['net'], sem_num_classes=args['sem_num_classes'], pretrained=False)
    # net = DetNet(args['net'], det_num_classes=args['det_num_classes'], pretrained=False)
    # net = DepthUNet(args['net'])

    # 3. cancel the comment of following for Task 2.
    # device = torch.device('cuda')
    # net.to(device)
    # input()  # This is for a pause of the running, to see the GPU usage
