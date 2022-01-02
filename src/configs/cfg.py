args = {
    # file path for pc
    # 'root': '.',
    'cityscapes_dir': '/home/ubuntu/Data/dataset/cityscapes',
    'a2d2_dir': '/home/ubuntu/Data/dataset/a2d2/camera_lidar_semantic',
    'save_path': '/home/ubuntu/Documents/workplace/mlt_ml_praktikum_ss2020/save',
    'a2d2_label_path': '/home/ubuntu/Data/bbox_labels.json',

    # net parameters
    'net': 'resnet50',  # resnet18, 34, 50, 101, 152
    'det_num_classes': 9,
    'sem_num_classes': 3,
    'pretrained': True,
    'seg_loss_weight': [1, 2, 4],
    'det_loss_weight': [1, 0.1, 1],
    'total_loss_weight': [1, 1, 1],

    # dataset parameters
    'input_size': (448, 224),  # (320, 160), (448, 224),
    'transform': True,

    'train_batch_size': 8,
    'val_batch_size': 8,
    'num_workers': 4,
    'shuffle': True,
    'partial': 100,
    'train_extra': True,

    # train parameters
    'optim': 'Adam',  # SGD
    'epoch': 100,
    'lr': 0.0005,
    'momentum': 0.9,
    'step_size': 100,
    'gamma': 0.5,

    # MTL parameters
    'loss strategy': 'geometric loss strategy',  # handcrafted
                                                # geometric loss strategy
                                                # focused loss strategy
                                                # dynamic weight avg

    # load net.pth
    'net_multi': '/home/ubuntu/Documents/save/multi/net_parameters_epoch_13.pth',
    'net_sem': '/home/ubuntu/Documents/save/single_sem/net_parameters.pth',
    'net_det': '/home/ubuntu/Documents/save/single_det/net_parameters.pth',
    'net_depth': '/home/ubuntu/Documents/save/single_depth/net_parameters.pth',
}
