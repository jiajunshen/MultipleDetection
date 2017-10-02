#!/bin/bash

python cifar10_train_epoch.py $1 && python cifar10_train_epoch_saliency_full.py $1 && python produce_mask_full.py /hdd/Documents/Data/ShapeNetCoreV2/cifar10_3class_test.npy \$SHAPENET/cifar10_test_mask_$1.npy \$SHAPENET/cifar10_test_decluttered_$1.npy && python train_traditional_cnn.py $1 \$SHAPENET/cifar10_test_decluttered_$1.npy
