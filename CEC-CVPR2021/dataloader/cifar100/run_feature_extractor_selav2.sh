CUDA_VISIBLE_DEVICES=0 python FeatureExtraction_MocoV2.py --arch resnet50 --layer_names avgpool --weights /scratch/tahmad/self_supervised_models/ImageNet_based/selav2_400ep_pretrain.pth.tar --feature_file_name /scratch/tahmad/FSCIL_datasets/cifar100_selav2_ResNet50_features_imagenet/session_01/features_cifar100_train_01.npy --split train --N_feature 2048 --session_ID 01 --model_Type_input self --saved-with-data-parallel
