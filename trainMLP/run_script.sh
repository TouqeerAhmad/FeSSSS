CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed concat --NUMBER_OF_FEATURE_FILES 60 --base_epochs 500 \
--root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_concat_0500.txt
rm -r CUB

CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed concat --NUMBER_OF_FEATURE_FILES 60 --base_epochs 1000 \
--root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_concat_1000.txt
rm -r CUB

CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed selfsupervised_only --NUMBER_OF_FEATURE_FILES 60 --base_epochs 500 \
--root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_self_supervised_0500.txt
rm -r CUB

CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed selfsupervised_only --NUMBER_OF_FEATURE_FILES 60 --base_epochs 1000 \
--root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_self_supervised_1000.txt
rm -r CUB

CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed supervised_only --NUMBER_OF_FEATURE_FILES 60 --base_epochs 500 \
--root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_supervised_0500.txt
rm -r CUB

CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed supervised_only --NUMBER_OF_FEATURE_FILES 60 --base_epochs 1000 \
--root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_supervised_1000.txt
rm -r CUB

CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed concat --NUMBER_OF_FEATURE_FILES 60 --base_epochs 500 --gnerateData \
--root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_concat_0500_gen_scalar.txt
rm -r CUB

CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed concat --NUMBER_OF_FEATURE_FILES 60 --base_epochs 1000 --gnerateData \
--root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_concat_1000_gen_scalar.txt
rm -r CUB

CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed concat --NUMBER_OF_FEATURE_FILES 60 --base_epochs 500 --gnerateData \
--useVectorVariance --root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_concat_0500_gen_vector.txt
rm -r CUB

CUDA_VISIBLE_DEVICES=0 python mlpTrain_Concat.py --dataSetInUse CUB --featureBeingUsed concat --NUMBER_OF_FEATURE_FILES 60 --base_epochs 1000 --gnerateData \
--useVectorVariance --root_path /scratch/tahmad/FSCIL_datasets/ --dirSelfSupervisedFeatures CUB_200_Mocov2_ResNet50_features_imagenet_800ep/ \
--dirSupervisedFeatures CUB_200_CEC_ResNet18_features_batch0/ > terminal_Incremental_Session_CUB.txt
cp terminal_Incremental_Session_CUB.txt logs_for_CUB/terminal_Incremental_Session_CUB_concat_1000_gen_vector.txt
rm -r CUB
