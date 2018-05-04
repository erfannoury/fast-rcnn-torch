# MaryamNet models

## Training MaryamNet run1 (5x5 Adam group=2) finetune all with BN
```bash
$ CUDA_VISIBLE_DEVICES=0 th main_train.lua \
    -pre_trained_file /nfs1/shared/for_erfan/final_results_evaluations_models/nseq5_group2_lr0.0001_lrdecay0_wd0.0005_adam_iter275000_net_cpu_firstbranch.t7 \
    -model_def ./models/Maryamnet/FRCNNMaryamNet.lua \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path ./cache/maryamnetlogs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path ./data/trained_models/nseq5_group2_lr0.0001_lrdecay0_wd0.0005_adam_iter275000_fullbn \
    -groups 2 -use_batchnorm -use_maryamnet_inputmaker
```

## Testing MaryamNet run1 (5x5 Adam group=2) finetune all with BN
```bash
$ CUDA_VISIBLE_DEVICES=0 th main_test.lua \
    -model_def ./models/Maryamnet/FRCNNMaryamNet.lua \
    -model_weights ./data/trained_models/nseq5_group2_lr0.0001_lrdecay0_wd0.0005_adam_iter275000_fullbn/frcnn_maryamnetbn_VOC2007_iter_80000.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path ./cache/maryamnetlogs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path ./data/trained_models/nseq5_group2_lr0.0001_lrdecay0_wd0.0005_adam_iter275000_fullbn \
    -groups 2 -use_batchnorm -use_maryamnet_inputmaker
```

# BSS (Alpha Blending) models

## Training BSS (12 classes, batchnorm in conv) finetune all with BN
```bash
$ CUDA_VISIBLE_DEVICES=0 th main_train.lua \
    -pre_trained_file /nfs1/shared/for_erfan/02_26_18/trained_models/main_bss2_imagenet_nClassesbss_12_iter100000_net_named.t7 \
    -model_def ./models/BSS/FRCNNBSS.lua \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 1 \
    -roi_per_img 128 -use_flipped \
    -log_path ./cache/bsslogs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path data/trained_models/bss2_nClasses12_fullbn \
    -groups 1 -use_batchnorm -use_maryamnet_inputmaker --freeze_batchnorm
```

## Testing BSS (12 classes, batchnorm in conv) finetune all with BN
```bash
$ CUDA_VISIBLE_DEVICES=0 th main_test.lua \
    -model_def ./models/Maryamnet/FRCNNMaryamNet.lua \
    -model_weights data/trained_models/bss2_nClasses12_fullbn/frcnn_bssbn_VOC2007_iter_80000.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path ./cache/bsslogs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path data/trained_models/bss2_nClasses12_fullbn \
    -groups 1 -use_batchnorm -use_maryamnet_inputmaker --freeze_batchnorm
```