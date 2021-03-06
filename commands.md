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

## Training MaryamNet run30 (2x2 Adam group=2) finetune all without BN
```bash
$ CUDA_VISIBLE_DEVICES=0 th main_train.lua \
    -pre_trained_file /nfs1/shared/for_erfan/final_results_evaluations_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_net_cpu.t7 \
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000 \
    -groups 2 -use_maryamnet255_inputmaker
```

## Testing MaryamNet run30 (2x2 Adam group=2) finetune all without BN
```bash
$ CUDA_VISIBLE_DEVICES=0 th main_test.lua \
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -model_weights /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000/frcnn_maryamnet_VOC2007_iter_80000.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000 \
    -groups 2 -use_maryamnet255_inputmaker
```

## Training MaryamNet run30 (2x2 Adam group=1) finetune all without BN, identity convbn
```bash
$ CUDA_VISIBLE_DEVICES=2 th main_train.lua \
    -pre_trained_file /nfs1/shared/for_erfan/final_results_evaluations_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_net_cpu.t7 \
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_identityconvbn_fixedbackened \
    -groups 1 -use_maryamnet255_inputmaker -use_identity_convbn -use_batchnorm
```

## Training MaryamNet run30 (2x2 Adam group=1) finetune all without BN, with convbn warmup
```bash
$ CUDA_VISIBLE_DEVICES='' th main_train.lua \
    -pre_trained_file /nfs1/shared/for_erfan/final_results_evaluations_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_net_cpu.t7 \
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_warmupconvbn \
    -groups 1 -use_maryamnet255_inputmaker -do_identity_convbn_warmup -use_batchnorm
```

## Testing MaryamNet run30 (2x2 Adam group=1) finetune all without BN, identity convbn
```bash
$ CUDA_VISIBLE_DEVICES=5 th main_test.lua \
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -model_weights /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_identityconvbn_fixedbackened/frcnn_maryamnetbn_VOC2007_iter_80000_05.12_11.00.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_identityconvbn_fixedbackened \
    -groups 1 -use_maryamnet255_inputmaker -use_identity_convbn -use_batchnorm
```
## Testing MaryamNet run30 (2x2 Adam group=1) finetune all without BN, with convbn warmup
```bash
$ CUDA_VISIBLE_DEVICES=5 th main_test.lua \
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -model_weights /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_warmupconvbn/frcnn_maryamnetbn_VOC2007_iter_80000_05.12_05.34_named.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_warmupconvbn \
    -groups 1 -use_maryamnet255_inputmaker  -use_batchnorm
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
    -groups 1 -use_batchnorm -use_maryamnet_inputmaker -use_flipped
```

## Testing BSS (12 classes, batchnorm in conv) finetune all with BN
```bash
$ CUDA_VISIBLE_DEVICES=2 th main_test.lua \
    -model_def models/BSS/FRCNNBSS.lua \
    -model_weights /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/bss2_nClasses12_fullbn/frcnn_bssbn_VOC2007_iter_80000_05.04_02.08.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path cache/bsslogs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/bss2_nClasses12_fullbn/ \
    -groups 1 -use_batchnorm -use_maryamnet_inputmaker -use_flipped
```


## Training BSS (12 classes, batchnorm in conv) finetune all with BN, Freeze BN
```bash
$ CUDA_VISIBLE_DEVICES=5 th main_train.lua \
    -pre_trained_file /nfs1/shared/for_erfan/02_26_18/trained_models/main_bss2_imagenet_nClassesbss_12_iter100000_net_named.t7 \
    -model_def ./models/BSS/FRCNNBSS.lua \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 1 \
    -roi_per_img 128 -use_flipped \
    -log_path ./cache/bsslogs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path data/trained_models/bss2_nClasses12_fullbn_freezebn \
    -groups 1 -use_batchnorm -use_maryamnet_inputmaker -use_flipped -freeze_batchnorm
```

## Testing BSS (12 classes, batchnorm in conv) finetune all with BN, Freeze BN
```bash
$ CUDA_VISIBLE_DEVICES=5 th main_test.lua \
    -model_def ./models/BSS/FRCNNBSS.lua \
    -model_weights /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/bss2_nClasses12_fullbn_freezebn/frcnn_bssbn_VOC2007_iter_80000_05.04_04.34.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path ./cache/bsslogs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/bss2_nClasses12_fullbn_freezebn \
    -groups 1 -use_batchnorm -use_maryamnet_inputmaker -use_flipped -freeze_batchnorm

```

## Training MaryamNet run45 (5x5 Adam group=1) finetune all without BN, identity convbn
```bash
$ CUDA_VISIBLE_DEVICES=5 th main_train.lua \
    -pre_trained_file /nfs1/code/maryam/final_experiments_results/nseq5_lr0.001_lrdecay100000_wd0.0005_adam_label24_siamese_255_BNfalse/nseq5_lr0.001_lrdecay100000_wd0.0005_adam_label24_siamese_255_BNfalse_iter225000_net.t7\
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq5_lr0.001_lrdecay100000_wd0.0005_adam_label24_siamese_255_BNfalse_iter225000_identityconvbn_fixedbackened \
    -groups 1 -use_maryamnet255_inputmaker -use_identity_convbn -use_batchnorm
```

## Training MaryamNet run45 (5x5 Adam group=1) finetune all without BN, with convbn warmup
```bash
$ CUDA_VISIBLE_DEVICES=4 th main_train.lua \
    -pre_trained_file /nfs1/code/maryam/final_experiments_results/nseq5_lr0.001_lrdecay100000_wd0.0005_adam_label24_siamese_255_BNfalse/nseq5_lr0.001_lrdecay100000_wd0.0005_adam_label24_siamese_255_BNfalse_iter225000_net.t7 \
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq5_lr0.001_lrdecay100000_wd0.0005_adam_label24_siamese_255_BNfalse_iter225000_warmupconvbn \
    -groups 1 -use_maryamnet255_inputmaker -do_identity_convbn_warmup -use_batchnorm
```
## Testing MaryamNet run45 (5x5 Adam group=1) finetune all without BN, identity convbn
```bash
$ CUDA_VISIBLE_DEVICES=5 th main_test.lua \
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -model_weights /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_identityconvbn_fixedbackened/frcnn_maryamnetbn_VOC2007_iter_80000_05.12_11.00.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq5_lr0.001_lrdecay100000_wd0.0005_adam_label24_siamese_255_BNfalse_iter225000_identityconvbn_fixedbackened \
    -groups 1 -use_maryamnet255_inputmaker -use_identity_convbn -use_batchnorm
```
## Testing MaryamNet run30 (2x2 Adam group=1) finetune all without BN, with convbn warmup
```bash
$ CUDA_VISIBLE_DEVICES=5 th main_test.lua \
    -model_def /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/models/Maryamnet/FRCNNMaryamNet.lua \
    -model_weights /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq2_group2_lr0.0001_lrdecay0_wd0.0005_adam_siamese_255_BN_jittering_iter75000_warmupconvbn/frcnn_maryamnetbn_VOC2007_iter_80000_05.12_05.34_named.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -log_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/cache/maryamnet255logs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -save_path /nfs1/code/maryam/Erfan_frcnn/fast-rcnn-torch/data/trained_models/nseq5_lr0.001_lrdecay100000_wd0.0005_adam_label24_siamese_255_BNfalse_iter225000_warmupconvbn \
    -groups 1 -use_maryamnet255_inputmaker  -use_batchnorm
```
