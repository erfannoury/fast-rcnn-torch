# Training MaryamNet run1 (5x5 Adam group=2) finetune all with BN
```bash
$ CUDA_VISIBLE_DEVICES=0 th main_train.lua \
    -pre_trained_file /nfs1/shared/for_erfan/final_results_evaluations_models/nseq5_group2_lr0.0001_lrdecay0_wd0.0005_adam_iter275000_net_cpu_firstbranch.t7 \
    -model_def ./models/Maryamnet/FRCNNMaryamNet.lua \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -pixel_means {-0.083300798050439, -0.10651495109198, -0.17295466315224} \
    -log_path ./cache/maryamnetlogs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -optim_regimes {{30000, 1e-3, 5e-4}, {30000, 1e-4, 5e-4}, {20000, 5e-5, 5e-5}} \
    -save_path ./data/trained_models/nseq5_group2_lr0.0001_lrdecay0_wd0.0005_adam_iter275000_fullbn \
    -groups 2 -use_batchnorm -use_maryamnet_inputmaker
```

# Testing MaryamNet run1 (5x5 Adam group=2) finetune all with BN
```bash
$ CUDA_VISIBLE_DEVICES=0 th main_test.lua \
    -model_def ./models/Maryamnet/FRCNNMaryamNet.lua \
    -model_weights ./data/trained_models/nseq5_group2_lr0.0001_lrdecay0_wd0.0005_adam_iter275000_fullbn/frcnn_maryamnetbn_VOC2007_iter_80000.t7 \
    -use_difficult_objs -scale 600 -max_size 1000 -img_per_batch 2 \
    -roi_per_img 128 -use_flipped \
    -pixel_means {-0.083300798050439, -0.10651495109198, -0.17295466315224} \
    -log_path ./cache/maryamnetlogs/ -dataset voc_2007\
    -dataset_path /nfs1/datasets/PASCAL/VOCdevkit \
    -optim_regimes {{30000, 1e-3, 5e-4}, {30000, 1e-4, 5e-4}, {20000, 5e-5, 5e-5}} \
    -save_path ./data/trained_models/nseq5_group2_lr0.0001_lrdecay0_wd0.0005_adam_iter275000_fullbn \
    -groups 2 -use_batchnorm -use_maryamnet_inputmaker
```