--- All parameters goes here
config = config or {}

function config.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text("Fast R-CNN for Torch")
    cmd:text()
    cmd:text("")
    -- Parameters
    cmd:option(
        "-resume_training",
        false,
        "True if you are resuming the training from a FRCNN model or false when starting from the imagenet model"
    )
    cmd:option(
        "-pre_trained_file",
        "./data/torch_imagenet_models/imgnet_alexnet.t7",
        "Path to the pretrained weights (used for training)"
    )
    cmd:option(
        "-model_def",
        "./models/AlexNet/FRCNN.lua",
        "Path to the FRCNN model definition"
    )
    cmd:option(
        "-model_weights",
        "./data/trained_models/frcnn_alexnet_VOC2007_iter_40000.t7",
        "Path to the FRCNN weights (used for testing)"
    )
    cmd:option(
        "-use_difficult_objs",
        true,
        "Whether to load the difficult examples or not"
    )
    cmd:option(
        "-scale",
        600,
        "Scale used for training and testing, currently only single scale is supported."
    )
    cmd:option(
        "-max_size",
        1000,
        "Max pixel size of the longest side of a scaled input image"
    )
    cmd:option(
        "-img_per_batch",
        2,
        "Images per minibatch"
    )
    cmd:option(
        "-GPU_ID",
        1,
        "Main GPU ID to be used"
    )
    cmd:option(
        "-n_threads",
        1,
        "Number of threads used for training (In Multi GPU mode)"
    )
    cmd:option(
        "-nGPU",
        1,
        "Number of GPUs to be used for training (not completely tested yet)"
    )
    cmd:option(
        "-roi_per_img",
        64,
        "Minibatch size"
    )
    cmd:option(
        "-fg_fraction",
        0.25,
        "Fraction of the minibatch that is labeled as foreground (i.e. class > 0)"
    )
    cmd:option(
        "-fg_threshold",
        0.5,
        "IoU threshold for a ROI to be considered as foreground (if >= FG_THRESH)"
    )
    cmd:option(
        "-bg_threshold_hi",
        0.5,
        "High IoU threshold for a ROI to be considered as background"
    )
    cmd:option(
        "-bg_threshold_lo",
        0.1,
        "Low IoU threshold for a ROI to be considered as background"
    )
    cmd:option(
        "-use_flipped",
        false,
        "Use horizontally-flipped images during training if true"
    )
    cmd:option(
        "-bbox_threshold",
        0.5,
        "IoU required between a ROI and a ground-truth box in order for that ROI to be used as a bounding-box regression training example"
    )
    cmd:option(
        "-nms",
        0.3,
        "Overlap threshold used for non-maxima suppression (suppress boxes with IoU >= this value)"
    )
    cmd:option(
        "-pixel_means",
        {-0.083300798050439, -0.10651495109198, -0.17295466315224},
        "Pixel mean values (BGR order)"
    )
    cmd:option(
        "-eps",
        1e-14,
        "Epsilon"
    )
    cmd:option(
        "-log_path",
        "./cache",
        "Path used for saving log data"
    )
    cmd:option(
        "-dataset",
        "voc_2007",
        "Dataset to be used"
    )
    cmd:option(
        "-dataset_path",
        "/nfs1/datasets/PASCAL/VOCdevkit/",
        "Path to the dataset root folder"
    )
    cmd:option(
        "-test_img_set",
        "test",
        "Image set to be used for testing"
    )
    cmd:option(
        "-train_img_set",
        "trainval",
        "Image set to be used for training"
    )
    cmd:option(
        "-cache",
        "./cache",
        "Directory used for saving cache data"
    )
    cmd:option(
        "-optim_momentum",
        0.9,
        "Momentum used for sgd optimizer"
    )
    cmd:option(
        "-optim_lr_decay_policy",
        "fixed",
        "Learning rate decay policy, can be 'fixed' or 'exp', if you are using exp then the second column in the optim_regime should be a table with two elements: the start and the end lr for that row"
    )
    cmd:option(
        "-optim_regimes",
        {{30000, 1e-3, 5e-4}, {30000, 1e-4, 5e-4}, {20000, 5e-5, 5e-4}},
        "Optimization regime, each row is the number of iterations, the learning rate, and the weight decay"
    )
    cmd:option(
        "-optim_snapshot_iters",
        10000,
        "Iterations between snapshots (used for saving the network)"
    )
    cmd:option(
        "-save_path",
        "./data/trained_models",
        "Path to be used for saving the trained models"
    )

    -- New configs
    cmd:option(
        "-groups",
        1,
        "Number of groups to use in group convolutions"
    )
    cmd:option(
        "-use_batchnorm",
        false,
        "Whether to use batch normalization in MaryamNet models"
    )
    cmd:option(
        "-use_maryamnet_inputmaker",
        false,
        "Whether to use MaryamNet input maker"
    )
    cmd:option(
        "-use_maryamnet255_inputmaker",
        false,
        "Whether to use MaryamNet input maker"
    )
    cmd:option(
        "-freeze_batchnorm",
        false,
        "Whether to freeze the Batch Normalization layers"
    )
    cmd:option(
        '-use_identity_convbn',
        false,
        "Whether to use frozen identity conv batch normalization"
    )

    -- Parsing the command line
    config = cmd:parse(arg or {})
    return config
end

return config
