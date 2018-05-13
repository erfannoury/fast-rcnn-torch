require "detection"
local utils = detection.GeneralUtils()
-- To define new models your file should:
-- 1) return the model
-- 2) return a local variable named regressor pointing to the weights of the bbox regressor
-- 3) return a local variable named classifier pointing to weights of the classifier (without SoftMax!)
-- 4) return the name of the model (used for saving models and logs)

local function create_model(opt)
    local name = "frcnn_maryamnet"
    if opt.use_bn then
        name = "frcnn_maryamnetbn"
    end
    backend = backend or cudnn

    -- SHARED PART
    local shared = nn.Sequential()
    local conv1 = backend.SpatialConvolution(
        3, 96, 11, 11, 4, 4, 2, 2, 1)
    conv1.name = "conv1"
    shared:add(conv1)
    if opt.use_bn then
        bn1 = nn.SpatialBatchNormalization(96)
        bn1.name = "batchnorm1"
        shared:add(bn1)
    end
    shared:add(backend.ReLU(true))
    shared:add(backend.SpatialMaxPooling(3, 3, 2, 2))

    local conv2 = backend.SpatialConvolution(
        96, 256, 5, 5, 1, 1, 2, 2, opt.groups)
    conv2.name = "conv2"
    shared:add(conv2)
    if opt.use_bn then
        bn2 = nn.SpatialBatchNormalization(256)
        bn2.name = 'batchnorm2'
        shared:add(bn2)
    end
    shared:add(backend.ReLU(true))
    shared:add(backend.SpatialMaxPooling(3, 3, 2, 2))

    local conv3 = backend.SpatialConvolution(
        256, 384, 3, 3, 1, 1, 1, 1, 1)
    conv3.name = "conv3"
    shared:add(conv3)
    if opt.use_bn then
        bn3 = nn.SpatialBatchNormalization(384)
        bn3.name = 'batchnorm3'
        shared:add(bn3)
    end
    shared:add(backend.ReLU(true))

    local conv4 = backend.SpatialConvolution(
        384, 384, 3, 3, 1, 1, 1, 1, opt.groups)
    conv4.name = "conv4"
    shared:add(conv4)
    if opt.use_bn then
        bn4 = nn.SpatialBatchNormalization(384)
        bn4.name = 'batchnorm4'
        shared:add(bn4)
    end
    shared:add(backend.ReLU(true))

    local conv5 = backend.SpatialConvolution(
        384, 256, 3, 3, 1, 1, 1, 1, opt.groups)
    conv5.name = "conv5"
    shared:add(conv5)
    if opt.use_bn then
        bn5 = nn.SpatialBatchNormalization(256)
        bn5.name = 'batchnorm5'
        shared:add(bn5)
    end
    shared:add(backend.ReLU(true))

    -- Convolutions and roi info
    local shared_roi_info = nn.ParallelTable()
    shared_roi_info:add(shared)
    shared_roi_info:add(nn.Identity())

    -- Linear Part
    local linear = nn.Sequential()
    linear:add(nn.View(-1):setNumInputDims(3))
    local fc6 = nn.Linear(9216, 4096)
    fc6.name = "fc6"
    linear:add(fc6)
    if opt.use_bn then
        linear:add(backend.BatchNormalization(4096))
    end
    linear:add(backend.ReLU(true))
    linear:add(nn.Dropout(0.5))

    local fc7 = nn.Linear(4096, 4096)
    fc7.name = "fc7"
    linear:add(fc7)
    if opt.use_bn then
        linear:add(backend.BatchNormalization(4096))
    end
    linear:add(backend.ReLU(true))
    linear:add(nn.Dropout(0.5))

    -- classifier
    local classifier = nn.Linear(4096, opt.nclass + 1)
    classifier.name = "classifier"
    -- regressor
    local regressor = nn.Linear(4096, 4 * (opt.nclass + 1))
    regressor.name = "regressor"

    local output = nn.ConcatTable()
    output:add(classifier)
    output:add(regressor)

    -- ROI pooling
    local ROIPooling = detection.ROIPooling(6, 6):setSpatialScale(1 / 16)

    -- Whole Model
    local model = nn.Sequential()
    model:add(shared_roi_info)
    model:add(ROIPooling)
    model:add(linear)
    model:add(output)

    model:cuda()
    return model, classifier, regressor, name, shared
end

return create_model
