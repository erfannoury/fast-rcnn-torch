-- Require the detection package
require "detection"

-- Paths
local dataset_name = config.dataset
local image_set = config.train_img_set
local dataset_dir = paths.concat(config.dataset_path, dataset_name)
local ss_dir = "./data/datasets/selective_search_data/"
local ss_file = paths.concat(ss_dir, dataset_name .. "_" .. image_set .. ".mat")
local param_path = config.pre_trained_file
local model_path = config.model_def
print('Creating save directory at ' .. config.save_path)
paths.mkdir(config.save_path)
print('Creating log directory at ' .. config.log_path)
paths.mkdir(config.log_path)


-- Loading the dataset
local dataset
model_opt = {}
if config.dataset == "MSCOCO" then
    print("MSCOCO " .. image_set)
    dataset = detection.DataSetCoco({image_set = image_set, datadir = dataset_dir, test_mode = false})
    model_opt.nclass = 80
else
    print("VOC " .. image_set)
    local year = 2007
    if config.dataset:find(2012) then
        year = 2012
    end
    dataset =
        detection.DataSetPascal(
        {image_set = image_set, datadir = dataset_dir, roidbdir = ss_dir, roidbfile = ss_file, year = year}
    )
    model_opt.nclass = 20
end


-- Creating the detection network
model_opt.test = false
model_opt.nclass = dataset:nclass()
model_opt.groups = config.groups
model_opt.fine_tunning = not config.resume_training
model_opt.use_bn = config.use_batchnorm
network = detection.Net(model_path, param_path, model_opt)

-- Creating the network wrapper
local network_wrapper = detection.NetworkWrapper() -- This adds train and test functionality to the global network

if config.do_identity_convbn_warmup then
    print("Warming up the network before training...")
    network_wrapper:warmup(dataset)
end
-- Train the network on the dataset
print("Training the network...")
network_wrapper:trainNetwork(dataset)
