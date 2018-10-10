--[[ Preprocessing script for training DCGAN ]]--

require 'lfs'
require 'xlua'
require 'paths'
require 'image'
require 'torch'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Preprocesses and saves images.')
cmd:text()
cmd:text('Optional arguments:')

cmd:option('-x1', 80, '')
cmd:option('-y1', 80, '')
cmd:option('-prescale_x', 90, '')
cmd:option('-prescale_y', 90, '')
cmd:option('-width', 36, '')
cmd:option('-height', 36, '')

opt = cmd:parse(arg)

local DATASET_DIR = '../datasets/lfw-deepfunneled'
local DATA_DIR = 'data'
local PREPRO_DIR = 'preprocessed_images'

if not (path.exists(path.join(DATA_DIR, PREPRO_DIR))) then
    if not (path.exists(DATA_DIR)) then
        lfs.mkdir(DATA_DIR)
    end
    lfs.mkdir(path.join(DATA_DIR, PREPRO_DIR))
end

print('Preprocessing images...')
for folder in paths.files(DATASET_DIR) do
    for file in paths.files(path.join(DATASET_DIR, folder), function(nm) return nm:find('.jpg') end) do
        if not (path.exists(path.join(DATA_DIR, PREPRO_DIR, file))) then
            local img = image.load(path.join(DATASET_DIR, folder, file))
            img = image.crop(img, opt.x1, opt.y1, opt.x1 + opt.prescale_x, opt.y1 + opt.prescale_y)
            img = image.scale(img, opt.width, opt.height)
            image.save(path.join(DATA_DIR, PREPRO_DIR, file), img)
        end
    end
end