require 'torch'
require 'hdf5'
require 'math'
require 'paths'
require 'image'
require 'nn'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(kwargs)
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.train_h5 = utils.get_kwarg(kwargs, 'train_h5')
  self.test_h5 = utils.get_kwarg(kwargs, 'test_h5')
  self.use_noise = utils.get_kwarg(kwargs, 'use_noise')

  self.train_data = self:load_raw_data(self.train_h5, 'train')
  self.test_data = self:load_raw_data(self.test_h5, 'test')
  self.cifar = utils.get_kwarg(kwargs, 'cifar')
  if self.cifar == 1 then
    if kwargs.use_cached_cifar_data == 1 and paths.filep('cifar10_data_yuv.t7') then
      print("Will use cached data")
      local cached_data = torch.load('cifar10_data_yuv.t7')
      self.train_data = cached_data.train
      self.test_data = cached_data.test
    else
      self.train_data.X, self.test_data.X = self:normalize()
      local cifar_10_preprocessed = {train=self.train_data, test=self.test_data}
      torch.save('cifar10_data_yuv.t7', cifar_10_preprocessed)
    end
  end

  self.val_batch_info = utils.get_kwarg(kwargs, 'val_batch_info')

  print('Did read and load data into memory')
end

function DataLoader:load_raw_data(h5_file, data_type)
  local h5_f = hdf5.open(h5_file, 'r')
  local data = h5_f:read('/data'):all()
  local label, noisy_label
  label = h5_f:read('/label'):all()
  if data_type == 'train' then
    noisy_label = h5_f:read('/noisy_label'):all()
  end
  local X = torch.Tensor(data:size()):copy(data)
  local y = torch.Tensor(label:size()):copy(label)
  local noisy_y
  if data_type == 'train' then
    noisy_y = torch.Tensor(noisy_label:size()):copy(noisy_label)
  end
  return {X=X, y=y, noisy_y=noisy_y}
end

function DataLoader:get_features_for_batch(batch_idx, batch_type)
  local X, y
  if self.cifar == 1 then X = torch.Tensor(#batch_idx, 3, 32, 32):zero()
  else X = torch.Tensor(#batch_idx, 1, 28, 28):zero() end
  y = torch.Tensor(#batch_idx):zero()
  
  local data, label
  if batch_type == 'train' then
    data = self.train_data.X
    if self.use_noise == 1 then label = self.train_data.noisy_y
    else label = self.train_data.y end
  elseif batch_type == 'test' then
    data = self.test_data.X
    label = self.test_data.y
  else assert(false) end

  assert(data ~= nil and label ~= nil)
  for i=1,#batch_idx do
    local idx = batch_idx[i]
    X[{{i},{}}] = data[{{idx},{}}]:clone()
    -- Do +1 since torch requires labels to start from 1
    y[{{i}}] = label[{{idx}}] + 1
  end

  return X, y
end

function DataLoader:train_data_size()
  return self.train_data.X:size(1)
end

function DataLoader:get_shuffle_order(size)
  return torch.randperm(size)
end

-- TODO(Mohit): For now we don't do any complex processing while loading
-- minibatches. We just create a shuffle order in the beginning and use
-- those indexes to get our training batch samples.
function DataLoader:next_train_batch()
  local total_train_data = self:train_data_size()
  local shuffle_order = self:get_shuffle_order(total_train_data)
  local num_batches = math.floor(total_train_data / self.batch_size)

  print("Total batches in 1 epoch "..num_batches)

  for i=1, num_batches do 
    local batch_data_idx = {}
    for j=1, self.batch_size do
      local idx = shuffle_order[(i-1)*self.batch_size+j]
      table.insert(batch_data_idx, idx)
    end
    local X_batch, y_batch = self:get_features_for_batch(batch_data_idx,
      'train')
    coroutine.yield(X_batch, y_batch)
    collectgarbage()
  end
end

function DataLoader:val_data_size()
  return self.test_data.X:size(1)
end

function DataLoader:next_unsorted_train_batch()
  local total_train_data = self:train_data_size()
  local num_batches = math.floor(total_train_data / self.batch_size)

  print("Total batches in 1 epoch "..num_batches)
  local batch = {}

  for i=1, num_batches do 
    local batch_data_idx = {}
    for j=1, self.batch_size do
      local idx = (i-1)*self.batch_size+j
      table.insert(batch_data_idx, idx)
    end
    local X_batch, y_batch = self:get_features_for_batch(batch_data_idx,
      'train')
    coroutine.yield(X_batch, y_batch, batch_data_idx)
    collectgarbage()
  end
end

function DataLoader:next_val_batch()
  local total_data = self:val_data_size()
  local num_batches = math.floor(total_data/self.batch_size)

  print("Total val batches in 1 epoch "..num_batches)

  local batch_data_idx = {}
  for i=1, total_data do 
    table.insert(batch_data_idx, i)
    if #batch_data_idx == self.batch_size then
      local X_batch, y_batch = self:get_features_for_batch(
        batch_data_idx, 'test')
      if self.val_batch_info == 1 then coroutine.yield(X_batch, y_batch, batch_data_idx)
      else coroutine.yield(X_batch, y_batch) end
      collectgarbage()
      batch_data_idx = {}
    end
  end
  if #batch_data_idx ~= 0 then
    local X_batch, y_batch = self:get_features_for_batch(
      batch_data_idx, 'test')
    if self.val_batch_info == 1 then coroutine.yield(X_batch, y_batch, batch_data_idx)
    else coroutine.yield(X_batch, y_batch) end
  end
end

function DataLoader:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.train_data.X:clone()
  local testData = self.test_data.X:clone()

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size(1) do
     xlua.progress(i, trainData:size(1))
     -- rgb -> yuv
     local rgb = trainData[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData:select(2,2):mean()
  local std_u = trainData:select(2,2):std()
  trainData:select(2,2):add(-mean_u)
  trainData:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData:select(2,3):mean()
  local std_v = trainData:select(2,3):std()
  trainData:select(2,3):add(-mean_v)
  trainData:select(2,3):div(std_v)

  self.train_data.mean_u = mean_u
  self.train_data.std_u = std_u
  self.train_data.mean_v = mean_v
  self.train_data.std_v = std_v

  -- preprocess testSet
  for i = 1,testData:size(1) do
    xlua.progress(i, testData:size(1))
     -- rgb -> yuv
     local rgb = testData[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData[i] = yuv
  end
  -- normalize u globally:
  testData:select(2,2):add(-mean_u)
  testData:select(2,2):div(std_u)
  -- normalize v globally:
  testData:select(2,3):add(-mean_v)
  testData:select(2,3):div(std_v)

  return trainData, testData
end

