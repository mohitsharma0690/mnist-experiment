require 'torch'
require 'hdf5'
require 'math'
require 'paths'
require 'image'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(kwargs)
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.train_h5 = utils.get_kwarg(kwargs, 'train_h5')
  self.test_h5 = utils.get_kwarg(kwargs, 'test_h5')

  self.train_data = self:load_data(self.train_h5, 'train')
  self.test_data = self.load_data(self.test_h5, 'test')

  print('Did read and load data into memory')
end

function DataLoader:load_raw_data(h5_file, data_type)
  local h5_f = hdf5.open(h5_file, 'r')
  local data = hdf5:read('/data'):all()
  local label, noisy_label
  label = hdf5:read('/label'):all()
  if data_type == 'train' then
    noisy_label = hdf5:read('/noisy_label'):all()
  end
  local X = torch.Tensor(data:size()):copy(data)
  local y = torch.Tensor(label:size()):copy(label)
  local noisy_y
  if noisy_y ~= nil then
    noisy_y = torch.Tensor(noisy_label:size()):copy(noisy_label)
  end
  return {X=X, y=y, noisy_y = noisy_y}
end

function DataLoader:get_features_for_batch(batch_idx, batch_type)
  local X = torch.Tensor(#batch_idx, 1, 28, 28):zero()
  local y = torch.Tensor(#batch_idx):zero()
  
  local data, label
  if batch_type == 'train' then
    data = self.train_data.X
    label = self.train_data.noisy_y
  elseif batch_type == 'test' then
    data = self.test_data.X
    label = self.test_data.y
  else assert(false) end

  assert(data ~= nil and label ~= nil)
  for i=1,#batch_idx do
    local idx = batch_idx[i]
    X[{{i},{}}] = data[{{idx},{}}]:clone()
    y[{{i}}] = label[{{idx}}]
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
      local idx = self.shuffle_order[(i-1)*self.batch_size+j]
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
      coroutine.yield(X_batch, y_batch)
      collectgarbage()
      batch_data_idx = {}
    end
  end
  if #batch_data_idx ~= 0 then
    local X_batch, y_batch = self:get_features_for_batch(
      batch_data_idx, 'test')
    coroutine.yield(X_batch, y_batch)
  end
end

