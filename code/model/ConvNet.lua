require 'torch'
require 'nn'
require 'loadcaffe'

local utils = require 'util.utils'

local ConvNet, parent = torch.class('nn.ConvNet', 'nn.Module')

function ConvNet:__init(kwargs)
  self.net = nil
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify')
end

-- This architecture is taken from Torch's demo code.
function ConvNet:getSimpleModel()
  local model = nn.Sequential()

  ------------------------------------------------------------
  -- convolutional network 
  ------------------------------------------------------------
  -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
  model:add(nn.SpatialConvolutionMM(1, 28, 5, 5))
  model:add(nn.Tanh())
  model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
  -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
  model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
  model:add(nn.Tanh())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  -- stage 3 : standard 2-layer MLP:
  model:add(nn.Reshape(64*3*3))
  model:add(nn.Linear(64*3*3, 200))
  model:add(nn.Tanh())
  model:add(nn.Linear(200, self.num_classify))

  return model

end

function ConvNet:createModel()
  local model = nn.Sequential()
  self.net = model
end

function ConvNet:updateType(dtype)
  self.net = self.net:type(dtype)
end

function ConvNet:updateOutput(input)
  return self.net:forward(input)
end

function ConvNet:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end

function ConvNet:parameters()
  return self.net:parameters()
end

function ConvNet:training()
  self.net:training()
  parent.training(self)
end

function ConvNet:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end

function ConvNet:clearState()
  self.net:clearState()
end

