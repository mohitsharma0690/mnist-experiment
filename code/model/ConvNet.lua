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
  model:add(nn.SpatialConvolution(1, 32, 5, 5))
  model:add(nn.Tanh())
  model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
  -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
  model:add(nn.SpatialConvolution(32, 64, 5, 5))
  model:add(nn.Tanh())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  -- stage 3 : standard 2-layer MLP:
  model:add(nn.Reshape(64*2*2))
  model:add(nn.Linear(64*2*2, 200))
  model:add(nn.Tanh())
  model:add(nn.Linear(200, self.num_classify))

  return model

end

-- This architecture is taken from
-- https://github.com/szagoruyko/cifar.torch/blob/master/models/vgg_bn_drop.lua
function ConvNet:getCifar10Model()

  local vgg = nn.Sequential()

  -- building block
  function ConvBNReLU(nInputPlane, nOutputPlane)
    vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
    vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
    vgg:add(nn.ReLU(true))
    return vgg
  end

  -- Will use "ceil" MaxPooling because we want to save as much
  -- space as we can
  ConvBNReLU(3,64):add(nn.Dropout(0.3))
  ConvBNReLU(64,64)
  vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())

  ConvBNReLU(64,128):add(nn.Dropout(0.4))
  ConvBNReLU(128,128)
  vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())

  ConvBNReLU(128,256):add(nn.Dropout(0.4))
  ConvBNReLU(256,256):add(nn.Dropout(0.4))
  ConvBNReLU(256,256)
  vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())

  ConvBNReLU(256,512):add(nn.Dropout(0.4))
  ConvBNReLU(512,512):add(nn.Dropout(0.4))
  ConvBNReLU(512,512)
  vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())

  ConvBNReLU(512,512):add(nn.Dropout(0.4))
  ConvBNReLU(512,512):add(nn.Dropout(0.4))
  ConvBNReLU(512,512)
  vgg:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
  vgg:add(nn.View(512))

  classifier = nn.Sequential()
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(512,512))
  classifier:add(nn.BatchNormalization(512))
  classifier:add(nn.ReLU(true))
  classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(512,10))
  vgg:add(classifier)

  -- initialization from MSR
  function MSRinit(net)
    local function init(name)
      for k,v in pairs(net:findModules(name)) do
        local n = v.kW*v.kH*v.nOutputPlane
        v.weight:normal(0,math.sqrt(2/n))
        v.bias:zero()
      end
    end
    -- have to do for both backends
    init'nn.SpatialConvolution'
  end

  MSRinit(vgg)

  -- check that we can propagate forward without errors
  -- should get 16x10 tensor
  --print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))
  
  return vgg
end

function ConvNet:getGradWeights(loss, x, y)
  local grad_threshold = G_global_opts.grad_threshold or 1e-5
  local wt_1, grad_wt_1 = self.net:get(1):parameters()
  local grad_wt_1_f = grad_wt_1[1]:float()
  grad_wt_1 = torch.max(torch.abs(grad_wt_1_f))
  -- Get grad weights above threshold
  local wt_th = torch.sum(torch.gt(grad_wt_1_f, grad_threshold))
  -- Get max weights
  local max_wt_1 = torch.max(torch.abs(wt_1[1]:float()))

  local curr_grads = {
    max_wt=max_wt_1,
    max_grad_wt=grad_wt_1,
    wt_gt_th=wt_th,
    total_wt=grad_wt_1_f:nElement()
  }

  if max_wt_1 ~= max_wt_1 or grad_wt_1 ~= grad_wt_1 then
    print("Nan detected")
    print("loss "..loss)
    assert(false)
  end

  return curr_grads
end

function ConvNet:createModel()
  local model
  if G_global_opts.cifar == 1 then model = self:getCifar10Model() 
  else model = self:getSimpleModel() end
  self.net = model
end

function ConvNet:updateType(dtype)
  self.net = self.net:type(dtype)
end

function ConvNet:updateOutput(input)
  return self.net:forward(input)
end

function ConvNet:forward(input)
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

