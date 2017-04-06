local utils = require '../util/utils.lua'

require 'nn'
require 'cunn'
require 'hdf5'
require 'math'
require 'paths'
require 'optim'

require 'model.ConvNet'

local train_cls = {}

function train_cls.setup(args)
  local self = train_cls
  self.dtype = args.dtype
  self.data_loader = args.data_loader
  self.model = args.model
  self.train_conf = args.train_conf
  self.val_conf = args.val_conf
  self.checkpoint = {
    train_loss_history={},
    val_loss_history={},
    grads_history={},
    fake_loss_history={},
  }

  print(self.model.net)
  
  self.crit = nn.CrossEntropyCriterion():type(self.dtype)
  self.crit_conf = self.get_loss_confusion()
  print(self.crit_conf)
  
  self.params, self.grad_params = self.model:getParameters()
end

function train_cls.read_data_co(data_co, data_loader)
  local success, x, y = coroutine.resume(data_co, data_loader)
  if not success then print('Data couroutine returns fail.') end
  if x == nil then
    print('x is nil returning 0 loss')
  elseif success == false then
    -- print crash logs
    if torch.isTensor(x) then 
      print(x:size())
    else
      print(x)
    end
  end
  return success,x,y
end

function train_cls.get_loss_confusion()
  local conf = torch.ones(G_global_opts.num_classify, G_global_opts.num_classify)
  conf = conf:mul(0.02)
  -- Add custom confusions here
  conf[1][10] = 0.5
  conf[2][8] = 0.5
  conf[3][6] = 0.5
  conf[4][9] = 0.5
  conf[5][7] = 0.5
  conf[6][3] = 0.5
  conf[7][1] = 0.5
  conf[8][5] = 0.5
  conf[9][4] = 0.5
  conf[10][2] = 0.5
  return conf
end

function train_cls.f(w)
  local self = train_cls
  assert(w == self.params)
  self.grad_params:zero() 

  local success, x, y = self.read_data_co(self.data_co, self.data_loader)
  if not x then return 0, self.grad_params end

  x = utils.convert_to_type(x, self.dtype)
  y = utils.convert_to_type(y, self.dtype)

  local scores = self.model:forward(x)
  local _, preds = torch.max(scores, 2)
  preds = preds:view(-1)
  local loss = self.crit:forward(scores, y)

  -- Scale the gradients wrt losses. Note that we can do this scaling here instead
  -- of doing it in the loss calculation since the scaling term is independent of
  -- the input vars.

  local grad_scores = self.crit:backward(scores, y) 
  -- Get the gradient scales based on the confusion matrix above
  local grad_scales = torch.zeros(grad_scores:size(1))
  for i=1,grad_scores:size(1) do
    grad_scales[i] = 1.0 - self.crit_conf[y[i]][preds[i]]
  end
  grad_scales = torch.repeatTensor(grad_scales, grad_scores:size(2), 1):t()
  grad_scales = grad_scales:cuda()
  grad_scores = grad_scores:cmul(grad_scales)
  self.model:backward(x, grad_scores)

  -- Get the real loss just for logging purpose
  local fake_softmax = nn.LogSoftMax():cuda()
  local fake_logsoftmax = fake_softmax:forward(scores)
  local fake_loss = 0
  for i=1,fake_logsoftmax:size(1) do
    local fake_pred = preds[i]
    fake_loss = fake_loss + (1.0 - self.crit_conf[y[i]][fake_pred])*fake_logsoftmax[i][y[i]]
  end
  fake_loss = (-1*fake_loss) / fake_logsoftmax:size(1)

  if G_global_opts.grad_clip > 0 then
    self.grad_params:clamp(
      -G_global_opts.grad_clip, G_global_opts.grad_clip)
  end

  -- Add to confusion matrix
  self.train_conf:batchAdd(scores, y)

  -- Add losses to history
  table.insert(self.checkpoint.train_loss_history, loss)

  if G_global_opts.debug_weights == 1 then
    local curr_grad = self.model:getGradWeights(loss, x, y) 
    table.insert(self.checkpoint.grads_history, curr_grad)
    table.insert(self.checkpoint.fake_loss_history, fake_loss)
  end

  return loss, self.grad_params
end

function train_cls.validate(val_data_co)
  local self = train_cls
  local val_loss, num_val = 0, 0

  self.val_conf:zero()

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv = coroutine.resume(
        val_data_co, self.data_loader) 

    if success and xv ~= nil then
      xv = utils.convert_to_type(xv, self.dtype)
      yv = utils.convert_to_type(yv, self.dtype)

      local scores = self.model:forward(xv)
      val_loss = val_loss + self.crit:forward(scores, yv)

      self.val_conf:batchAdd(scores, yv)
      num_val = num_val + 1
    elseif success ~= true then
      print('Validation data coroutine failed')
      print(xv)
    end
  end

  val_loss = val_loss / num_val
  print('val_loss = ', val_loss)
  table.insert(self.checkpoint.val_loss_history, val_loss)
  print(self.val_conf)
  self.model:training()
end

function train_cls.train(train_data_co, optim_config, stats)
  local self = train_cls
  self.data_co = train_data_co
  self.model:training()

  local _, loss = optim.adam(self.f, self.params, optim_config)
  
  local print_every = G_global_opts.print_every
  if stats.curr_batch>0 and stats.curr_batch % print_every == 0 then
     local msg = 'Epoch: [%d/%d]\t Iteration:[%d/%d]\tTotal loss: %.2f\t'
     msg = msg..'Real loss: %.2f'

    local logs = self.checkpoint
    local args = {
      msg,
      stats.curr_epoch, stats.total_epoch,
      stats.curr_batch, stats.total_batch,
      logs.train_loss_history[#logs.train_loss_history],
      logs.fake_loss_history[#logs.fake_loss_history],
    }
    print("Gradients: ")
    print(logs.grads_history[#logs.grads_history])
    print(string.format(unpack(args)))
  end

  return loss
end

function train_cls.getCheckpoint()
  local self = train_cls
  local cp = {}
  for k,v in pairs(self.checkpoint) do cp[k] = v end
  cp.train_conf = self.train_conf:__tostring__()
  cp.val_conf = self.val_conf:__tostring__()
  return cp
end

return train_cls

