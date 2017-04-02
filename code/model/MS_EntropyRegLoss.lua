require 'nn';
local utils = require '../util/utils.lua'

local MS_EntropyRegLoss, Criterion = torch.class('nn.MS_EntropyRegLoss', 'nn.Criterion')

function MS_EntropyRegLoss:__init(w)
   Criterion.__init(self)
   self.w = w or 1.0
   self.max_grad = 10000.0
end

-- Calculate loss as -w*Beta*log(Beta)
function MS_EntropyRegLoss:updateOutput(input, target)
  local beta = input:clone()
  beta = -self.w * beta:cmul(torch.log(beta))
  self.output = torch.sum(beta) / beta:nElement()
  return self.output
end

-- derivative dL/dBeta = -w*(1 + log(Beta))
function MS_EntropyRegLoss:updateGradInput(input, target)
  local beta = input:clone()
  self.gradInput = -self.w * (1 + torch.log(beta))
  -- The gradient is not defined for Beta=0) so we should theoretically clamp the
  -- gradient
  self.gradInput = self.gradInput:clamp(-self.max_grad, self.max_grad)
  local l1_norm = torch.max(torch.abs(self.gradInput))
  assert(l1_norm == l1_norm)
  return self.gradInput
end

return nn.MS_EntropyRegLoss

