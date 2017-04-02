require 'nn';
local utils = require '../util/utils.lua'

local MS_EntropyRegLoss, Criterion = torch.class('nn.MS_EntropyRegLoss', 'nn.Criterion')

function MS_EntropyRegLoss:__init(w)
   Criterion.__init(self)
   self.w = w or 1.0
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
  self.gradInput = -self.w * torch.add((torch.log(beta), torch.ones(beta:size())))
  return self.gradInput
end

return nn.MS_EntropyRegLoss

