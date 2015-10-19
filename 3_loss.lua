require 'torch'
require 'nn'


--log likelihood
criterion = nn.BCECriterion()

--L1 penalty to last layer
if opt.L1Sparsity ~= 0 then
   L1criterion = nn.L1Penalty(opt.L1Sparsity,false,false)
   L1criterion:cuda()
end
