require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cunn'
require 'paths' 

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-LearningRate', 1, 'learning rate at t=0')
cmd:option('-LearningRateDecay', 8e-7, 'learning rate decay rate per evaluation')
cmd:option('-batchSize', 16, 'mini-batch size (1 = pure stochastic)')
cmd:option('-epochSize',110000, 'epoch size')
cmd:option('-weightDecay', 1e-6, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-stdv', 0.05,'std for initializing parameters')
cmd:option('-renorm',true,'specify whether to renormalize kernel (for SpationConvolutionMM and Linear only)')
cmd:option('-max_kernel_norm', 0.9, 'if renorm is true, constrain kernel norm to the value')
cmd:option('-setDevice', 1, 'specify which gpu to use')
cmd:option('-windowsize',1000,'input sequence length')
cmd:option('-continue',false,'continue training')
cmd:option('-verbose',false,'verbose')
cmd:option('-L1Sparsity',0,'L1 penalty to last hidden layer')
cmd:text()
opt = cmd:parse(arg or {})



torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.setDevice)
torch.manualSeed(opt.seed)




rundir = cmd:string('model', opt, {setDevice=true,continue=true})
opt.save = opt.save .. '/' .. rundir


if not paths.dirp(opt.save) then
   --os.execute('rm -r ' .. opt.save)
   os.execute('mkdir -p ' .. opt.save)
end



cmd:addTime('model')
cmd:log(opt.save .. '/log.txt', opt)




dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'

Do = true
while Do do
   Do = train()
end
