require 'math'
require 'torch'
require 'optim'
require 'string'


torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
math.randomseed(opt.seed)


model:cuda()
criterion:cuda()


model:training()
model.max_kernel_norm=opt.max_kernel_norm
if opt.stdv ~=0 then
   model:reset(stdv)
end

if opt.continue then
   model = torch.load(paths.concat(opt.save, 'latestmodel.net'))
   model:cuda()
end




trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'),true)
validLogger = optim.Logger(paths.concat(opt.save, 'valid.log'),true)


if model then
   parameters, gradParameters = model:getParameters()
end


optimState = {
   learningRate = opt.LearningRate,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   learningRateDecay = opt.LearningRateDecay
}
optimMethod = optim.sgd


if opt.continue then
  optimState = torch.load(paths.concat(opt.save, 'latestmodel.optimState'))
end

----------------------------------------------------------------------


countdown = 0
cumt=1
Counter=1
minValidloss = 1e30

function train()
   epoch = epoch or 1
   model:training()


   -- shuffle at each epoch
   shuffle = torch.randperm(tr_size)

   print('Training:')
   print("Epoch " .. epoch )

   local tloss=0.0
   for t = cumt,math.min(cumt+opt.epochSize-1,trainData:size()),opt.batchSize do
      collectgarbage()
      
      local inputs = torch.Tensor(opt.batchSize, nfeats, width, height)
      local targets = torch.Tensor(opt.batchSize, noutputs)

      -- load a mini batch of training data
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
  	 input = trainData.data[ { {shuffle[i]},{},{ 1+(1000-opt.windowsize)/2.0,1000-(1000-opt.windowsize)/2.0 } } ]:float()
         local target = trainData.labels[shuffle[i]]:float()
	 inputs[k]= input
         targets[k]= target
	 k = k + 1
      end

      inputs = inputs:cuda()
      targets = targets:cuda()


      local feval = function(x)
           if x ~= parameters then
               parameters:copy(x)
           end

           gradParameters:zero()
           local NLL = 0
           -- forward pass - compute output
           local output = model:forward(inputs)
           local err = criterion:forward(output, targets)

           -- backforward pass - compute gradient
           if opt.L1Sparsity ~= 0 then
               L1criterion.l1weight = opt.L1Sparsity
	       local df_l1 = L1criterion:backward(model:get(model:size()-1).output)
	       model:get(model:size()-1):backward(model:get(model:size()-2).output,df_l1)
	   end
           local df_do = criterion:backward(output, targets)
           model:backward(inputs, df_do)
			  

           NLL = NLL + err
	   tloss = tloss + err

	   --apply max kernel norm constraint
           if opt.renorm then
      	       for i = 1,#model.modules do
                   if string.find(tostring(model.modules[i]), 'SpatialConvolutionMM') or  string.find(tostring(model.modules[i]),'Linear')  then
		       model.modules[i].weight:renorm(2,1,opt.max_kernel_norm)
	     	   end
	       end
      	   end
           return NLL,gradParameters
      end

      optimMethod(feval, parameters, optimState)

   end
   cumt = cumt + opt.epochSize
   if cumt > trainData:size() then
      cumt = 1
   end

   print("Average NLL (Train) = " .. (tloss * opt.batchSize / opt.epochSize) )
   trainLogger:add{['Average NLL (Train)'] = tloss * opt.batchSize / opt.epochSize}


   -- turn off dropout evalutation
   model:evaluate()

   print('Testing on valid set:')
   tloss = 0.0


   shuffle = torch.randperm(te_size)

   for t = 1,4000,opt.batchSize do
      local inputs = torch.Tensor(opt.batchSize, nfeats, width, height)
      local targets = torch.Tensor(opt.batchSize, noutputs)
      collectgarbage()

      -- load a mini-batch of validation data
      k = 1
      for i = t,math.min(t+opt.batchSize-1,4000) do
         input = validData.data[ { {shuffle[i]},{},{ 1+(1000-opt.windowsize)/2.0,1000-(1000-opt.windowsize)/2.0 } } ]:float()
         local target = validData.labels[shuffle[i]]:float()
         inputs[k]= input
         targets[k]= target
         k = k + 1
      end

      inputs = inputs:cuda()
      targets = targets:cuda()


      local output = model:forward(inputs)
      local err = criterion:forward(output, targets)
      tloss = tloss + err
   end


   print("Average NLL (Valid) = " .. (tloss * opt.batchSize / validData:size()) )
   validLogger:add{['Average NLL (Valid)'] = tloss * opt.batchSize / validData:size()}
   

   if tloss < minValidloss then
      minValidloss = tloss
      countdown = 0
      -- save model to bestmodel.net if it has the best valid set NLL so far
      local filename = paths.concat(opt.save, 'bestmodel.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      torch.save(filename, model)

   else
	countdown = countdown + 1
	if countdown > 300 then
	   return false
	end
   end

   print("Best NLL (Valid) =" .. minValidloss / te_size)

   -- save current model
   local filename = paths.concat(opt.save, 'latestmodel.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   torch.save(filename, model)
   
   -- save optimState
   local filename = paths.concat(opt.save, 'latestmodel.optimState')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   torch.save(filename, optimState)

   -- next epoch
   epoch = epoch + 1
   return true
end
