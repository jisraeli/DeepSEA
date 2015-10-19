require 'torch'
require 'hdf5'
require 'mattorch'


print 'Loading dataset'

train_file = 'train.mat'
valid_file = 'valid.mat'
noutputs=919


tr_size = 4400000
te_size = 4000 


loaded = mattorch.load(train_file)
trainData = {
    data = loaded['trainxdata']:transpose(3,1),
    labels = loaded['traindata']:transpose(2,1),
    size = function() return tr_size end
}

loaded = mattorch.load(valid_file)
validData = {
    data = loaded['validxdata']:transpose(3,1),
    labels = loaded['validdata']:transpose(2,1),
    size = function() return te_size end
}
   
print 'Finished loading dataset'




