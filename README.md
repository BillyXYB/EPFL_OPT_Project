# EPFL_OPT_Project
## This is a simulator training framework for Asynchronous SGD. 

### Create a simulator with different worker number, and different backbone
```
test_as = Asynchronous_Simulator(num_workers=1, model_name='small')
```
Examples of supported backbones : small, ResNet18, Vgg11 ... 
(more could be seen in files under modle). 
### Train with a learning rate using SGD
```
test_as.train(lr=0.01)
```
### Run test on the dataset to see the test accuracy
```
test_as.test()
```
### Adding momentum to ASGD
```
test_as = Asynchronous_Simulator(num_workers=1, model_name='small')
test_as.train(max_epoch=2, lr = 0.01, momentum = 0.9)
test_as.test()
```
### SGD+dropout
example dropout rate = 0.5
```
test_as = Asynchronous_Simulator(num_workers=1, model_name="small_drop_5")
test_as.train(max_epoch=2, lr = 0.01)
test_as.test()
```
