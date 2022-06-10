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
