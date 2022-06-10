# EPFL_OPT_Project

### Create a simulator with different worker number
test_as = Asynchronous_Simulator(num_workers=1)
### Train with a learning rate using SGD
test_as.train(lr=0.01)

### Run test on the dataset to see the test accuracy
test_as.test()
