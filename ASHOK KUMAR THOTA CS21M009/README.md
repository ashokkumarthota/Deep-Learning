# For all coding questions, I followed the same flow as mentioned below

#### 1. Importing the Required Packages
#### 2. Setting the random seed
#### 3. Splitting the fashion-mnist dataset as train and validation, and Normalisation
#### 4. Reserving 10% training data for testing (this data is different from validation)
#### 5. One Hot Encoding 
#### 6. Wandb  Configuration, Sweep Configuration, and generating Sweep Id (if required)
#### 7.  Main Class or Logic
#### 8. Calling Agent with Sweep Id or Creating Instance of the class and calling fit function
#### 9. Displaying Accuracy



# Questions

| Questions      | Links |
| ----------- | ----------- |
| Question 1     | [Question 1](https://github.com/ashokkumarthota/Deep-Learning/blob/main/ASHOK%20KUMAR%20THOTA%20CS21M009/1ST%20QUESTION%20(PLOTTING%20IMAGES).ipynb)       |
| Question 2,3,4,5,6    | [Question 2](https://github.com/ashokkumarthota/Deep-Learning/blob/main/ASHOK%20KUMAR%20THOTA%20CS21M009/2ND%20QUESTION%20(FFNN).ipynb)       , [Question 3](https://github.com/ashokkumarthota/Deep-Learning/blob/main/ASHOK%20KUMAR%20THOTA%20CS21M009/3RD%20QUESTION%20(OPTIMIZATION%20ALGO).ipynb)     ,  [Question 4,5,6](https://github.com/ashokkumarthota/Deep-Learning/blob/main/ASHOK%20KUMAR%20THOTA%20CS21M009/4TH%2C5TH%2C6TH%20QUESTION%20(SWEEPS).ipynb)            |
| Question 7    | [Question 7](https://github.com/ashokkumarthota/Deep-Learning/blob/main/ASHOK%20KUMAR%20THOTA%20CS21M009/7TH%20QUESTION%20(CONFUSION%20MATRIX).ipynb)       |
|Question 8| [Question 8](https://github.com/ashokkumarthota/Deep-Learning/blob/main/ASHOK%20KUMAR%20THOTA%20CS21M009/8TH%20QUESTION(MEAN%20SQUARE%20ERROR).ipynb)  |



## Neural Network Class
#### Class Constructor
```
FFSN_MultiClass(
  n_inputs : Number of features in the data (24*24=784 in this case)
  ,n_outputs : Number of output classes (10 in this case)
  ,noof_hidden : Number of Hidden Layers [2,3,4]
  ,size_of_every_hidden : Size of every Hidden Layer [32,64,128]
  ,init_method = Initialisation mathods ['random','xavier','he'] 
  ,activation_function = Activation function ['sigmoid','tanh','relu','leaky_relu']
  )
```
#### Object Creation 
```
ffsn_multi = FFSN_MultiClass(
  n_inputs=784
  ,n_outputs=10
  ,noof_hidden=2
  ,size_of_every_hidden=32
  ,init_method='xavier'
  ,activation_function='relu')
```

#### fit function
```
ffsn_multi.fit(
  x_train : Normalized training data
  ,y_OH_train : Classes for x_train in one hot encoding formate
  ,epochs=5 : Number of epochs 
  ,learning_rate : Learning rate
  ,algo : Optimization algorithm
  ,mini_batch_size : Batch size
  )
```
#### calling the fit function
```
ffsn_multi.fit(
  x_train
  ,y_OH_train
  ,epochs=5
  ,learning_rate=0.001
  ,algo= "Adam"
  ,mini_batch_size=64)
```
