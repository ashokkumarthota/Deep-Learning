# Packages

 **fashnMnist/NeuralNetwork.py**
- Implementation of neural network.It has all important common functions for forward propagation, back propagation
which are used by all other optimizers

**fashnMnist/FashnMnist.py**
- This class call NeuralNetwork or other optimizers based on input provided by used 
		
**fashnMnist/Initializers.py**
-This classs used to define  various weight initializers .Mostly called from NeuralNetwork
		
**fashnMnist/Preprocessor.py**
- This classs used to preprocess input data like apply one hot encoding,normalization etc
		
**fashnMnist/Activations.py**
- This classs used to define  various Activation functions and there derivatives.Mostly called from NeuralNetwork.
		
**All optimizers:**	
- fashnMnist/optimizer/Adam.py
- fashnMnist/optimizer/NAG.py
- fashnMnist/optimizer/NAdam.py
- fashnMnist/optimizer/RMSProp.py
- fashnMnist/optimizer/MomentumGradiantDecent.py
		
Please note Basic gradiant decent and stochastic gradient descent thechnique implemented in **/fashnMnist/NeuralNetwork.py**

## Neural Network
```
FashnMnist(
	x={Features normalized using class Preprocessor}
	,y=[Training Labels.Data should be one hot encoded]
	,lr=[Learning rate ,DataType:{Float}, default:.1]
	,epochs =[Number of epochs]
	,batch=[size of batches under one epoch]
	,layer1_size=[Total hidden Nurons should present in first hidden /layes , DataType{Int}]
	,layer2_size= [Total hidden Nurons should present in second hidden layes , DataType{Int}]
	,layer3_size=[Total hidden Nurons should present in Third hidden layes , DataType{Int}]
,layer4_size=[Total hidden Nurons should present in fourth hiddenlayes , DataType{Int}]
	,layer5_size=[Total hidden Nurons should present in fifth hidden layes , DataType{Int}]
	,optimizer=['rms','adam','nadam','sgd','mgd','nag' ,default:'mgd']
	,initializer=['he','xavier','random',default: 'he']
	,activation=['tanh','sigmoid','relu' default:'tanh']
	,weight_decay=[weight decay for L2 regularization ,DataType:Float,default=0]
	,dropout_rate=[DataType:Float,default=0]
	)
```
Example
```
#setup
model=FashnMnist(
		x=x_trainNorm
		,y=y_trainNorm
		,lr=.001
		,epochs=10
		,batch=32
		,layer1_size=128
		,layer2_size=64
		,optimizer="nadam"
		,initializer="he"
		,activation="relu"
		,dropout_rate=.1
		)

# train the model
model.train() 

#GetPredicted result
pred,accTrain,lossTrain = model.GetRunResult(x_trainNorm,y_trainNorm)

```



	
