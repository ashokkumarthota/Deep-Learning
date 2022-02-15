import numpy as np
from fashnMnist.NeuralNetwork import NeuralNetwork
from fashnMnist.optimizer.MomentumGradiantDecent import MomentumGradiantDecent
from fashnMnist.optimizer.RmsProp import RmsProp
from fashnMnist.optimizer.NAG import NAG
from fashnMnist.optimizer.Adam import Adam
from fashnMnist.optimizer.NAdam import NAdam


class FashnMnist:
    def __init__(self,x, y,lr = .5,epochs =100,batch=500,HiddenLayerNuron=[60,10],decay_rate=0,activation='tanh',optimizer='GradiantDecent',beta1=0.9,beta2=0.99,gamma=0.9,beta=.9,initializer='he',dropout_rate=0):
                self.network=None
 
                self.y=y
                self.optimizer=optimizer
                if (self.optimizer=='gd'):
                    self.network=NeuralNetwork( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate,initializer=initializer)
                    
                if (self.optimizer=='sgd'):
                    self.network=NeuralNetwork( x=x, y=y, lr = lr,  epochs =epochs,batch=1,HiddenLayerNuron=HiddenLayerNuron,activation=activation,initializer=initializer)
                    
                if (self.optimizer=='mgd'):
                    self.network=MomentumGradiantDecent( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate,initializer=initializer)
                    
         
                if(self.optimizer=='nag'):
                     self.network=NAG( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,gamma=gamma,initializer=initializer)
                
                if(self.optimizer=='adam'):
                    self.network=Adam( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate,beta1=beta1,beta2=beta2,initializer=initializer,dropout_rate=dropout_rate)
                    
                if(self.optimizer=='nadam'):
                    self.network=NAdam( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate,beta1=beta1,beta2=beta2,initializer=initializer,dropout_rate=dropout_rate)
                
                if(self.optimizer=='rms'):
                     self.network=RmsProp( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate,initializer=initializer,dropout_rate=dropout_rate)
                    
                
    def train(self): 
        self.network.train()
       
    def predict(self,x,y):
        prediction=self.network.predict(x)
        prediction=y-prediction
        accurecy=sum(x==0 for x in prediction)
        accurecy=accurecy/len(prediction)
        print('Test accuracy='+str(accurecy*100))
        return prediction
    