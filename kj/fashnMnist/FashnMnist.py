import numpy as np
import pandas as pd
import sys
from fashnMnist.NeuralNetwork import NeuralNetwork
from fashnMnist.MomentumGradiantDecent import MomentumGradiantDecent
from fashnMnist.RmsProp import RmsProp
from fashnMnist.NAG import NAG
from fashnMnist.Adam import Adam
from fashnMnist.NAdam import NAdam
import matplotlib.pyplot as plt

class FashnMnist:
    def __init__(self,x, y,lr = .5,epochs =100,batch=500,HiddenLayerNuron=[60,10],decay_rate=0,activation='tanh',optimizer='GradiantDecent',beta1=0.9,beta2=0.99):
                self.network=None
 
                self.y=y
                self.optimizer=optimizer
                if (self.optimizer=='gd'):
                    self.network=NeuralNetwork( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate)
                    
                if (self.optimizer=='sgd'):
                    self.network=NeuralNetwork( x=x, y=y, lr = lr,  epochs =epochs,batch=1,HiddenLayerNuron=HiddenLayerNuron,activation=activation)
                    
                if (self.optimizer=='mgd'):
                    self.network=MomentumGradiantDecent( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate)
                    
         
                if(self.optimizer=='nag'):
                     self.network=NAG( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation)
                
                if(self.optimizer=='adam'):
                    self.network=Adam( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate,beta1=beta1,beta2=beta2)
                    
                if(self.optimizer=='nadam'):
                    self.network=NAdam( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate,beta1=beta1,beta2=beta2)
                
                if(self.optimizer=='rms'):
                     self.network=RmsProp( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate)
                    
                
    def train(self,b=0.1,eta=0.1,gamma=.9,batch=0,beta1=0.9,beta2=0.99,beta=0.9):
       
        if (self.optimizer=='gd' or self.optimizer=='sgd'):
            self.network.train(batch)
            
        if (self.optimizer=='mgd'):
            self.network.train()
        
        if(self.optimizer=='nag'):
            self.network.train(batch)
        
        if(self.optimizer=='rms'):
            self.network.train(beta)
                
        if(self.optimizer=='adam'):
                    self.network.train(beta1=beta1,beta2=beta2,batch=batch)
                
        if(self.optimizer=='nadam'):
                    self.network.train(beta1=beta1,beta2=beta2,batch=batch)
    def predict(self,x,y):
        prediction=self.network.predict(x)
        prediction=y-prediction
        accurecy=sum(x==0 for x in prediction)
        accurecy=accurecy/len(prediction)
        print('Test accuracy='+str(accurecy*100))
        return prediction
    