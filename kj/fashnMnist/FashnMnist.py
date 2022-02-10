import numpy as np
import pandas as pd
import sys
from fashnMnist.NeuralNetwork import NeuralNetwork
from fashnMnist.MomentumGradiantDecent import MomentumGradiantDecent
import matplotlib.pyplot as plt

class FashnMnist:
    def __init__(self,x, y,lr = .5,epochs =100,batch=500,HiddenLayerNuron=[60,10],activation='tanh',optimizer='GradiantDecent'):
                self.network=None
                # invoking the __init__ of the parent class 
                self.y=y
                self.optimizer=optimizer
                if (self.optimizer=='GradiantDecent'):
                    self.network=NeuralNetwork( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation)
                    
                if (self.optimizer=='MomentumGradiantDecent'):
                    self.network=MomentumGradiantDecent( x=x, y=y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation)
                    
                
                
    def train(self,b=0.1,eta=0.1,gamma=.9,batch=0):
       
        if (self.optimizer=='GradiantDecent'):
            self.network.train(batch)
            
        if (self.optimizer=='MomentumGradiantDecent'):
            self.network.TrainWithMomentumGradientDescent(b,eta,gamma)
        
    
    def predict(self,x,y):
        prediction=self.network.predict(x)
        prediction=y-prediction
        accurecy=sum(x==0 for x in prediction)
        accurecy=accurecy/len(prediction)
        print('Test accuracy='+str(accurecy*100))
        return prediction
    