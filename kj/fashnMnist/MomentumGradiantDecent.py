import numpy as np
import sys
from fashnMnist.NeuralNetwork import NeuralNetwork
class MomentumGradiantDecent(NeuralNetwork):
    def __init__(self, x, y, lr = .5,  epochs =100,batch=500,HiddenLayerNuron=[60,10],activation=['tanh','tanh']
                ):
          
                # invoking the __init__ of the parent class 
                NeuralNetwork.__init__(self, x, y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation)
                
                
          
    def TrainWithMomentumGradientDescent(self,b=0.1,eta=0.1,gamma=.9,batch=0):
        #initialize all parameters
        if batch!=0:
            self.batch=batch
        print('Starting Momentum Gradient Descent')
        print('.....................................')
        v_w, v_b  = self.DW, self.DB
        for epoch in range(self.epochs):
            
            self.resetWeightDerivative()
            
            for i in range(0, self.x.shape[0], self.batch):
                self.xBatch =self.x[i:i+self.batch]
                self.yBatch  = self.y[i:i+self.batch]
                pred=self.feedforward()
                self.backprop()
                
            #Update parameter and return new v_w and v_b
            v_w, v_b=self.updateParamForMomentumGradientDescent(v_w, v_b,eta,gamma)
                   
            #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
            pred=self.feedforward()
            acc=self.accurecy(pred,self.y)
            loss=self.calculateLoss()
            
            
            #print details 
            self.printDetails(epoch,self.epochs,acc,loss)
        print('Completed')
        print('.....................................')
        
    def updateParamForMomentumGradientDescent(self,v_w,v_b,eta,gamma): 
        totalLayer=len(self.HiddenLayerNuron)
        for i in range(totalLayer):
            v_w[i]= gamma*v_w[i]+eta* self.DW[i]  
            v_b[i]= gamma*v_b[i]+eta* self.DB[i]
            self.W[i] = self.W[i] - (self.lr)*v_w[i]
            self.b[i] = self.b[i] - (self.lr)* v_b[i]
            
        return v_w,v_b
   
    
            