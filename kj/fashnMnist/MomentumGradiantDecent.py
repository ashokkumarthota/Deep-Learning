import numpy as np
import sys
from fashnMnist.NeuralNetwork import NeuralNetwork
class MomentumGradiantDecent(NeuralNetwork):
    def __init__(self, x, y, lr = .5,  epochs =100,batch=500,HiddenLayerNuron=[60,10],activation='tanh',decay_rate=0.01 
                ):
          
                # invoking the __init__ of the parent class 
                NeuralNetwork.__init__(self, x, y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate )
                
                
          
    def train(self):
        #initialize all parameters
        
        print('Starting Momentum Gradient Descent')
        print('.....................................')
        v_w, v_b  = self.DW, self.DB
        prevacc=0
        for epoch in range(self.epochs):
            #reset weight derivatives and shuffle data
            self.resetWeightDerivative()
            self.shuffle()
            #control momentum
            gamma=self.momentumUpdate(epoch+1)
            
            for i in range(0, self.x.shape[0], self.batch):
                self.xBatch =self.x[i:i+self.batch]
                self.yBatch  = self.y[i:i+self.batch]
                pred=self.feedforward()
                self.backprop()
            prevW=self.W
            prevB=self.b
            prevvw=v_w
            prevvb=v_b    
            #Update parameter and return new v_w and v_b
            v_w, v_b=self.updateParamForMomentumGradientDescent(v_w, v_b,epoch)
                   
            #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
            pred=self.feedforward()
            acc=self.accurecy(pred,self.y)
            loss=self.calculateLoss()
            if(acc<prevacc):
                self.lr= self.lr*(1-self.decay_rate)
                self.W=prevW
                self.b=prevB
                acc=prevacc
                v_w=prevvw
                v_b=prevvb
              
            else:
                prevacc =acc
            
            #print details 
            self.printDetails(epoch,self.epochs,acc,loss)
        print()
        print('Completed')
        print('.....................................')
        
    def updateParamForMomentumGradientDescent(self,v_w,v_b,epoch): 
        totalLayer=len(self.HiddenLayerNuron)
        gamma=self.getGamma(epoch)
        for i in range(totalLayer):
            v_w[i]= gamma*v_w[i]+(self.lr)* self.DW[i]  
            v_b[i]= gamma*v_b[i]+(self.lr)* self.DB[i]
            self.W[i] = self.W[i] - (self.lr)*v_w[i]
            self.b[i] = self.b[i] - (self.lr)* v_b[i]
            
        return v_w,v_b
   
    def getGamma(self,epoch):
        x=np.log((epoch/250)+1)
        x=-1-1*(x)
        x=2**x
        x=1-x
        return min(x,.9)
            