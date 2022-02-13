import numpy as np
import sys
from fashnMnist.NeuralNetwork import NeuralNetwork

class NAG(NeuralNetwork):
    def __init__(self, x, y, lr = .5,  epochs =100,batch=100,HiddenLayerNuron=[60,10],activation='sigmoid',
                 beta=0.9):
                
          
                # invoking the __init__ of the parent class 
                NeuralNetwork.__init__(self, x, y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation)
                
                
          
    def train(self,gamma=0.8,batch=0):
        totalLayer=len(self.HiddenLayerNuron)
        
        #initialize all parameters
        if batch!=0:
            self.batch=batch
        print('Starting NAG')
        print('.....................................')
        prev_w,prev_b  = self.DW, self.DB
        for epoch in range(self.epochs):
            gamma= self.momentumUpdate(epoch+1)
            
            self.resetWeightDerivative()
            vw=self.W
            vb=self.b
            for w in range(totalLayer):
                self.W[w]=self.W[w]-gamma*prev_w[w]
                self.b[w]=self.b[w]-gamma*prev_b[w]
                
            for i in range(0, self.x.shape[0], int(self.batch)):
                    #update weight and bias
                
                
                self.xBatch =self.x[i:i+self.batch]
                self.yBatch  = self.y[i:i+self.batch]
                pred=self.feedforward()
                self.backprop()
                
            self.W=vw
            self.b=vb
                    
            #Update parameter and return new v_w and v_b
            prev_w,prev_b=self.updateParamWithNAG( gamma,prev_w,prev_b)
               
            #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
           
            pred=self.feedforward()
            
            acc=self.accurecy(pred,self.yBatch)
            loss=self.calculateLoss()
            
            
            #print details 
            self.printDetails(epoch,self.epochs,acc,loss)
        print()
        print('Completed')
        print('.....................................')
        
    def updateParamWithNAG(self, gamma,prev_w,prev_b): 
        totalLayer=len(self.HiddenLayerNuron)
        
       
        for i in range(totalLayer):
           
            vw= gamma*(prev_w[i])+(self.lr)*(self.DW[i] )
            vb= gamma*(prev_b[i])+(self.lr)*(self.DB[i] )
            
          
            self.W[i] = self.W[i] - vw
            self.b[i] = self.b[i] - vb
            
            prev_w[i],prev_b[i]=vw,vb
        
        return prev_w,prev_b       
   
    
            