import numpy as np
import sys
from fashnMnist.NeuralNetwork import NeuralNetwork

class RmsProp(NeuralNetwork):
    def __init__(self, x, y, lr = .5,  epochs =100,batch=100,HiddenLayerNuron=[60,10],activation='sigmoid'
                 ):
                
          
                # invoking the __init__ of the parent class 
                NeuralNetwork.__init__(self, x, y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation)
                
                
          
    def TrainWithRMSProp(self,beta=0.9,batch=0):
        #initialize all parameters
        if batch!=0:
            self.batch=batch
        print('Starting RMSProp')
        print('.....................................')
        v_w,v_b  = self.DW, self.DB
        for epoch in range(self.epochs):
            
            self.resetWeightDerivative()
            
            for i in range(0, self.x.shape[0], self.batch):
                self.xBatch =self.x[i:i+self.batch]
                self.yBatch  = self.y[i:i+self.batch]
                pred=self.feedforward()
                self.backprop()
                
            #Update parameter and return new v_w and v_b
            v_w,v_b=self.updateParamWithrms( beta,v_w,v_b,epoch+1)
               
            #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
           
            pred=self.feedforward()
            
            acc=self.accurecy(pred,self.yBatch)
            loss=self.calculateLoss()
            
            
            #print details 
            self.printDetails(epoch,self.epochs,acc,loss)
        print('Completed')
        print('.....................................')
        
    def updateParamWithrms(self, beta,v_w,v_b,epoch): 
        totalLayer=len(self.HiddenLayerNuron)
        
        betaDash=(1-beta)
        
        eps=.0001#small number
        for i in range(totalLayer):
           
            vw= (beta*v_w[i])+(betaDash* np.square(self.DW[i]))
            vb= (beta*v_b[i])+(betaDash* np.square(self.DB[i]))
            vw1= np.sqrt(vw+eps)
            vb1= np.sqrt(vb+eps)
            self.W[i] = self.W[i] - (self.lr/vw1)*(self.DW[i] )
            self.b[i] = self.b[i] - (self.lr/vb1)* (self.DB[i] )
            
            v_w[i]=vw1
            v_b[i]=vb1
        
        return v_w,v_b       
   
    
            