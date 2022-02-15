import numpy as np
import sys
sys.path.append('../fashnMnist/')
from fashnMnist.NeuralNetwork import NeuralNetwork

class RmsProp(NeuralNetwork):
    def __init__(self, x, y, lr = .5,decay_rate=0.01 , epochs =100,batch=100,HiddenLayerNuron=[60,10],activation='tanh',beta=.9,initializer='he'
                 ):
                
          
                # invoking the __init__ of the parent class 
                NeuralNetwork.__init__(self, x, y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,decay_rate=decay_rate,beta=beta,initializer=initializer)
                
                
          
    def train(self):
        
        #initialize all parameters
        
        print('Starting RMSProp')
        print('.....................................')
        v_w,v_b  = self.DW, self.DB
        prevacc=0
        Timenochange=0
        for epoch in range(self.epochs):
            
            self.resetWeightDerivative()
           
            beta=self.momentumUpdate(epoch+1)
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
            v_w,v_b=self.updateParam( beta,v_w,v_b)
               
            #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
           
            pred=self.feedforward()
            
            acc=self.accurecy(pred,self.yBatch)
            loss=self.calculateLoss()
            
            
            if(acc<prevacc):
                self.W=prevW
                self.b=prevB
                acc=prevacc
                v_w=prevvw
                v_b=prevvb
                Timenochange+=1
                if(Timenochange>=5):
                    self.lr= self.lr/2
                    Timenochange-=1
                else:
                    self.lr= self.lr*(1-self.decay_rate)
                 
            else:
                Timenochange=0
                prevacc =acc
            #print details 
            self.printDetails(epoch,self.epochs,acc,loss)
        print()
        print('Completed')
        print('.....................................')
        
    def updateParam(self, beta,v_w,v_b): 
        totalLayer=len(self.HiddenLayerNuron)
        
        betaDash=(1-beta)
        
        eps=.00001#small number
        for i in range(totalLayer):
           
            vw= (beta*v_w[i])+(betaDash* np.square(self.DW[i]))
            vb= (beta*v_b[i])+(betaDash* np.square(self.DB[i]))
            vw1= np.sqrt(vw+eps)
            vb1= np.sqrt(vb+eps)
            self.W[i] = self.W[i] - (self.lr/vw1)*(self.DW[i] )
            self.b[i] = self.b[i] - (self.lr/vb1)* (self.DB[i] )
            
            v_w[i]=vw
            v_b[i]=vb
        
        return v_w,v_b       
   
    
            