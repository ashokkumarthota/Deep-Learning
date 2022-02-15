import numpy as np
import numpy as np
from fashnMnist.Initializers import Initializers
from fashnMnist.Activations import Activations
import math
class NeuralNetwork:
    def __init__(self, x, y, lr = .01,  epochs =100,batch=32,HiddenLayerNuron=[32,64,10],activation='tanh' ,decay_rate=0,beta1=0.9,beta2=0.9,beta=0.9,gamma=0.9,initializer='he',weight_decay=0.0001,dropout_rate=0):
        
        self.initializer=initializer
        self.HiddenLayerNuron=HiddenLayerNuron
        self.x = x
        self.y = y
        self.xBatch=x
        self.yBatch=y 
        
        #initialize weights
        self.init_weights()
        
        #initialize hyper parameters
        self.decay_rate=decay_rate        
        self.beta1=beta1
        self.beta2=beta2
        self.epochs = epochs
        self.fixedlr = lr
        self.lr = lr
        self.activation=activation
        self.loss = []
        self.acc = []
        self.batch=batch
        self.dropout_rate=dropout_rate
        self.gamma=gamma
        self.beta=beta
        self.weight_decay=weight_decay
    def stepDecay(self,epoch):
      
        return self.lr * (1.0/(1+self.decay_rate * epoch))
    def momentumUpdate(self,t,maxm=.999):
        x=np.log(np.floor(t/250)+1)/np.log(2)
        x=1-2**(-1-x)
        return min(x,maxm)
    def init_weights(self):
        self.W=[]
        self.b=[]
        self.DW=[]
        self.DB=[]
        init=Initializers(self.initializer)
        bounds=(-0.1, 0.1)
        prevInput=self.x.shape[1]
        for i in self.HiddenLayerNuron:
           
            self.W.append(init.initialize(prevInput ,i))
            self.b.append(init.initialize(i))
            self.DW.append(np.zeros((prevInput ,i)))
            self.DB.append(np.zeros(i))
            prevInput=i
    
    def resetWeightDerivative(self):
        self.DW=[]
        self.DB=[]
        prevInput=self.x.shape[1] 
        for i in self.HiddenLayerNuron:
            self.DW.append(np.zeros((prevInput ,i)))
            x=np.zeros(i)
            self.DB.append(x)
            prevInput=i
            
       
   
    def feedforward(self):
        self.z=[]
        self.a=[]
        self.yHat=[]
        self.D=[]
        totalLayer=len(self.HiddenLayerNuron)
        activation=Activations(self.activation)
       
        x=self.xBatch
        for i in range(totalLayer):
            self.z.append(x.dot(self.W[i]) + self.b[i]) 
            if (i==totalLayer-1):
                 self.a.append(self.z[i])
            else:
                self.a.append(activation.applyActivation(self.z[i]))
                if(self.dropout_rate!=0):
                    dropRate=(1-self.dropout_rate)
                    d= np.random.rand(self.a[i].shape[0], self.a[i].shape[1])
                    d=d<dropRate
                    self.D.append(d)
                    self.a[i]=self.a[i]*d
                    self.a[i]=self.a[i]/dropRate
                    
                    
            x=self.a[i]
                    
            
            
              
        self.yHat=activation.softmax(x)
        
        return self.yHat
        
        
    def backprop(self):
        totalLayer=len(self.HiddenLayerNuron)
         
        self.error = self.yHat - self.yBatch
        dcost = (1/self.x.shape[0])*self.error
        
   
        prevLayerDW=dcost
        i=totalLayer-1
        activation=Activations(self.activation)
        while( i>=0):
            
           
            x=[]
            
            #get input of current hidden layer
            if(i==0):
                x=self.xBatch
            else:
                x=self.a[i-1]
                if(self.dropout_rate!=0):
                    x=x* self.D[i-1]
                    x=x/(1-self.dropout_rate)
                    
                    
            t=np.dot(x.T,prevLayerDW)
            
            self.DB[i]+= np.sum(prevLayerDW,axis = 0)
            self.DW[i]+=t
            
            if (i>0):
                prevLayerDW=np.dot(prevLayerDW,self.W[i].T)
                prevLayerDW=prevLayerDW*activation.applyActivationDeriv(self.z[i-1])
                
               

            i-=1
        
       
    def updateParam(self): 
        totalLayer=len(self.HiddenLayerNuron)
        
           
        for i in range(totalLayer):
            self.W[i] = self.W[i] - (self.lr)* self.DW[i]
            self.b[i] = self.b[i] - (self.lr)* self.DB[i]
        
    def accurecy(self,pred,y):
        y=np.argmax(y, axis=1)
        pred=np.argmax(pred, axis=1)
        tmp=pred-y
        count=np.count_nonzero(tmp == 0)
        acc=100*(count/y.shape[0])
        return acc
         
    def predict(self,x):
        self.xBatch=x
        pred=self.feedforward()
         
        pred=np.argmax(pred, axis=1)
        return pred
    
    def shuffle(self):
        idx = [i for i in range(self.x.shape[0])]
        np.random.shuffle(idx)
        self.x = self.x[idx]
        self.y = self.y[idx]
        
        
    def calculateLoss(self):
        self.loss=[]
        self.error = (-1)*np.log(self.yHat)*self.yBatch
        
        self.error=np.sum(self.error,axis=1)
        loss=np.mean(self.error)
        return loss
    
    def train(self):
        print('.....................................')
        print('Starting Gradient Descent..')
        print('.....................................')
        
        for epoch in range(self.epochs):
            self.shuffle()
            self.resetWeightDerivative()
            
            for i in range(0, self.x.shape[0], self.batch):
                self.xBatch =self.x[i:i+self.batch]
                self.yBatch  = self.y[i:i+self.batch]
                pred=self.feedforward()
                self.backprop()

                #update parameters
            self.updateParam()    
            
            
            #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
            pred=self.feedforward()
            acc=self.accurecy(pred,self.y)
            loss=self.calculateLoss()
            
            
            #print details 
            self.printDetails(epoch,self.epochs,acc,loss)
        print()
        print('Completed') 
        print('.....................................')
        
    def printDetails(self,epoch,totalepoch,acc,loss):
        if((epoch+1)%20==0):
            print('\r steps={}/{} , Accuracy ={} ,Loss={}'.format((epoch+1),totalepoch,round(acc, 2) , round(loss,5))) 
        else:
             print('\r steps={}/{} , Accuraacy ={} ,Loss={}'.format((epoch+1),totalepoch,round(acc, 2) , round(loss,5)),end =" ") 
       
              