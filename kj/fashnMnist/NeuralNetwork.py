import numpy as np
class NeuralNetwork:
    def __init__(self, x, y, lr = .5,  epochs =100,batch=0,HiddenLayerNuron=[60,10],activation='tanh' ):
       
        
        self.HiddenLayerNuron=HiddenLayerNuron
        self.x = x
        self.y = y
        
        #define batch size
        self.batch=self.x.shape[0]
        if (batch!=0):
            self.batch=batch
            
        self.epochs = epochs
        self.lr = lr
        self.activation=activation
        self.loss = []
        self.acc = []
        self.batch=batch
        self.init_weights()
        self.xBatch=x
        self.yBatch=y
         
      
    
    def init_weights(self):
        self.W=[]
        self.b=[]
        self.DW=[]
        self.DB=[]
        prevInput=self.x.shape[1]
        for i in self.HiddenLayerNuron:
            self.W.append(np.random.randn(prevInput ,i))
            self.DW.append(np.zeros((prevInput ,i)))
            prevInput=i
            self.b.append(np.random.randn(i))
            x=np.zeros(i)
            self.DB.append(x)
         
   


    def resetWeightDerivative(self):
        self.DW=[]
        self.DB=[]
        prevInput=self.x.shape[1] 
        for i in self.HiddenLayerNuron:
            self.DW.append(np.zeros((prevInput ,i)))
            x=np.zeros(i)
            self.DB.append(x)
            prevInput=i
            
       
    
            
    def sig(self,x):
          return  1/(1+np.exp(-x))

    # Sigmoidal derivative
    def dsig(self,x):
          return self.sig(x) * (1- self.sig(x))
        
        
    def reLU(self, x):
        #return np.maximum(0,x)
        return  np.where(x < 0, 0, x)
    
    def dReLU(self,x):
        return 1 * (x > 0) 

    
    
    
    def tanh(self, x):
        return np.tanh(x)
    
    def dtanh(self,x):
        tanh_x = self.tanh(x)
        return (1 - np.square(tanh_x))
    
    
    
    
    def softmax(self, z):   
        z=np.exp(z)
        tmp=np.sum(z, axis = 1) 
        for i in range(z.shape[0]):       
            z[i]=z[i]/tmp[i]
        return z
    
    def feedforward(self):
        self.z=[]
        self.a=[]
        self.yHat=[]
        totalLayer=len(self.HiddenLayerNuron)
        x=self.xBatch
        for i in range(totalLayer):
            self.z.append(x.dot(self.W[i]) + self.b[i]) 
            if(self.activation=='sigmoid'):
                self.a.append(self.sig(self.z[i]))
            else:
                if(self.activation=='relu'):
                    self.a.append(self.reLU(self.z[i]))
                else:
                    if(self.activation=='tanh'):
                        self.a.append(self.tanh(self.z[i]))
                    else:
                        self.a.append(self.z[i])
            x=self.a[i]  
        self.yHat=self.softmax(x)
        
        return self.yHat
        
        
    def backprop(self):
        totalLayer=len(self.HiddenLayerNuron)
         
        self.error = self.yHat - self.yBatch
        dcost = (1/self.x.shape[0])*self.error
        
   
        prevLayerDW=dcost
        i=totalLayer-1
        while( i>=0):
            
           
            x=[]
            
            #get input of current hidden layer
            if(i==0):
                x=self.xBatch
            else:
                x=self.a[i-1]
            t=np.dot(x.T,prevLayerDW)
           
            self.DB[i]+= np.sum(prevLayerDW,axis = 0)
            self.DW[i]+=t
            
            if (i>0):
                prevLayerDW=np.dot(prevLayerDW,self.W[i].T)
                 
                if(self.activation=='sigmoid'):
                    tmp=prevLayerDW*self.dsig(self.z[i-1])
                    prevLayerDW=tmp
                if(self.activation=='relu'):
                    tmp=prevLayerDW*self.dReLU(self.z[i-1])
                    prevLayerDW=tmp
                    
                if(self.activation=='tanh'):
                    tmp=prevLayerDW*self.dtanh(self.z[i-1])
                    prevLayerDW=tmp
                 

            i-=1
        
        #self.DW.reverse()
        #self.DB.reverse()
        
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
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]
        
        
    def calculateLoss(self):
        self.loss=[]
        self.error = (-1)*np.log(self.yHat)*self.yBatch
       
        self.error=np.sum(self.error,axis=1)
        
        loss=np.mean(self.error)
        return loss
    
    def train(self,batch=0):
        print('.....................................')
        print('Starting Gradient Descent..')
        print('.....................................')
        if batch!=0:
            self.batch=batch
        for epoch in range(self.epochs):
            
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
        print('Completed') 
        print('.....................................')
        
    def printDetails(self,epoch,totalepoch,acc,loss):
        if((epoch+1)%10==0):
            print('\r steps={}/{} , Accuraacy ={} ,Loss={}'.format((epoch+1),totalepoch,round(acc, 2) , round(loss,5))) 
        else:
             print('\r steps={}/{} , Accuraacy ={} ,Loss={}'.format((epoch+1),totalepoch,round(acc, 2) , round(loss,5)),end =" ") 
       
              