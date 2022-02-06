import numpy as np
class NeuralNetwork:
    def __init__(self, x, y, lr = .1,  epochs =300,HiddenLayerNuron=[60,10],activation=['identity','identity']):
       
        self.HiddenLayerNuron=HiddenLayerNuron
        self.x = x 
        self.y = y
        self.batch= self.x.shape[0]
        self.epochs = epochs
        self.lr = lr
        self.activation=activation
        self.loss = []
        self.acc = []
        
        self.init_weights()
    
     
    
    def init_weights(self):
        self.W=[]
        self.b=[]
        prevInput=self.x.shape[1]
        for i in self.HiddenLayerNuron:
            self.W.append(np.random.randn(prevInput ,i))
            prevInput=i
            self.b.append(np.random.randn(i))
            
            
    def sig(self,x):
          return  1/(1+np.exp(-x))

    # Sigmoidal derivative
    def dsig(self,x):
          return self.sig(x) * (1- self.sig(x))
        
        
    def ReLU(self, x):
        #return np.maximum(0,x)
        return  np.where(x < 0, 0, x)
    
    def dReLU(self,x):
        return 1 * (x > 0) 

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
        x=self.x
        for i in range(totalLayer):
            self.z.append(x.dot(self.W[i]) + self.b[i]) 
            if(self.activation[i]=='sigmoid'):
                self.a.append(self.sig(self.z[i]))
            else:
                if(self.activation[i]=='relu'):
                    self.a.append(self.ReLU(self.z[i]))
                else:
                    self.a.append(self.z[i])
            x=self.a[i]  
        self.yHat=self.softmax(x)
        
        return self.yHat
        
        
    def backprop(self):
        totalLayer=len(self.HiddenLayerNuron)
         
        self.error = self.yHat - self.y
        dcost = (1/self.x.shape[0])*self.error
        
        self.DW=[]
        self.DB=[]
           
        
    
        
        """i=totalLayer-1
        while( i>=0):
            print("W")
            print(self.W[i].shape)
            print("b")
            print(self.b[i].shape)
            print("z")
            print(self.z[i].shape)
            print("A")
            print(self.a[i].shape)
            print("error")
            print(dcost.shape)
            i-=1"""
            
         
        """
        DW3 = np.dot(dcost.T,self.a2).T
        DW2 = np.dot((np.dot((dcost),self.W3.T) * self.dReLU(self.z2)).T,self.a1).T
        DW1 = np.dot((np.dot(np.dot((dcost),self.W3.T)*self.dReLU(self.z2),self.W2.T)*self.dReLU(self.z1)).T,self.x).T
        db3 = np.sum(dcost,axis = 0)
        db2 = np.sum(np.dot((dcost),self.W3.T) * self.dReLU(self.z2),axis = 0)
        db1 = np.sum((np.dot(np.dot((dcost),self.W3.T)*self.dReLU(self.z2),self.W2.T)*self.dReLU(self.z1)),axis = 0        
        """
        prevLayerDW=dcost
        i=totalLayer-1
        while( i>=0):
            
           
            x=[]
            
            #get input of current hidden layer
            if(i==0):
                x=self.x
            else:
                x=self.a[i-1]
            t=np.dot(x.T,prevLayerDW)
            self.DW.append(t)
            self.DB.append( np.sum(prevLayerDW,axis = 0))
            if (i>0):
                prevLayerDW=np.dot(prevLayerDW,self.W[i].T)
                 
                if(self.activation[i]=='sigmoid'):
                    tmp=prevLayerDW*self.dsig(self.z[i-1])
                    prevLayerDW=tmp
                if(self.activation[i]=='relu'):
                    tmp=prevLayerDW*self.dReLU(self.z[i-1])
                    prevLayerDW=tmp
                 

            i-=1
        
        self.DW.reverse()
        self.DB.reverse()
        
    def UpdateParam(self): 
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
        self.x=x
        pred=self.feedforward()
         
        pred=np.argmax(pred, axis=1)
        return pred
    
    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]

    def train(self):
        for epoch in range(self.epochs):
            pred=self.feedforward()
          
            y=self.y
            
            acc=self.accurecy(pred,y)
            self.backprop()
            self.UpdateParam()
            print('\r steps={}/{} , Accuraacy ={}'.format((epoch+1),self.epochs,round(acc, 2)),end =" ") 
           
                
            