import numpy as np
import sys
from fashnMnist.NeuralNetwork import NeuralNetwork
class Adam(NeuralNetwork):
    def __init__(self, x, y, lr = .5,  epochs =100,batch=32,HiddenLayerNuron=[32,10],activation='tanh',beta1=0.9,beta2=0.99,decay_rate=0):
          
                # invoking the __init__ of the parent class 
                NeuralNetwork.__init__(self, x, y, lr = lr,  epochs =epochs,batch=batch,HiddenLayerNuron=HiddenLayerNuron,activation=activation,beta1=beta1,beta2=beta2,decay_rate=decay_rate)
                
                
          
    def train(self,beta1=0.9,beta2=0.99,batch=0):
        #initialize all parameters
        if batch!=0:
            self.batch=batch
        print('Starting Adam')
        print('.....................................')
        m_w,v_w,m_b, v_b  = self.DW, self.DW, self.DB, self.DB
        prevacc=0
        for epoch in range(self.epochs):
            #store history
            prevBeta1=beta1
            prevBeta2=beta2
            #control momentum
            beta1=self.momentumUpdate(epoch+1)
            beta2=self.momentumUpdate(epoch+1)
            
            
            #reset all derivatives
            self.resetWeightDerivative()
            
            for i in range(0, self.x.shape[0], self.batch):
                
                self.xBatch =self.x[i:i+self.batch]
                self.yBatch  = self.y[i:i+self.batch]
                
                
                pred=self.feedforward()
               
                self.backprop()
                
           
            #save history
            prev_m_w= m_w
            prev_v_w=v_w
            prev_m_b=m_b
            prev_v_b =v_b
            prevW=self.W
            prevB=self.b
            m_w,v_w,m_b, v_b =self.updateParamWithAdam( m_w,v_w,m_b, v_b ,beta1,beta2,(epoch+1))
           
            #verify loss after each epoch
            self.xBatch = self.x
            self.yBatch  =self.y
            pred=self.feedforward()
            acc=self.accurecy(pred,self.y)
            loss=self.calculateLoss() 
            
            if(acc<prevacc):
                #reset to old histry as accurecy dropping
                beta1=prevBeta1
                beta2=prevBeta2
                m_w=prev_m_w
                v_w=prev_v_w
                m_b=prev_m_b
                v_b=prev_v_b 
            
                self.W=prevW
                self.b=prevB
                acc=prevacc
                
                self.lr= self.lr*(1-self.decay_rate)
                 
            else:
                
                prevacc =acc
                 
           
            #print details       
            self.printDetails(epoch,self.epochs,acc,loss)
           
        print()
        print('Completed')
        print('.....................................')
   
        
    def updateParamWithAdam(self, m_w,v_w,m_b, v_b ,beta1,beta2,epoch): 
        totalLayer=len(self.HiddenLayerNuron)
        
        beta1Hat=1.0-(beta1**epoch)
        beta2Hat=1.0-(beta2**epoch)
        
        beta1Dash=(1.0-beta1)
        beta2Dash=(1.0-beta2)
        eps=.00001#small number
        
        newvw=[]
        newvb=[]
        newmw=[]
        newmb=[]
        for i in range(totalLayer):
            vw1= np.multiply(v_w[i],beta2)
            vw2=np.square(self.DW[i])
            vw2= np.multiply(vw2 ,beta2Dash)
            vw3=np.add(vw1,vw2)
           
            vb1=np.multiply(v_b[i],beta2)
            vb2=np.square(self.DB[i])
            vb2=np.multiply(vb2 ,beta2Dash)
            
            vb3=np.add(vb1,vb2)
           

            mw1= np.multiply(m_w[i],beta1)
            mw2= np.multiply(self.DW[i] ,beta1Dash)
            mw=np.add(mw1,mw2)
            mb1= np.multiply(m_b[i],beta1)
            mb2= np.multiply(self.DB[i] ,beta1Dash)
            mb=np.add(mb1,mb2)

             #bias correction
            vw= vw3/beta2Hat
            vb= vb3/beta2Hat 
            mw= mw/beta1Hat
            mb= mb/beta1Hat
            
            newvw.append(vw)
            newvb.append(vb)
            newmw.append(mw)
            newmb.append(mb)
            
            vw= np.sqrt(vw)+eps
            vb= np.sqrt(vb)+eps
            
           
            self.W[i] = self.W[i] - (self.lr/vw)*(mw)
            self.b[i] = self.b[i] - (self.lr/vb)* (mb)
         
        return newmw,newvw,newmb,newvb
   
    
            