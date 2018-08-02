# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 01:07:36 2018

@author: Sachin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
nc=np.size(dataset,axis=1)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,nc-1:nc].values

x1=X
y1=y
iteration=1500
alpha=0.01

sm=np.sum(X,axis=0) #summation for a column
m=np.size(X,axis=0) #number of rows.
n=np.size(X,axis=1) #number of columns.

#feature scaling
X[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()  

y[:,0]=(y[:,0]-y[:,0].mean())/y[:,0].std()    

X=np.append(arr=np.ones((m,1)),values=X,axis=1)


      
def cost_func(X,y,theta,m):            #cost function
    hyp=X*theta.T                     
    error=hyp-y
    cost=np.sum(np.power(error,2))/(2*m)
    return cost

def grad(X, y,alpha,iteration):        #gradient descent.
    m=np.size(X,axis=0)                #number of rows.
    n=np.size(X,axis=1)                #number of columns.
    theta=np.matrix([0.0 for i in range(n)]) #initialise theta.
    cost=[0.0 for i in range(iteration)]     #initialize cost 
    for j in range(iteration):
        hyp=X*theta.T
        error=hyp-y 
        delta=np.sum(np.multiply(error,X),axis=0) 
        theta=theta-(alpha/m)*delta
        cost[j]=cost_func(X,y,theta,m)
    return theta,cost,delta

theta,cost,delta=grad(X,y,alpha,iteration)    

cost=np.asarray(cost)
it=np.matrix([[i] for i in range(iteration)])


"""To check whether the model is correct. we need to plot cost vs iterations,
where we observe that it decreases for every iteration."""
plt.plot(it,cost)
plt.title('Cost vs iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.show()
    
plt.scatter(X[:,1],y,color='red')
plt.plot(X[:,1],X*theta.T,color='blue')
plt.title('Linear Regression')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show() 





