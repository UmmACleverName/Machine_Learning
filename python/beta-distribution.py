import numpy as np
import matplotlib.pyplot as plt 




def gamma_function(t):
    a  = 0
    b  = 50.0
    x  = np.linspace(a,b,100)
    Sum = 0.0
    for i in range(0,len(x)-1):
        delta = abs(x[i+1] - x[i])
        xi = x[i]
        Sum += ( xi**(t-1) * np.exp(-xi) ) * delta

    return Sum 
    




def beta_distribution(x,a,b):

    F_a  = gamma_function(a)
    F_b  = gamma_function(b)
    F_ab = gamma_function(a+b)

    coef = F_ab / (F_a * F_b)

    return coef * ( x**(a - 1.0) ) * ( (1.0 - x)**(b - 1) ) 










N1 = 3 
N0 = 17 
a  = 2.0
b  = 2.0 
theta = np.linspace(0,1.0,100)

lik = []
pro = []
pos = []

for theta_i in theta:
    N = N1 + N0
    lik.append(beta_distribution(theta_i,N1,N0)) 
    pro.append(beta_distribution(theta_i,a,b)) 
    pos.append(beta_distribution(theta_i,N1 + a, N0 + b)) 





plt.plot(theta, lik ,'green')
plt.plot(theta, pro ,'red')
plt.plot(theta, pos ,'black')

plt.title("Beta-Binomial Model") 
plt.xlabel("theta range")
plt.legend(["lik","prior","post"], loc='upper right') 


plt.show() 











