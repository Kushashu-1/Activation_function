import numpy as np

## input
def take_range(x_from,to):
    x  = [i for i in np.arange(x_from ,to ,0.5)]
    return x
## Activation function 
def hard_tanh(x):
  if x<-1:
    return -1
  elif x>=-1 and x<=1:
    return x
  return 1

def hard_sigmoid(x):
  return max(0,min(1,(x+1)/2))

def sigmoid(x):
  return 1/(1+np.exp(-1*x))

def Selu(x):
  return x*sigmoid(x)

def swish(x , b):
  if b=='1':
    return x*(sigmoid(1*x))
  elif b=='2':
    return x*(sigmoid(2*x))
  else:
    return x*(sigmoid(3*x))

def expo_linear_unit(x):
  # Considering linear constant (a = 3)
  if x>=0:
    return x
  else:
    return 3*(np.exp(x)-1)

def linear_function(x):
  # Considering linear constant = 4
  return 4*x

def binary_step(x):
  if x<0:
    return 0
  else:
    return 1

def leaky_relu(x):
    if x>0:
      return x
    else:
      return 0.01*x

def soft_max(x):
  p = [np.exp(i) for i in x]
  total = sum(p)
  res = [i/total for i in p]
  return res


def tanh(x):
  p = np.exp(x)
  q = np.exp(-x)
  if p-q==0:
    return 0;
  return (p+q)/(p-q)

def relu(x):
  if x>0:
    return x
  else:
    return 0


## Y_range
def generate_y(selected_function, x,b):
    if selected_function == 'Exponential Linear Unit':
        y = [expo_linear_unit(i) for i in x]
    elif selected_function == 'Linear function':
        y = [linear_function(i) for i in x]
    elif selected_function == 'Binary function':
        y = [binary_step(i) for i in x]
    elif selected_function == 'Softmax function':
        y = soft_max(x)
    elif selected_function == 'Leaky Relu function':
        y = [leaky_relu(i) for i in x]
    elif selected_function == 'Relu function':
        y = [relu(i) for i in x]
    elif selected_function == 'Sigmoid function':
        y = [sigmoid(i) for i in x]
    elif selected_function == 'Hard Tanh':
        y = [hard_tanh(i) for i in x]
    elif selected_function == 'Hard Sigmoid':
        y = [hard_sigmoid(i) for i in x]
    elif selected_function == 'Swish function':
        y = [swish(i,b) for i in x]
    elif selected_function == 'Silu function':
        y = [Selu(i) for i in x]
    elif selected_function == 'Tanh function':
        y = [tanh(i) for i in x]
    return y
    


