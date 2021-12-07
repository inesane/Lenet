#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
import json
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
from matplotlib import pyplot as plt


# In[2]:


f = open('model_parameters.json',)
   
para = json.load(f)

f.close()

A = para['A']
S = para['S']
S2C3 = para['S2C3']
S2C3 = {int(k):[int(i) for i in v] for k,v in S2C3.items()}
conv = para['conv']
bConv = para['bConv']
w1 = para['w1']
b1 = para['b1']
w2 = para['w2']
b2 = para['b2']


# In[3]:


def avgPool(im, k):
    ret = np.zeros(((len(im)//2), len(im[0])//2))
    for i in range(len(ret)):
        for j in range(len(ret[0])):
            ret[i][j] = np.mean(im[i*2:i*2+k, j*2:j*2+k])
    return ret

def convolve(im,kernel,bias=0):
    kernel_size = np.array(np.array(kernel).shape)
    im_size = np.array(np.array(im).shape)
#     print(im_size, kernel_size, np.array([1,1]))
    final_dim = im_size - kernel_size + np.array([1,1])
    final_im = np.zeros(tuple(final_dim))
    for i in range(final_dim[0]):
        for j in range(final_dim[1]):
            final_im[i][j]= np.sum(np.multiply(im[i:i+kernel_size[0],j:j+kernel_size[1]],kernel))
    final_im += bias
    return final_im

def normalize(im):
    return np.where(im==0,-0.1,1.175)

def preprocess(im):
    return normalize(np.pad(im, (2, 2)))

def forward(im):
    global ret1
    global ret2
    global ret3
    global ret4
    global ret5

    # Convolution 1
    ret1 = []
    for i in range(len(conv[0])):
        ret1.append(A*np.tanh(S * convolve(im, conv[0][i], bConv[0][i])))
        
    # Average pooling
    for i in range(len(ret1)):
        ret1[i] = avgPool(ret1[i], 2)
        
    # Convolution 2
    ret2 = []
    for i in range(len(conv[1][0])):
        tmp = np.zeros((len(ret1[0]) - 2*(len(conv[1][0][i])//2), len(ret1[0][0]) - 2*(len(conv[1][0][i])//2)))
        for j in S2C3[i]:
            tmp += convolve(ret1[j], conv[1][j][i])
        tmp +=  bConv[1][i]
        ret2.append(A*np.tanh(S * tmp))
        
    # Average pooling
    for i in range(len(ret2)):
        ret2[i] = avgPool(ret2[i], 2)
        
    # Convolution 3
    ret3 = []
    for i in range(len(conv[2][0])):
        tmp = np.zeros((len(ret2[0]) - 2*(len(conv[2][0][i])//2), len(ret2[0][0]) - 2*(len(conv[2][0][i])//2)))
        for j in range(len(conv[2])):
            tmp += convolve(ret2[j], conv[2][j][i])
        tmp +=  bConv[2][i]
        ret3.append(A*np.tanh(S * tmp))
    
    # Fully connected 1
    ret4 = []
    for i in range(len(w1[0])):
        tmp = 0
        for j in range(len(w1)):
            tmp += w1[j][i]*ret3[j]
        tmp += b1[i]
        ret4.append(A*np.tanh(S * tmp))

    # Fully connected 2
    ret5 = []
    for i in range(len(w2[0])):
        tmp = 0
        for j in range(len(w2)):
            tmp += w2[j][i]*ret4[j]
        tmp += b2[i]
        ret5.append(tmp)
    
    # Softmax activation
    expSum = np.sum(np.exp(ret5))
    ret5 = np.exp(ret5)/expSum
    return ret5


# In[4]:


temp = np.array(0)
temp2 = np.array(0)
temp3 = np.array(0)
max_index = 10

def loadImage(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) / 255

def openfilename():
    filename = filedialog.askopenfilename(title ='"pen')
    return filename

def open_img():
    global temp
    x = openfilename()
 
    temp = loadImage(x)
    img = Image.fromarray(np.uint8(temp*255))
    
    img = img.resize((28, 28), Image.ANTIALIAS)
 
    img = ImageTk.PhotoImage(img)
  
    panel = Label(root, image = img)
     
    panel.image = img
    panel.grid(row = 2)

def run_img():
    global temp2
    global temp3
    global max_index
    temp2 = preprocess(temp)
    temp3 = forward(temp2)
    ind = temp3.tolist()
    max_index = ind.index((max(ind)))
    
    print(max_index)
    return (max_index)


# In[ ]:


root = Tk()
 
root.title("Image Loader")
 
root.geometry("1200x1200+500+500")
 
root.resizable(width = True, height = True)
 
btn = Button(root, text ='open image', command = open_img).grid(row = 1, columnspan = 4)

btn2 = Button(root, text ='run image', command = run_img).grid(row = 10, columnspan = 4)

root.mainloop()


# In[ ]:




