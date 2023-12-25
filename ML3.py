# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:52:40 2022

@author: Yazka
"""

import numpy as np
# Generate some random data
x_data = np.random.rand(100).astype(np.float32)
x_data
y_data = x_data * 0.1 + 0.3
noise = np.random.normal(scale=0.01, size=(100,))
noise.shape

def chart(x, y):
    """
    This only prints a simple chart 
    """
    #matplotlib inline
    from matplotlib import pyplot as plt
    # Plot outputs
    plt.ylabel("Pretul casei")
    plt.scatter(x, y,  color='black')
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
chart(x_data, y_data + noise)

def gradient(a, b, x, y):
    y_predicted = x * a + b
    error = (y - y_predicted)
    a_ = -(1.0/len(x)) * 2 * np.sum(error * x)
    b_ = -(1.0/len(x)) * 2 * np.sum(error)
    return a_, b_

a, b = np.random.randn(2)

alpha = 0.5
for i in range(100):
    (a_, b_) = gradient(a, b, x_data, y_data)
    a = a - alpha * a_
    b = b - alpha * b_
print(a, b)

def final_plot(x, y, a, b):
    #matplotlib inline
    from matplotlib import pyplot as plt

    # Plot outputs
    plt.ylabel("Pretul casei")
    plt.scatter(x, y,  color='black')
    plt.xticks(())
    plt.yticks(())
    plt.plot([0, -b/a], [b, 0])
    plt.show()

final_plot(x_data, y_data + noise, a, b)