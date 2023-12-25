# -*- coding: utf-8 -*-
import numpy as np
print(np.__version__)
np.show_config()
#Creați un vector aleatoriu de dimensiunea 10 și înlocuiți valoarea maximă cu 0
Z = np.zeros(10)
print(Z)
#Extrageți partea întreagă a unui tablou aleatoriu Z folosind 3 metode diferite Z = np.random.uniform(0,10,10)
Z = np.random.random((3,3,3))
print(Z)
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
Z = np.random.random(30)
m = Z.mean()
print(m)
#Creați o matrice 5x5 cu valori de rând cuprinse între 0 și 4
Z = np.diag(np.arange(1, 5), k=-1)
print(Z)
# Luați în considerare două matrice aleatoare A și B, verificați dacă sunt egale
#A = np.random.randint(0,2,5)
#B = np.random.randint(0,2,5)
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)
#Test train split
import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
X
list(y)
