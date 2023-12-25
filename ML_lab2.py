dict = {'color': 'green', 'points': 5} #Creare dictionar

print(dict['color']) #afisare valoare asociata cu cheia 'color'
print(dict['points']) #afisare valoare asociata cu cheia 'points'

dict['x'] = 25 #adaugam o pereche noua cheie-valoare
dict['speed'] = 1.5 #adaugam o pereche noua cheie-valoare

dict['color'] = 'yellow' #modifica valoarea p/u cheia 'color'
dict['points'] = 10 #modifica valoarea p/u cheia 'points'

del dict['points'] #sterge perechea asociata cu cheia 'points'

for k, v in dict.items():
    print(k + ": " + str(v)) #parcurgerea si afisarea prechelor cheie-valoare dictionarului

for key in dict.keys():
    print(key) #parcurgerea si afisarea cheiilor dictionarului

for value in dict.value():
    print(value) #parcurgerea si afisarea valorilor dictionarului

for key in sorted(dict.keys()):
    print(key) #parcurgerea si afisarea cheiilor dict in ordine alfabetica

num_responses = len(dict) #declararea unei variabile cu valoarea lungimii dict

list = [] #declaram o lista nula
list.append(dict) #copiam elementele dict in lista

dict = {'color': ['green', 'red'], 'points': [5, 6]} #salvarea unei liste in dictionar

#-------------------------------------

'''100 numpy exercises'''
#### 1. Import the numpy package under the name `np` (★☆☆)
import numpy as np

#### 2. Print the numpy version and the configuration (★☆☆)
print(np.__version__)
np.show_config()

#### 3. Create a null vector of size 10 (★☆☆) 
v = np.zeros(10)

#### 4.  How to find the memory size of any array (★☆☆) 
print("%d" % (v.size * v.itemsize)) 

#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆) 
print(np.info(np.add))

#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) 
a = np.zeros(10)
a[4] = 1
print(a)

#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆) 
a = np.arange(10,50)
print(a)

#### 8.  Reverse a vector (first element becomes last) (★☆☆) 
a[::-1]

#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) 
a = np.arange(0,9)
a.reshape(3,3)

#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆) 
a = np.array([1,2,0,0,4,0])
print(np.nonzero(a))

#### 11. Create a 3x3 identity matrix (★☆☆) 
m = np.eye(3)
m

#### 12. Create a 3x3x3 array with random values (★☆☆) 
a = np.random.random((3,3,3))
a

#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) 
a = np.random.random((10,10))
print("min = %f and max = %f" % (a.min(), a.max()))

#### 14. Create a random vector of size 30 and find the mean value (★☆☆) 
a = np.random.random(30)
print(a.mean())

#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) 
a = np.ones((5,5))
a[1:-1, 1:-1] = 0
a

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆) 
a = np.pad(a, pad_width = 1, constant_values = 0)
a

#### 17. What is the result of the following expression? (★☆☆) 
0 * np.nan #nan
np.nan == np.nan #false
np.inf > np.nan #false
np.nan - np.nan #nan
np.nan in set([np.nan]) #True
0.3 == 3 * 0.1 #False

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) 
m = np.diag(1+np.arange(4),k=-1)
m

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) 
m = np.zeros((8,8))
m[1::2,::2] = 1
m[::2,1::2] = 1
m

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? 
print(np.unravel_index(100, (6,7,8)))




