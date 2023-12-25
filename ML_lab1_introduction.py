print('Hello world')

def print_vowels(text):
    vowels = 'aeiouy'
    for char in text:
        if char in vowels:
            print("Found " + char)
    print_vowels('testing')
    
x=3
last_name = 'Muller'
    

x = 'asdf'
type(x)
x= "asdf"
type(x)
x=3
type(x)
x = 3.0
type(x)
x = True
type(x)
 
x = None
type(x)

int('42')

float('42')

str(1.41421)

str(True)

bool(0)

bool(123)

bool('')

bool('asdf')

bool(None)

7+3

7-3
7 / 3

7 // 3 #truncating division

7%3 #modulo operator

7 * 3

7 ** 3 #exponentiation

(3*3 + 4*4) / 5 #usual operator precedence

3 == 3

3!= 3

3 <=3

3 <3

3>2

3 >=2

True or False

True and False

not False

0 or 'asdf'

2 | 6 # or
2 & 6 #and
~2 #not
3^1 #xor

3 if 5 == 5 else 'asdf'
x = 'asdf'
x+x

x*3

x[0]

x[0:2]

x[:2]

x[2:]
    
x[::2]
x[::-1]
x[-1]
x[:-2]
len(x)
'The string "%s" has %d characters.' % (x, len(x))


len(x)
x[:2]

x.append(None)
x  

x[:2] = ['a', 'b'] 
x 

x = (1, 2, 3, 'vier')
x+x

x = 1, 2, 3, 'vier'
print(x)

x = {1: 'eins', 2: 'zwei'}
x[3] = 'three'
x
del x[1]
x       
    
 len(x)
 x = {1, 2, 'vier'}
 2 in x
 
 4 not in x
 
 x.add(3)
x
x.add(1)
x
x & {2, 3, 4, 5}
x = (2, 3)
a, b = x
print(a, b)


b, a = a, b
print(a, b)

x = [1, 2, 3, 4]
a, b, *c = x
print(a, b, c)

x = 3
if x > 3:
    y = 'Greater than 3'
    x = x - 2
elif x > 2:
    y = 'Greater than 2'
    x = x - 1
else:
    y = 'Smaller than 2'
print(y)
 
x = []
if x:
    print(x)
if len(x):
    print(x)
if len(x) != 0:
    print(x)
     
    for x in 0, 1, 2:
        print(2**x)
        
   for x in range(3):
    print(2**x)  
    
    for x in 'asdf':
    print(x)
    
  x = {1: 'one', 2: 'two', 3: 'three'}
for k, v in x.items():
    print('%d --- %s' % (k, v))   
    
    x = {1, 4, 8}
while x:  # as long as x is not empty
    print(x.pop())  # remove some item
    
x = [2, 4, 8]
for item in x:
    if item % 3 == 0:
        print("Found item divisible by 3!")
        break
else:
    print("Did not find any.")

divisor = 0
try:
    y = 12 / divisor
except IOError:
    print("Input/output problem")
except Exception as e:
    print("Oh no: " + e.args[0])
else:
    print("It worked!")
f = open('Python Intro.ipynb', 'r')
try:
    print(f.readline())
finally:
    f.close()
    
    with open('Python Intro.ipynb', 'r') as f:
    print(f.readline())
    
   x = [2**exponent for exponent in range(5)]
print(x) 
x = []
for exponent in range(5):
    x.append(2**exponent)
    x = {key: 2**key for key in range(5)}
print(x)

x = (2**exponent for exponent in range(5))

for value in x:
    print(value)
    
    hypothenuse(b=4, a=3)
    
    def ascii_shift(text, shift=13):
    return ''.join(chr(ord(c) + shift) for c in text)

ascii_shift('abcde')

def find_first(x, condition):
    for elem in x:
        if condition(elem):
            return elem
        
        find_first(range(10), lambda v: v**2 > 25)
       
        
       x = range(5)
y = {'hello': 'world'}
print_everything(*x, **y)



def find_first(x, condition, default=None):
    """
    Returns the first element of `x` for which
    the function `condition` returns a true-ish
    value, otherwise returns `default`.
    
    Parameters
    ---------
    x : iterable
        The elements to search in.
    condition : callable
        The condition to evaluate.
    default
        Value to return if no element matches.

    Returns
    -------
    The first element of x that fulfills the
    condition. If no element matches, returns
    the default value.
    """
    for elem in x:
        if condition(elem):
            return x
    else:
        return default
    
    isinstance(x, int)
    isinstance(x, object)
    issubclass(int, object)
x = 3  # ints are immutable
y = x  # y references the same Python object
x += 2  # same as x = x + 2, because int does not implement +=
print(x)
print(y)


print(id(x))
print(id(y))
print(x is y)

x = [1]
y = x
x = x + [5]  # creates a new object "x + [5]"
print(x, id(x))
print(y, id(y))
print(x is y)

class Animal(object):
    """
    An animal with a given `name` and `sound`.
    """
    def __init__(self, name, sound='...'):
        self.name = name
        self.sound = sound

    def talk(self):
        print(self.name + ': ' + self.sound)
        
    def shout(self):
        print(self.name + ': ' + self.sound.upper())
        
        class Duck(Animal):
    """
    A duck with a given `name`.
    """
    def __init__(self, name):
        super().__init__(name, 'quack')

class Dog(Animal):
    """
    A dog with a given `name`.
    """
    def __init__(self, name):
        super().__init__(name, 'woof')
        
        animals = [Animal('alf', 'grunt'), Duck('daisy'),
           Duck('dagobert'), Dog('fiffy')]
        animals[-1].name