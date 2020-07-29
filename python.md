**Decorator**

This takes a function, func, as an argument. It also defines a function, log_function_called, which calls func() and executes some code, print(f'{func}') called. Then it return the function it defined

```
def logging(func):
  def log_function_called():
    print(f'{func} called.')
    func()
  return log_function_called
 ```
Let’s write other functions that we’ll eventually add the decorator to (but not yet).
```
def my_name():
  print('chris')
  
def friends_name():
  print('naruto')
  
my_name()
friends_name()

#=> chris
#=> naruto
Now add the decorator to both.
@logging
def my_name():
 print('chris')
@logging
def friends_name():
 print('naruto')
my_name()
friends_name()
#=> <function my_name at 0x10fca5a60> called.
#=> chris
#=> <function friends_name at 0x10fca5f28> called.
#=> naruto
````

**func object**

func is the object representing the function which can be assigned to a variable or passed to another function. func() with parentheses calls the function and returns what it outputs.

**Map , reduce and filter**

map - applying a function to every element in a sequence. ex:-list(map(add_three, li))

reduce - 
reduce takes a function and a sequence and iterates over that sequence. On each iteration, both the current element and output from the previous element are passed to the function. In the end, a single value is returned.
```
from functools import reduce
def add_three(x,y):
    return x + y
li = [1,2,3,5]
reduce(add_three, li)
#=> 11
```
filter-
Each element is passed to a function which is returned in the outputted sequence if the function returns True and discarded if the function returns False.
```
def add_three(x):
    if x % 2 == 0:
        return True        
    else:
        return False
li = [1,2,3,4,5,6,7,8]
[i for i in filter(add_three, li)]
#=> [2, 4, 6, 8]
```

**Shallow copy and Deep Copy**

l1=l2
any change in l1 will be reflected on l2

shallow copy:
l1=list(l2)  or l1=l2.copy()
Adding an element in l2 wont reflect on l1
but if you modify initial values.. this change is reflected
```
li3 = [['a'],['b'],['c']]
li4 = list(li3)
li3.append([4])
print(li4)
#=> [['a'], ['b'], ['c']]
li3[0][0] = ['X']
print(li4)
#=> [[['X']], ['b'], ['c']]
```

deep copy:
ex:- li6 = copy.deepcopy(li5) 
li5 and li6 are two seperate lists. changes in one is not reflected on other

**Pickling**

Pickling is the go-to method of serializing and unserializing objects in Python.
In the example below, we serialize and unserialize a list of dictionaries.
```
import pickle
obj = [
    {'id':1, 'name':'Stuffy'},
    {'id':2, 'name': 'Fluffy'}
]
with open('file.p', 'wb') as f:
    pickle.dump(obj, f)
with open('file.p', 'rb') as f:
    loaded_obj = pickle.load(f)
print(loaded_obj)
#=> [{'id': 1, 'name': 'Stuffy'}, {'id': 2, 'name': 'Fluffy'}]
```

**Lambda function**

Lambda functions are anonymous functions in Python. They’re very helpful when you need to define a function that’s very short and consists of only one expression

**Property**

Python nuances:
1. An underscore _ at the beginning is used to denote private variables in Python.
2.property() is a built-in function that creates and returns a property object. The syntax of this function is:
property(fget=None, fset=None, fdel=None, doc=None)

```
# Using @property decorator
class Celsius:
    def __init__(self, temperature=0):
        self.temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    @property
    def temperature(self):
        print("Getting value...")
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        print("Setting value...")
        if value < -273.15:
            raise ValueError("Temperature below -273 is not possible")
        self._temperature = value


# create an object
human = Celsius(37)

print(human.temperature)

print(human.to_fahrenheit())

coldest_thing = Celsius(-300)
```

**Generator**

Yield instead of return statement

https://www.programiz.com/python-programming/generator
```
def fibonacci_numbers(nums):
    x, y = 0, 1
    for _ in range(nums):
        x, y = y, x+y
        yield x

def square(nums):
    for num in nums:
        yield num**2

print(sum(square(fibonacci_numbers(10))))
```

**Closure**

A function defined inside another function is called a nested function. Nested functions can access variables of the enclosing scope.
we have a closure in Python when a nested function references a value in its enclosing scope.
1.We must have a nested function (function inside a function).
2.The nested function must refer to a value defined in the enclosing function.
3.The enclosing function must return the nested function.

Closures can avoid the use of global values and provides some form of data hiding. It can also provide an object oriented solution to the problem.

**Python's garbage collector**

Python's garbage collector runs during program execution and is triggered when an object's reference count reaches zero
Dynamically created objects are assigned memory in heap

Mark and Sweep Algorithm
Any garbage collection algorithm must perform 2 basic operations. One, it should be able to detect all the unreachable objects and secondly, it must reclaim the heap space used by the garbage objects and make the space available again to the program.
The above operations are performed by Mark and Sweep Algorithm in two phases:
1) Mark phase
2) Sweep phase

```
Sweep()
For each object p in heap
    If markedBit(p) = true then
        markedBit(p) = false
    else
        heap.release(p)
```


