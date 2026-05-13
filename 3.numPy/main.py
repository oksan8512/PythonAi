import numpy as np

myarray = np.array([23,23,45,67,89,90])
print(myarray)

zeros = np.zeros((3,4)) # масив з 0 
print(zeros)

ones = np.ones((2,3)) # масив з 1
print(ones)

arr = np.arange(1,10,2) # масив від 1 до 10 з кроком 2
print(arr)

myArray =  np.array([[2,3,"Сало"], [4,False,6]])
print(myArray)

print("Shape", myArray.shape) #Розмір масиву
print("ndim", myArray.ndim) #кількість вимірів у масиві
print("dtype", myArray.dtype) #тип даних у елементах
print("size", myArray.size) # кількість усіх елементів


myList = np.array([23,46,26,16,29,54,21])
#отримуємо значення 1 елемента масиву
print("First element",myList[0])
print("1:4",myList[1:4]) #з 1 по 4 не включно
print(":3",myList[:3]) #3 з початку
print("2:",myList[2:]) #2 індеквсу і до кінця
print("::2",myList[::2]) #кожен другій

my2d = np.array([[3,4,6],[1,2,5],[0,9,-4]])
print("2d Array", my2d)
#MyArray
print("2d array my2d[1:,:2]", my2d[1:,:2])

a = np.array([2,3,5,7,8])
b = np.array([12,1,6,9,-2])
print("a:", a)
print("b:", b)

print("a + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", a / b)
print("a // b:", a // b)
print("a % b:", a % b)
# print("a ** b:", a ** b)
print("a == b:", a == b)
print("a != b:", a != b)

# Статичтині методи у python
data = np.array([2,3,4,9,2,5,5,6,7,8])
print("data", data)
print("mean", np.mean(data)) #середнє арифметичне
print("median", np.median(data)) # медіана - середнє значення, більш точне
print("max", np.max(data)) # максимальне значення
print("min", np.min(data)) # мінімальне значення