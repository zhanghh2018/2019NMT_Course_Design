import numpy as np

def decToBin(num):
    arry = []   #定义一个空数组，用于存放2整除后的商
    while True:
        arry.append(str(num % 2))  #用列表的append方法追加
        num = num // 2   #用地板除求num的值
        if num == 0:     #若地板除后的值为0，那么退出循环
            break

    return "".join(arry[::-1]).zfill(3) #列表切片倒叙排列后再用join拼接
def binToDec(binary):
    result = 0   #定义一个初始化变量，后续用于存储最终结果
    for i in range(len(binary)):
        #利用for循环及切片从右至左依次取出，然后再用内置方法求2的次方
        result += int(binary[-(i + 1)]) * pow(2, i)

    return result

class Logistic():

    def __init__(self):

        pass

    def sigmoid(self, z):
        '''激活函数'''
        return 1 / (1 + np.exp(-z))

    def logistic(self, X, theta):
        '''一层神经网络进行简单的逻辑运算'''
        h = self.sigmoid(X*theta.T)
        #print(h.shape[0])
        for i in range(int(h.shape[0])):
            h[i, 0] =1  if h[i, 0]>=0.5  else 0
            #print( h[i, 0])
        return h

num1=input("please input a number:")
num2=input("please input a number:")
#print(int(num1))
string1=decToBin(int(num1))
string2=decToBin(int(num2))
array1 = np.array(list(string1), dtype = int)
array2 = np.array(list(string2), dtype = int)
#print(array1)
#print(array2)
dig=np.array([1,1,1])
#print(dig)
con=list(zip(array1,array2,dig))
#con=list(zip(con,dig))
#print(con)
X = np.matrix(con)
theta1 = np.matrix([[-30, 20, 20], [10, -20, -20]])     #第一层网络的权重
theta2 = np.matrix([-10, 20, 20])                      #第二层网络的权重
theta3=np.matrix([10,-20])

log = Logistic()       #实例化

a1 = log.logistic(X, theta1)         #第一层
#print(a1)
b=np.ones([3,1])
a1 = np.c_[b, a1]                    #添加偏置单元
#print(a1)
a2 = log.logistic(a1, theta2)        #第二层 np.insert(a, 0, values=b, axis=1)
#print("a2.shape:",a2.shape)
a2=np.c_[b,a2]
#print("a2:",a2)
a3=log.logistic(a2, theta3)
#print(str(a3).replace("[","").replace("]","").replace(".",""))
num_str=""
for i in range(a3.shape[0]):
    temp=str(a3[i][0]).replace("[","").replace("]","").replace(".","")
    num_str+=temp
#print(num_str)
print(binToDec(num_str))
