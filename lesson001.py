#!/usr/bin/env python
# coding: utf-8

# In[1]:


#coding=utf-8
'''
主要进行矩阵计算
'''
import numpy as np

dt = np.dtype([('age', np.int8)])
a = np.array([(10,),(20,),(30,)], dtype=dt)
print(a)
print(a['age'])


# In[29]:


import numpy as np

student = np.dtype([('name','S20'),('age','i1'),('marks','f4')])
a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype=student)
print(a)
print(a['marks'])


# In[35]:


import numpy as np

a = np.arange(24)
print(a.ndim)
b = a.reshape(2,4,3)
print(b.ndim)
print(b.shape)


# In[38]:


import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a)
print(a.shape)
a.shape = (3,2)
print(a)
print(a.shape)


# In[40]:


import numpy as np
x = np.array([1,2,3,4,5], dtype=np.int8)
print(x.itemsize)


# In[41]:


import numpy as np
x = np.array([1,2,3,4,5])
print(x.flags)


# In[49]:


import numpy as np
x = np.empty([3,2], dtype=int)
# print(x.dtype)
print(x)#数组元素为随机值，如果dtype=np.int8(32,64) 则就不是的


# In[52]:


import numpy as np
#创建指定大小的数组，数组元素以0来填充
#默认类型dtype为浮点数
x = np.zeros((2,2), dtype=float, order='F')
print(x)


# In[56]:


import numpy as np
#创建指定形状的数组，数组元素以1来填充
x = np.ones([3,4], dtype=int, order='C')
#x = np.ones((3,4), dtype=int, order='C')与上同等效果
print(x)


# In[58]:


import numpy as np
# x = [1, 2, 3]  列表
# x = (1,2,3)
a = np.asarray(x)
print(a)


# In[82]:


import numpy as np

#numpy.frombuffer 用于实现动态数组
#接受buffer输入参数，以流的形式读入转化成ndarray对象
s = b"hello world"
x = np.frombuffer(s, dtype='S1')
# print(x)


# In[2]:


import numpy as np

#从迭代对象中简历ndarray

list = range(5)
it = iter(list)
x= np.fromiter(it, dtype=float)
print(x)


# In[29]:


import numpy as np
#从数值范围 创建数组

# x = np.arange(5)
# x = np.arange(5, dtype=float)
# x = np.arange(10, 20, 2) # 1--20 步长为2
# x = np.linspace(0, 10, num=5, endpoint=False, dtype=int ) # 床架一个一维数组  等差数列
# num 样本数量  endpoint 是否包含截止点
x = np.linspace(0, 10, num=6, endpoint=False, dtype=int ).reshape((2,3))
# 生成的矩阵 mXn m*n= num
print(x)
print(x.dtype)


# In[38]:


import numpy as np

y = np.linspace(1, 9, num=5, endpoint=True, dtype=int )
print(y)
x = np.logspace(1, 9, num=5, endpoint=True, base=2, dtype=int)
# 以base为底的一个幂 依次创建一个等比数列
# 1,9 num=3 会生成3个数的等差数列 1, 5, 9
# 在以base=2为底   依次为 2^1, 2^5, 2^9
print(x)


# In[43]:


import numpy as np
#切片和索引

a = np.arange(10)
s = slice(2,7,2)# (start, stop, step)  slice内置函数
b = a[2:7:2]# (start: stop :step) 直接操作数组

#冒号 : 的解释：如果只放置一个参数，如 [2]，将返回与该索引相对应的单个元素。
#如果为 [2:]，表示从该索引开始以后的所有项都将被提取。
#如果使用了两个参数，如 [2:7]，那么则提取两个索引(不包括停止索引)之间的项。
print(a[s])
print(b)
print(a[2])
print(a[2:])
print(a[2:7])


# In[51]:


import numpy as np
# 多维数组切片和索引
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)

print('从数组索引1的位置开始切割：')
print(a[1:])

print('************************************')
print(a[0,2])  # 第1行第3列
print(a[...,1]) #第2列全部元素（只有第2列）
print(a[...,1:]) #第2列以及剩下全部列元素


# In[73]:


import numpy as np
#整数数组索引，布尔索引，花式索引

#数组索引
# x = np.array([[1,2],[3,4].[5,6]])
# y=x[[0,1,2],[0,1,0]] # 获取数组 (0,0) (1,1) (2,0)
# #获取第一行第一列，第二行第二列，第三行第一列
# print(y)

# x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
# print('数组：')
# print(x)
# print('***********************************')
# rows = np.array([[0,0],[3,3]])
# cols = np.array([[0,2],[0,2]])
# y = x[rows,cols]
# print('取的就是四个角的元素：')
# # （0,0）（0,2）（3,0）（3,2）
# print(y)


# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# b = a[1:3, 1:3]
# c = a[1:3, [1,2]]
# d = a[..., 1:]

# print(b)
# print(c)
# print(d)

#*******************************************************
#布尔索引
# x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
# print("数组")
# print(x)
# print("大于5的元素")
# print(x[x>5])
# a = np.array([np.nan, 1, 2, np.nan, 3,4,5])
# print(a[~np.isnan(a)])#过滤NaN
# a = np.array([1, 2+6j, 5, 3+5j])
# print(a[np.iscomplex(a)])

#*******************************************************
#花式索引
#花式索引是利用整数数组索引，根据索引数组的值作为目标数组的某个轴的下标来取值
#对于使用一维数组作为索引，如果目标数组是一维数组，那么索引的结果就是对应位置的元素
#如果目标数组是二维数组，那么索引的结果对应下标的行
#花式索引跟切片不一样，他总是将数据复制到新数组中

#输入顺序索引数组
# x = np.arange(32).reshape((8,4))
# print(x[[4,2,1,7]])#取得第5行，第2行，第2行，第8行 从下标0开始

# #输入倒序索引数组
# x = np.arange(32).reshape((8,4))
# #输入倒数第4行（第5行），倒数第2行（第7行），倒数第1行（第8行），倒数第7行（第2行）
# print(x[[-4,-2,-1,-7]]) 

#传入多个索引数组
# x = np.arange(32).reshape((8,4))
# print(x[np.ix_([1,5,7,2],[0,3,1,2])])#二维数组中，前面一个是行，后面一个是列
# #[(1,0),(1,3),(1,1),(1,2)] 表示第2行的元素，顺序按着指定的位置索引排列，以下几行类推

x = np.arange(32).reshape((2,4,4))
print("数组：")
print(x)
print("找到的位置：")
print(x[np.ix_([1,0],[1,0,3,2],[0,3,1,2])])


# In[86]:


import numpy as np
# broadcast
#用于两个数组的运算，不同形状（shape）的数组也可以进行计算

#数组相同
# a = np.array([1,2,3,4])
# b = np.array([10,20,30,40])
# c = a*b
# print(c)

#数组不相同时,
a = np.array([[0,0,0],
             [10,10,10],
             [20,20,20],
             [30,30,30]])
b = np.array([[1,2,3]]) # b = np.array([1,2,3])  也可以
print(a+b)
#4x3 的二维数组与长为 3 的一维数组相加，等效于把数组 b 在二维上重复 4 次再运算：

bb = np.tile(b, (4, 1))
print(bb)
#广播的规则
#让所有输入的数组都像其中形状最长的数组看齐，形状中不足的部分都通过在前面加1补齐
#输出数组的形状是输入数组形状的各个维度上的最大值
#如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为1时，
#这个数组能够用来计算，否则出错
#当输入数组的某个维度的长度为1时，沿着此维度运算时都用此维度上的第一组值

# 对两个数组，分别比较他们的每一个维度，（若其中一个数组没有当前维度则忽略）
#（1）数组拥有相同形状
#(2)当前维度的值相等
#（3）当前维度值有一个是1


# In[8]:


import numpy as np
a = np.arange(6).reshape(2,3)
print('原始数组是：')
print(a)

print('迭代输出元素：')
#该实例不使用标准C或者Fortran顺序，选择顺序是和数组内存布局一致的，
#这样做是为了提升访问的效率，默认是行序优先（row-major-order）
print("默认输出：")
for x in np.nditer(a):
    print(x, end=",")

print("\n")
print("*************************")
print("输出转置矩阵：")
for x in np.nditer(a.T):
    print(x, end=", ")
print("\n")

print("**********************")
print("输出转置矩阵，并以C-order输出")
for x in np.nditer(a.T.copy(order='C')):
    print(x, end=", ")
print('\n')

# a和a.T的遍历顺序是一致的，默认都是按存储顺序来遍历，
#C-order是以行访问，不管内部存储顺序，
# C-order 行序优先  for x in np.nditer(a, order='C')
# Fortran 列序优先  for x in np.nditer(a.T, order="F")


# In[10]:


import numpy as np

a = np.arange(0, 60, 5)
a = a.reshape(3,4)
print('原始数组：')
print(a)
print("\n")
print("*****************")
print("原始数组的转置")
b = a.T
print(b)
print("\n")
# print("以C风格排序遍历")
# c = b.copy(order = "C")
# print(c)
# for  x in np.nditer(c):
#     print(x, end =", ")
# print('\n')
# print("以F风格顺序排序")
# c = b.copy(order="F")
# print(c)
# print("\n")
# for x in np.nditer(c):
#     print(x, end=", ")

#也可以直接显示的设置
print("以C风格排序遍历")
for  x in np.nditer(a, order="C"):
    print(x, end =", ")
print('\n')
print("以F风格顺序排序")
for  x in np.nditer(a, order="F"):
    print(x, end =", ")
print("\n")


# In[30]:


import numpy as np

a = np.arange(0, 60, 5)
a = a.reshape(3,4)
print('原始数组：')
print(a)
print("\n")

# for x in np.nditer(a, op_flags=['readwrite']):
#     x[...]=2*x
# print("修改后的数组")
# print(a)

for x in np.nditer(a, flags=['external_loop'], order="F"):
    print(x, ", ")

# b = [1,2,3,4,5,6,7]
# for i, x in enumerate(b):
#     print(i, x)


# In[31]:


import numpy as np

# 广播迭代

a = np.arange(0, 60, 5)
a = a.reshape(3,4)

b = np.array([1,2,3,4], dtype=int)
print(b)
print("修改后的数组")
for x, y in np.nditer([a,b]):
    print("%d:%d" % (x,y), end =", ")
    
#如果两个数组是可广播的，nditer组合对象能够同时迭代他们。假如
#数组a的维度为3X4，数组b的维度为1X4，则使用迭代器数组b被广播到a的大小


# In[88]:


import numpy as np

#numpy数组操作

a = np.arange(8)
a = a.reshape(2,4)# 注意shape作为属性的  等价a.shape=(2,4)
# a.shape=(2,4)
# a = np.zeros([2,2], dtype=np.str)
# a = np.array([['1','c'],['c','f']])
print('原始数组')
print(a)
print('\n')

# numpy.reshape(arr, newshape, 'order')  
# order c-行，f-列，‘A’-原顺序，‘k’-元素在内存中出现的顺序
# newshape 整数或者整数数组
# b = np.reshape(a,[4,2], order='A') #但这种情况下写k会出错 
# print(b)


# #flat  元素的迭代器
# for row in a:
#     print(row)

# #对数组每个元素（一维数组）都可以进行迭代处理，可使用flat属性，
# #该属性是一个数组元素迭代器
# print('迭代后的数组')
# for element in a.flat:
#     print(element)

# flatten
#返回一份数组拷贝，对拷贝所做的修改不会影响原始数组
#返回的形式 为一维数组
# a = np.array([[[1,2,3],[1,2,3]]])
# b = a.flatten(order='f')# 里面可以传递order
# print(a)
# print(a.flatten(order='f'))
# print(b)


#ravel
#展开平的数组，c风格，返回的是数组视图，修改会影响原始数组
b = np.ravel(a, order='f')
b = a.ravel(order='f')
for x in np.nditer(b,op_flags=['readwrite']):
    x[...]= 2*x
print(a)
print(a.ravel(order='f'))
print(b)


# In[105]:


import numpy as np

#numpy数组操作
#数组翻转

a = np.arange(8)
a = a.reshape(2,4)
print('原始数组')
print(a)
print('\n')

# 对换数组的维度
# b = np.transpose(a) # 等价于 a.T
b = np.transpose(a,(1,0)) #第二个参数是axes 
# help(np.transpose)
print(b)
print("\n")
print(a.T)


# In[114]:


import numpy as np
# rollaxis

#函数向后滚动特定的轴到一个特定位置，
#注意：多维数组的axis是与数组下标相关的
#要理解这个概念  参考：https://blog.csdn.net/liaoyuecai/article/details/80193996
#获取某个axis轴相同的元素，其实就是获取下标相同的一系列元素
#（arr，axis，start） axis 要向后滚动的轴，其他轴的相对位置不会改变
# start 默认为0，表示完整的滚动

a = np.arange(8).reshape(2,2,2)
print('原数组')
print(a)
print('\n')

# #将轴2滚动到轴0（宽度到深度）
# print(np.rollaxis(a, 2)) #默认start为0  np.rollaxis(a, 2, 0)
# print('\n')

#将轴0滚动到轴1
print(np.rollaxis(a,2,1))


# In[115]:


import numpy as np

#交换数组的两个轴
# 记住轴其实就是维度 操作轴其实就是对下标维度的排序操作
#与rollaixs 对比记忆
a = np.arange(8).reshape(2,2,2)
print("原数组：")
print(a)

print("交换两个axis")
print(np.swapaxes(a, 2, 0))

print("对比rollaxis")
print(np.rollaxis(a,2,0))


# In[20]:


import numpy as np

#修改数组维度

#broadcast 用于模仿广播的对象，该对象封装了一个数组广播到另一个数组的结果
x = np.array([[1],[2],[3]])
y = np.array([4,5,6])


#对y广播x
b = np.broadcast(x, y)
print(b)
# 它拥有iterator属性，基于自身组件的迭代器元组
print('对y广播x：')
r,c = b.iters
i = 0
# while(i < b.size):
#     print(next(r), next(c))
#     i+=1
print('\n')

#广播对象的形状
# print(b.shape)
# print('\n')

c = np.empty(b.shape)#创建对y广播x之后的数组
c.flat = [u + v for u,v in b]
# print(c)

# for i in c.flat:   #转换为迭代器，把每一个元素都返回
#     print(i)

# for i in c: #返回多个一维数组
#     print(i)

# print("\n")
# print(x + y)

#  b，x + y，c，以上三个输出结果都是一样的


# In[22]:


import numpy as np

#修改数组维度
x = np.array([[1],[2],[3]])
y = np.array([4,5,6])


#对y广播x
b = np.broadcast(x, y)
for u,v in b:
    print(u,v)


# In[48]:


import numpy as np

# broadcast_to 函数将数组广播到新形状。它在原始数组上返回只读视图，它通常不连续。
# 如果新形状不符合numpy的广播规则，该函数可能会抛出ValueError
# numpy.broadcast_to(array, shape, subok)

# a = np.arange(4).reshape(1,4)
# print("原数组：")
# print(a)
# print("\n")

# print("调用broadcast_to 函数之后：")
# print(np.broadcast_to(a, (2,4)))

#***************************************

# #expand_dims函数通过在指定位置插入新的轴来扩展数组形状
# # numpy.expand_dims(arr, axis)
# # 是扩展 添加
# x = np.array(([1,2],[3,4]))
# print("数组x：")
# print(x)
# print("\n")

# y = np.expand_dims(x, axis=0)
# print('数组 y:')
# print(y)
# print("\n")

# print('x的形状', x.shape)
# print('y的形状', y.shape)

# z = np.expand_dims(x, axis=1)
# print('z的形状', z.shape)
# print("z")
# print(z)

# #ndim 获取维数
# print('x_ndim',x.ndim)
# print('y_ndim',y.ndim)
# print('z_ndim',z.ndim)


#numpy.squeeze
#函数从给定数组的形状中删除一维条目
#numpy.squeeze(arr, axis)

x = np.arange(9).reshape(1,3,3)
print('数组x:')
print(x)
print('\n')

# y = np.squeeze(x) # axis 不传 默认删除第一个遇到的单维条目
y = np.squeeze(x, axis=1)# axis的值 只能选择单维条目的轴

print('数组y:')
print(y)

print('x的形状：',x.shape)
print('y的形状：',y.shape)


# In[52]:


import numpy as np

#连接数组
#np.concatenate（（a1,a2, ...）,axis）
#((a1,a2,...))相同类型的数组 axis—沿着他连接数组的轴

# a = np.array([[1,2], [3,4]])
# print('第一个数组：')
# print(a)
# print('\n')

# b=np.array([[5,6],[7,8]])
# print('第二个数组')
# print(b)
# print('\n')

# #两个数组的维数是相同的，可以实现连接
# print('沿轴0连接两个数组：')
# print(np.concatenate(a, b))
# print('\n')

# print('沿轴1连接两个数组：')
# print(np.concatenate(a,b), axis=1)


# In[62]:


import numpy as np


#参考链接：https://www.cnblogs.com/nkh222/p/8932369.html
#stack 用于沿新轴连接数组序列
#np.stack(arrays, axis) 
#arrays 相同形状的数组序列，axis 返回数组中的轴，输入数组沿着它来堆叠

#记住轴的  下标顺序 从左到右增序排序

a = np.array([[1,2],[3,4]])
print('第一个数组：')
print(a)
print('\n')

b = np.array([[5,6],[7,8]])
print('第二个数组')
print(b)
print('\n')

# # 沿 0轴  其实就是0*x*x  0-以具体的数组维数确定 
# print('沿轴0堆叠两个数组')
# c = np.stack((a,b), 0)
# print(c)
# print(c.shape)

# # 沿 1轴  其实就是x*0*x  0-以具体的数组维数确定 
# print('沿轴1堆叠两个数组')
# c = np.stack((a,b), 1)
# print(c)
# print(c.shape)



# #numpy.hstack   stack函数的变体，它通过水平堆叠来生成数组  按列对接
# #等价于  numpy.concatenate(array, axis=1)
# print('水平堆叠：')
# c = np.hstack((a, b))
# print(c)
# print('\n')

# #numpy.vstack(array, axxis)  stack函数的变体，它通过垂直堆叠来生成数组
# #等价于  numpy.concatenate(array, axis=0)
# print("垂直堆叠")
# c = np.vstack((a,b))
# print(c)


# #numpy.dstack(array, axxis)  stack函数的变体，按堆栈数组顺序对接
# #（沿第三维，可以理解为深度或高度）
# #等价于  numpy.concatenate(array, axis=2)
# print("垂直堆叠")
# c = np.vstack((a,b))
# print(c)


# In[69]:


import numpy as np

#分割数组

# numpy.split(array, indices_or_sections, axis)
# indices_or_sections 如果是整数，就用该数平均切分，
#如果是一个数组，为沿轴切分的位置（左开右闭）

#axis:沿着哪个维度进行切分，默认为0，横向切分，为1时，纵向切分

# a = np.arange(9)
# print('第一个数组：')
# print(a)
# print("\n")

# print('将数组分为三个大小相等的子数组')
# b = np.split(a, 3)
# print(b)
# print('\n')

# print('将数组在一维数组中表明的位置分割')
# b = np.split(a, [4,7])
# print(b)

# #hsplit  水平分割数组，通过指定要返回的相同形状的数组数量来拆分原数组
# #  按列拆开
# harr = np.floor(10*np.random.random((2,6)))
# print("原数组：")
# print(harr)

# print("拆分后：")
# print(np.hsplit(harr, 3))


# vsplit  垂直分割数组，通过指定要返回的相同形状的数组数量来拆分原数组
#  按行拆开
harr = np.floor(10*np.random.random((2,6)))
print("原数组：")
print(harr)

print("拆分后：")
print(np.vsplit(harr, 2))  #只能拆分成2个


# In[81]:


import  numpy as np

#数组元素的添加与删除

# a = np.array(([1,2,3],[4,5,6]))
# print("第一个数组")
# print(a)
# print("\n")

# print("第一个数组的形状：")
# print(a.shape)
# print('\n')

# # numpy.resize(arr, shape) 返回指定大小的新数组
# #如果新数组大小大于原始大小，则包含原始数组中的元素的副本
# b = np.resize(a, (3,2))
# print("第二个数组：")
# print(b.shape)
# print(b)
# print('\n')
# print("修改第二个数组大小")
# b = np.resize(a, (3,3))
# #数组大小大于原始大小，复制原始数组中的元素添加进去，能添加多少添加多少
# print(b)

# append(arr, values, axis)
# values 要向arr添加的值，需要和arr形状相同
# 当axis无定义时，是横向加成，返回总是为一维数组，当axis有定义的时候，分别
# 为0和1的时候（列数要相同）。当axis=1，数组是加载右边（行数相同）

# print("向数组中添加元素：")
# print(np.append(a, [7,8,9]))
# print('\n')

# print('沿轴0添加元素：')
# print(np.append(a, [[7,8,9]], axis=0))
# print('\n')

# print('沿轴1添加元素')
# print(np.append(a, [[5,5,5],[7,8,9]], axis=1))


# # numpy.insert(arr, obj, values, axis)
# # 给定索引之前，沿给定轴在输入数组中插入值，
# # 如果值的类型转换为要插入，则它与输入数组不同，插入没有原地的，函数会返回一个新数组。
# # 此外，如果未提供轴，则输入数组会被展开
# # obj 在其之前插入值的索引
# # values 要插入的值
# # axis，如果不提供，则输入数组会被展开

# a = np.array([[1,2],[3,4],[5,6]])
# print("第一个数组：")
# print(a)
# print('\n')

# print('未传递Axis参数，在插入之前输入数组会被展开')
# #全部展开就是说变成一维数组了，前面一个数对应在其索引前加入
# print(np.insert(a, 3, [11,12]))
# print("\n")

# print('传递了Axis参数，在广播值数组来配输入数组。')
# print('沿轴0广播：')
# print(np.insert(a, 1, [11], axis=0))

# print('\n')
# print('沿轴1广播：')
# print(np.insert(a, 1, [11], axis=1))


# numpy.delete(arr, obj, axis)
# 从输入数组中删除指定子数组的新数组，
# 与insert()情况一样，如果未提供轴参数，则输入数组将展开

a = np.arange(12).reshape(3,4)
print('第一个数组：')
print(a)
print('\n')

print('未传递Axis参数')
print(np.delete(a,5))
print("\n")

print('删除第二行：')
print(np.delete(a, 1, axis=0))
print('\n')

print('删除第二列：')
print(np.delete(a, 1, axis=1))
print('\n')

print('包含从数组中删除的替代值的切片：')
a = np.array([1,2,3,4,5,6,7,8,9,10])
print(np.delete(a, np.s_[::2]))  #后期学到


# In[90]:


import  numpy as np

#去除数组中的重复元素
#numpy.unique(arr, return_index, return_inverse, return_couonts)
# return_index: bool 为True 返回新列表元素在旧列表中的位置（下标），并以列表形式存储
# return_inverse: bool 为True 返回旧列表元素在新列表中的位置（下标），并以列表形式存储
# return_counts: bool 为True 返回去重数组中的元素在原数组中的出现次数

a = np.array([5,2,6,2,7,5,6,8,2,9])
print("第一个数组：")
print(a)
print("\n")

print("第一个数组的去重值：")
u = np.unique(a)
print(u)
print("\n")

print("去重数组的索引数组：")
u, indices = np.unique(a, return_index=True)
print(indices) #在原有列表中对应的下标值
print("\n")
print('每个和原数组下标对应的数值:')
print(a)
print('\n')

print("去重数组的下标") #旧列表元素在新列表中的位置
u, indices = np.unique(a, return_inverse = True)
print(indices)
print("数组为：")
print(u)
print("\n")

print("使用下标重构原数组")  
#return_inverse表明了可以通过去重的数组，去重新创建原数组
print(u[indices])
print("\n")

print("返回去重元素的重复数量")
u, y = np.unique(a, return_counts = True)
print(u)
print(y)


# In[107]:


import numpy as np

#位运算 一些的运算分别等价于  &， ~， |  ^
#位运算规则 与其他语言规则一样

a,b = 13, 17
print(bin(a), bin(b))


#与
# #bitwise_and  对数组元素执行位与操作
# print(np.bitwise_and(13, 17))
# print(13&17)


# #或
# #bitwise_or 函数对数组中整数的二进制形式执行位于运算
# print(np.bitwise_or(13, 17))
# print(13|17)

# #取反操作
# # invert() 对整数进行取反操作，即0变1,1变0
# #对于有符号整数，取该二进制数的补码，然后+1.
# #二进制数，最高位为0表示正数，最高位为1表示负数
# #取反原则 与其他编程语言规则一样
# #正数直接将1变0,0变1即可
# #负数 除符号位（最高位）不变，其余均取反，并加1 得到补码
# print(np.invert(np.array([13], dtype=np.uint8)))
# print(bin(242))
# print(np.binary_repr(242, width=8))  #返回指定宽度中十进制数的二进制表示


# #left_shift  左移操作
# #函数将数组元素的二进制形式向左移动到指定位置，右侧附加相等数量的0
# print("将10左移两位：")
# print(np.left_shift(10, 2))
# print('\n')

# print("40的二进制表示")
# print(bin(40)) # np.binary_repr(40, width=8)

# #right_shift
# # 二进制形式向右移动到指定位置，左侧附加相等数量的0
# print("将40右移两位：")
# print(np.right_shift(40, 2))
# print('\n')


# In[128]:


import numpy as np

#python 内置标准字符串函数
# 对dtype值为 numpy.string_ 或者  numpy.unicode_ 数组执行向量化字符串操作

# #numpy.char.add() 依次对两个数组的元素进行字符串连接

# print("连接两个字符串")
# print(np.char.add(['hello'],[' world']))
# print("\n")

# print(np.char.add(['hello'],["abc", ' world'])) #不同形状的数组也可以直接连接

# # numpy.char.multiply(arr, count)
# #执行多重连接
# print(np.char.multiply('Runnoob ', 3))

# numpy.char.center()
# 用于将字符串居中，并使用指定字符在左侧和右侧进行填充
# numpy.char.center(arr, width, fillchar)
# width 字符左右要填充的宽度， fillchar  待填充的字符
# print(np.char.center("lina", 20, fillchar="*")) #********lina********

# 将字符串的第一个字母转换为大写
# print(np.char.capitalize('hello'))

# 将每个单词的第一个字母转换为大写
#numpy.char.title(str)
# print(np.char.title('i like runnoob'))

# # 对数组的每个元素转换为小写，它对每个元素调用str.lower()
# print(np.char.lower(['RUNOOB',"BAIDU"]))
# print(np.char.lower('BAIDU'))

#数组的每个元素转换为大写，它对每个元素调用str.upper
# print(mp.char.upper(['runoob','google']))

# # 对字符串进行分割，并返回数组。默认情况下，分隔符为空格
# #np.char.split(arr, sep) sep=分隔符
# print(np.char.split('i like runnoob'))
# #分隔符
# print(np.char.split('www.baidu.com', sep='.'))

#以换行符作为分割符来分割字符串，并返回数组
#numpy.char.splitlines()
# print(np.char.splitlines('i\nlike runnoob'))
# \n, \r,\r\n  都可看做空格符

# #numpy.char.strip(arr, value)  移除开头和结尾的特点字符
# # arr 数组， value-要移除的字母
# print(np.char.strip('ashok arunooba', 'a'))
# print(np.char.strip(['arunooba', 'admin', 'java'], 'a'))

# #numpy.char.join(value, arr)
# #value 指定的连接符  arr字符串或元素或数组
# #通过指定分割符来连接数组中的元素或字符串
# print(np.char.join(':', 'runoob'))
# print(np.char.join([':','-'],['runoob','google']))

# #使用字符串替换字符串中的所有子字符串
# #numpy.char.replace(arr, oldStr, newStr)
# print(np.char.replace('i like goole', 'oo', 'XX'))

#编码
#数组中的每个元素调用str.encode函数 默认编码utf-8
a = np.char.encode('runnoob', 'cp500')
print(a)
#解码
#np.char.decode  对编码的元素进行接解码 调用str.encode解码
print(np.char.decode(a, 'cp500'))


# In[142]:


import numpy as np

#三角函数
a = np.array([0, 30, 45, 60, 90])

# numpy.sin(arr) 正弦
# 通过pi/180 转化为弧度
print("输出不同角度的正弦值")
sin = np.sin(a*np.pi/180)
print(sin)
print('\n')

# numpy.cos(arr) 余弦
print("输出不同角度余弦值")
cos = np.sin(a*np.pi/180)
print(cos)
print('\n')

#numpy.tan(arr) 正切
print("输出不同角度正切值")
tan = np.tan(a*np.pi/180)
print(tan)
print('\n')

#numpy.degrees() 将弧度转换为角度
print("输出不同角度的反正弦")
arcsin = np.arcsin(sin)
print(arcsin)
print('\n')

#对应的角度
print("对应的角度")
print(np.degrees(arcsin))
print("\n")

#反正切
print("反正切")
print(np.arctan(tan))
print("\n")

#反余弦
print("反余弦")
print(np.arccos(cos))
print("\n")


# In[150]:


import numpy as np

# #四舍五入  
# #np.around(arr, decimals)
# #decimals 舍入的小数位数，默认值为0 如果为负数，整数将四舍五入到小数点左侧的位置

# a = np.array([1.0, 5.55, 123, 0.567, 25.532])
# print('原数组')
# print(a)
# print('\n')

# print('舍入后')
# print(np.around(a)) # decimals 为0 只保留了整数位
# print('\n')

# print(np.around(a, decimals = 1))
# print(np.around(a, decimals = -1))

# # 向下舍整数   往小的取
# a = np.array([-1.7, 1.5, -0.2, 0.6, 10])
# print('提供的数组')
# print(a)
# print('\n')
# print('修改后的数组')
# print(np.floor(a)) #没有指定位数的参数

# #向上入整数  往大的取
# a = np.array([-1.7, 1.5, -0.2, 0.6, 10])
# print('提供的数组：')
# print(a)
# print('ceil修改的数组：')
# print(np.ceil(a))


# In[161]:


import numpy as np
#算术函数

# a = np.arange(9, dtype=np.float_).reshape(3,3)
# print("第一个数组：")
# print(a)
# print('\n')

# print('第二个数组')
# b = np.array([10, 9, 10])
# print(b)
# print('\n')

# print('两个数相加：')
# print(np.add(a, b))
# print("\n")

# print("两个数相减：")
# print(np.subtract(a, b))
# print("\n")

# print("两个数相乘")
# print(np.multiply(a, b))
# print("\n")

# print("两个数相除")
# print(np.divide(a, b))

# #求数组元素的倒数
# #numpy.reciprocal(arr)
# a = np.array([0.25, 1.33, 1, 100])
# print('数组：')
# print(a)
# print("\n")

# print(np.reciprocal(a))

# 将输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂
# numpy.power(arr, values)
# values--指数 arr--数组的元素作为底数
# a = np.array([10, 100, 1000])
# print('数组是：')
# print(a)
# print('\n')

# print("调用pow")
# print(np.power(a, 2))
# print("\n")

# b = np.array([1,2,3])
# print(b)
# print('\n')
# print(np.power(a, b)) #对应元素位置调用。两个数组要满足广播规则

#求余数
#numpy.remainder(arr1, arr2) numpy.mod(arr1, arr2)
# arr1作为被除数， arr2 作为除数
a = np.array([10, 20, 30])
b = np.array([3, 5, 7])
print('第一个数组：')
print(a)
print('\n')

print('第二个数组：')
print(b)
print('\n')

print('调用mod()函数：')
print(np.mod(a, b))
print("\n")
print('调用remainder()函数：')
print(np.remainder(a, b))


# In[24]:


import numpy as np

#统计函数
# a = np.array([[3,7,5], [8,4,3], [2,4,9]])
# print('原数组是：')
# print(a)
# print('\n')

# #numpy.amin(arr, axis) 计算数组的元素沿轴指定最小值
# print('调用amin()函数')
# print(np.amin(a, 0))
# print(np.amin(a))  #不指定轴 表示所有元素的最小值

# print("调用amax()函数")
# print(np.amax(a,0))
# print(np.amax(a)) #不指定轴 表示所有元素的最小值

# #计算数组中元素最大值与最小值之差
# print('调用ptp()函数')
# print(np.ptp(a))  #所有元素中最大值与最小值之差
# print("\n")
# print(np.ptp(a, 0))

# #统计小于某个值的观察值的百分比
# #numpy.percentile(arr, p, axis)
# #p 要计算的百分位数  0-100
# #第p个百分位数是这样一个值，它使得至少有p%的数据项小于或等于这个值，
# #且至少有（100-p）%的数据项大于或等于这个值
# #比如一个学生考了54分，对应的是第70%，那么我们可以知道大约70%的学生考的分数低于54分
# a = np.array([[10,7,4],[3,2,1]])
# print('数组是：')
# print('\n')

# print('调用percentile()函数')
# print(np.percentile(a, 50)) #获取50%的分位数 就是a排序之后的中位数
# print("\n")

# print(np.percentile(a, 50, axis=0))
# print("\n")
# print(np.percentile(a, 50, axis=1))
# print("\n")

# print(np.percentile(a, 50, axis=1, keepdims=True))  #保持维度不变


# #计算数组a中的元素的中位数（中值）
# a = np.array([[30,65,70],[80.95,10],[50,90,60]])
# print('数组：')
# print('\n')

# print('调用median()函数')
# print(np.median(a)) #所有数组展开之后，排序之后的中位数
# print('\n')

# print('沿轴0调用median()函数：')
# print(np.median(a, axis = 0))
# print('\n')

# print('沿轴1调用median()函数：')
# print(np.median(a, axis = 1))
# print('\n')

# # 返回数组中元素的算术平均值 沿轴的元素总和除以元素的数量
# # numpy.mean(arr, axis) 
# # axis 有轴 则沿其计算

# a = np.array([[1,2,3],[3,4,5],[4,5,6]])
# print('我们的数组：')
# print(a)
# print('\n')

# print('调用mean()函数')
# print(np.mean(a))  #所有元素的平均值
# print('\n')

# print('沿轴0')
# print(np.mean(a, axis = 0))  #沿轴0
# print('\n')

# print('沿轴1')
# print(np.mean(a, axis = 1))  #沿轴1
# print('\n')

# #average(arr, weights, returned, axis)
# #根据另一个数组中给出的各自的权重计算数组中元素的加权平均值
# #该函数可以接受轴参数，没有的话，数组会展开

# a = np.array([1,2,3,4])
# print('数组：')
# print(a)
# print('\n')

# print('调用average')
# #不指定权重相当于 mean
# print(np.average(a))
# print('\n')

# #指定权重
# wts = np.array([4,3,2,1])
# print(np.average(a, weights=wts))
# print('\n')

# #返回权重的和
# print("加权平均值：权重的和")
# print(np.average(a, weights=wts, returned=True))

# #多维数组中 可以指定计算的轴
# a = np.arange(6).reshape(3,2)
# print('数组是：')
# print(a)
# print('\n')

# wts = np.array([3,5])
# print('指定轴1的计算加权平均值')
# print(np.average(a, axis=1, weights = wts))

#标准差，
#一组数据平均值分散程度的一种度量
#方差的算术平方根
# std=sqrt(mean((x-x.mean())**2))
print(np.std([1,2,3,4]))


#方差，
#每个样本值与全体样本值的平均数之差的平方值的平均数mean((x-x.mean())**2)
# std=sqrt(mean((x-x.mean())**2))
print(np.var([1,2,3,4]))


# In[28]:


import numpy as np

#排序 
#quicksort 快排， mergesort 归并排序  heap堆排序
#返回输入数组的排序副本
#numpy.sort(a, axis, kind, order)
# a 数组
# kind 默认为quicksort
# # order 如果数组包含字段，要排序的字段
# a = np.array([[3,7],[9,1]])
# print('数组：')
# print(a)
# print("\n")

# print('调用sort()')
# print(np.sort(a))
# print('\n')
# print('按列展开')
# print(np.sort(a, axis=0))
# print('\n')
# print('按行展开')
# print(np.sort(a, axis=1))
# print('\n')

# 在sort函数中排序字段
dt = np.dtype([('name','S10'),('age',int)])
a = np.array([('raju', 21),('anil', 25),('ravi', 17),('amar', 27)], dtype=dt)
print(np.sort(a, order='age'))


# In[92]:


import numpy as np

# x = np.array([3,1,2])
# print('我们的数组是：')
# print(x)
# print('\n')

# #数组值从小到大的索引值
# print('调用argsort')
# y = np.argsort(x) #对数组的元素进行排序，返回显示的是其索引值
# print(y)
# print('\n')
# print('以排序后的顺序重构原数组：')
# print(x[y]) #得到的结果就是从小到大的排序
# print('使用循环构造')
# for i in y:
#     print(x[i], end='')

# #lexsort（tuple） 用于多个序列进行排序，
# #把它想象成对电子表格进行排序，每一列代表一个序列，排序时优先照顾靠后的列
# #可以理解按列排序，后面的列会最终起到主导作用
# #比如：小升初考试，重点班录取学生按照总成绩录取。在总成绩相同时，
##数学成绩高的优先录取，在总成绩和数学成绩都相同时，按照英语成绩录取…… 这里，
##总成绩排在电子表格的最后一列，数学成绩在倒数第二列，英语成绩在倒数第三列。

# nm = ('raju', 'anil', 'ravil', 'amar')
# dv = ('f.y', 's.y', 's.y', 'f.y')
# ind = np.lexsort((dv, nm))
# print('调用lexsort()函数')
# print(ind)

# #排序时，先排nm，在排dv，只有当nm中有元素相同时，其他列排序才会起作用
# print('\n')
# print('使用这个索引来获取排序后的数据')
# print([nm[i] + "," + dv[i] for i in ind])

#numpy.msort(a)  数组按第一个轴排序，返回排序后的数组副本。 
# 等价于numpy.sort(a, axis=0)
# sort_complex(a) 对复数按照先实部后虚部的顺序进行排序
# partition(a, kth(axis, kind, order)) 指定一个数，对数组进行分区
# argpartition(a, kth(axis, kind, order)) kind指定算法沿着指定轴对数组进行分区

# #复数排序
# print(np.sort_complex([5,3,6,2,1]))
# a = np.array([1+0j, 2+0j, 3+0j, 5+0j, 6+0j])
# print(np.sort_complex(a))

# #partition分区排序
# a = np.array([3,4,2,1])
# #将数组a中的所有元素（包括重复元素）从小到大排列，比第3小的放在前面，大的放在后面
# print(np.partition(a, 3))
# print('\n')

# #argpartition 
# arr = np.array([46, 57, 39, 1, 10, 0, 120])
# #将元素第3小的元素（排好序之后就是index=2）放在左边，比其大的放在右边（右边的未排序）
# index = np.argpartition(arr, 2)  #可以加一个[2]  就可以得到值
# print(index)
# print(arr[index])
# print('\n')
# #将元素第2大的元素（排好序之后就是index=2）放在右边，比其大的放在左边（左边的未排序）
# index = np.argpartition(arr, -2)  #可以加一个[-2]  就可以得到值
# print(index)
# print(arr[index])


#argmax() 沿给定轴返回最大元素的索引
#atgmin() 沿给定轴返回最小元素的索引

# a = np.array([[30,40,70],[80,20,10],[50,90,60]])
# print('我的数组是')
# print(a)
# print('\n')
# print('调用argmax')
# print(np.argmax(a)) #没有axis  就是展开数组之后最大元素的下标
# print('展开数组：')
# print(a.flatten())

# print('\n')
# print('沿轴0的最大值索引')
# maxindex = np.argmax(a, axis=0)
# print(maxindex)

# print('\n')
# print('沿轴1的最大值索引')
# maxindex2 = np.argmax(a, axis=1)
# print(maxindex2)

# # argmin 同理


# #numpy.nonzero()  返回输入数组中非零元素的索引
# a = np.array([[30,40,0],[0,20,0],[50,0,60]])
# print(np.nonzero(a)) #下标  结果(0,0),(0,1),(1,1),(1,1),(2,0),(2,2)
# print(a[np.nonzero(a)])

#numpy.where() 
#返回满足给定条件元素的索引
# x = np.arange(9.0).reshape(3,3)
# x = np.arange(9.0, step=0.1) #   range(start, stop[, step]) 
print(x)
# print('我们的数组是：')
# print(x)
# print('大于3的元素的索引')
# y = np.where(x>3)
# print(y)
# print('使用这些索引来获取满足条件的元素')
# print(x[y])


#numpy.extract()  
#根据某个条件从数组中抽取元素，返回满足条件的元素
x = np.arange(9.0).reshape(3,3)
print('我们的数组是：')
print(x)
# 定义条件，选择偶数元素
condition = np.mod(x,2) == 0 #true的数组表达式
print('按元素的条件值')
print(condition)
print('使用条件提取元素')
np.extract(condition, x)  # numpy.extract([条件], x)


# In[99]:


import  numpy as np

#多字节对象都被存储为连续的字节序列。字节顺序，是跨越多字节的程序对象的存储规则
#大端模式： 指数据的高字节保存在内存的低地址中，而数据的低字节把保存在内存的高地址中，
#有点类似把数据当做字符串顺序处理：地址由小向大增加，而数据从高往低位放。（符合阅读）

#小端模式：数据的高字节存在内存的高地址中，而数据的低字节保存在内存的低地址中，将地址
#的高低和数据位权有效的结合起来，高地址部分权值高，低地址部分权值低。

# numpay.ndarray.byteswap()  将ndarray中每个元素中的字节进行大小端转换
a = np.array([1, 256, 8755], dtype=np.int16)
print('数组是：')
print(a)
print('以十六进制表示内存中的数据：')
print(list(map(hex, a)))  #map返回一个迭代器，因此需要传入list函数
print('调用byteswap')
# byteswap()函数通过传入true来原地交换
print(a.byteswap(True))
print(list(map(hex, a)))


# In[112]:


import numpy as np

#副本和视图
#副本是一个数据的完整拷贝，如果对副本进行修改，它不会影响到原始数据，
#物理内存不在同一个位置
#视图树数据的一个别称或引用，通过该别称或引用亦便可访问、操作原有数据，
#但原有数据不会产生拷贝。如果我们对视图进行修改，它会影响到原始数据，物理内存在同一位置

#视图一般发生在：
#（1）numpy的切片操作返回原数据的视图
#（2）调用ndarray的view()函数产生一个视图
#副本一般发生在：
#python序列切片操作，调用deepCopy()函数
#调用ndarray的copy()函数产生一个副本

#无复制  简单理解为 直接赋值

# a = np.arange(6)
# print('我们的数组是：')
# print(a)
# print('调用id()函数：')
# print(id(a))
# print('a 赋值给 b：')
# b = a
# print(b)
# print('b 拥有相同id()')
# print(id(b))

# b.shape = 3, 2
# print(b)
# print(a)

#视图或浅拷贝
#ndarray.view()创建一个新的数组对象，新数组的维数更改不会影响到原始数组的维数
a = np.arange(6).reshape(3, 2)
print('数组a:')
print(a)
print('创建a的视图')
b = a.view()
print(b)
print(id(a) == id(b))
b.shape=(2, 3)
print(b)
print(a)

b[0, 0] = 234  #修改数组里面的值 就会影响原始数组
print(a)
print(b)


# #使用切片创建视图修改数据会影响到原始数组：
# arr = np.arange(12)
# print('数组：')
# print(arr)
# print('创建切片')
# a = arr[3:]
# b = arr[3:]
# a[1] = 123
# 
# b[2] = 234
# print(arr)
# print(id(a), id(b), id(arr[3:]))
# #变量a，b都是arr的一部分视图，对视图的修改会直接反映到原数据中，但是id确是不同的，
# # 视图虽然指向原数据，但是他们和赋值引用还是有区别的

# ndarray.copy() 创建一个副本，对副本数据进行修改，不会影响到原始数据，
#他们物理内存不在同一位置

# a = np.array([[10,10], [2,3], [4,5]])
# print('数组a:')
# print(a)
# print('创建a的深层副本：')
# b = a.copy()
# print('数组b：')
# print(b)

# b[0,0] = 100
# print('修改后的数组b：')
# print(b)
# print('a 保持不变：')
# print(a)


# In[24]:


import numpy.matlib
import numpy as np

#要记得导入numpy.matlib
#矩阵
#numpy.matlib 该模块中的函数返回的是一个矩阵，而不是ndarray对象
#一个mxn的矩阵是一个由m行n列元素排列成的矩形阵列
#矩阵里的元素可以是数字、符号或数学式。

#numpy.matlib.empty(shape, dtype, order)
# shape 定义新矩阵形状的整数或整数元组
# Dtype 可选，数据类型
# order C(行序优先) 或者F（列序优先）

# print(np.matlib.empty((2,2)))

#numpy.matlib.zeros(shape, dtype, order)
#函数创建一个以0填充的矩阵
# print(np.matlib.zeros((2,2), np.float_, 'c'))

#numpy.matlib.ones(shape, dtype, order)
#函数创建一个以1填充的矩阵
# print(np.matlib.ones((2,2)))

#numpy.matlib.eye(n, M, k, dtype)
#函数创建一个对角矩阵，对角元素为1，其余元素为0
# n 返回矩阵的行数
# M 返回矩阵的列数，默认为n
# k 对角线的索引
# dtype 数据类型
# print(np.matlib.eye(n = 3, M = 4, k = 0, dtype = float))

#numpy.matlib.identity()  函数返回给定大小的单位矩阵
# 单位矩阵是个方阵，从左上角到右下角的对角线（主对角线）上的元素均为1，其余元素为0
# np.matlib.identity(n, dtype)
# print(np.matlib.identity(3, dtype=float))

#numpy.matlib.rand()
#函数创建一个给定大小的矩阵，数据时随机填充的
# np.matlib.rand(shape)
# print(np.matlib.rand(3,3))

#注意：矩阵总是二维的，而ndarray是一个n维数组，两个对象都是可互换的
i = np.matrix('1,2;3,4')
print(i)
print("\n")
j = np.asarray(i)
print(j)
print("\n")
k = np.asarray(j)
print(k)


# In[47]:


import numpy as np

#numpy 提供了线性代数函数库linalg 该库包含了线性代数所需的所有功能
#numpy.dot(a, b, out = None)  两个数组的点积，对应元素的乘积
# 对于两个一维的数组，计算的是这两个数组对应下标元素的乘积和(数学上称之为内积)；
# 对于二维数组，计算的是两个数组的矩阵乘积；对于多维数组，它的通用计算公式如下，即结果数组中的每个元素都是：
# 数组a的最后一维上的所有元素与数组b的倒数第二位上的所有元素的乘积和： dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])。
# 只需a矩阵的最后一维dim等于b矩阵倒数第二维dim即可，
# 对应二维情况就是第一个的列数等于第二个矩阵行数；
# 也就是说点积发生在a,b矩阵最后两个维度上；

# a —— ndarray 数组，
# b  —— ndarray数组
# out —— ndarray 可选，用来保存dot()的计算结果

# a = np.array([[[1,2], [3,4]], [[1,2], [3,4]]])
# b = np.array([[[11,12], [13,14]], [[11,12], [13,14]]])
# print(np.dot(a,b))

#计算式为， [[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]

# #numpy.vdot()
# # 两个向量的点积，如果第一个参数是复数，那么他的共轭复数会用于计算，
# # 如果参数是多维数组，它会被展开
# a = np.array([[1,2], [3,4]])
# b = np.array([[11,12], [13,14]])
# print(np.vdot(a, b))
# #计算式为：1*11 + 2*12 + 3*13 + 4*14 = 130
# # 也就是对这两个向量对弈位一一相乘之后，求和的操作

# #numpy.inner() 内积
# # 函数返回一维数组的向量内积，对于更高的维度，它返回最后一个轴上的和的乘积
# # print(np.inner(np.array([1,2,3]), np.array([0,1,0])))
# # # 多维数组，每一行相乘结果勋在对应的行上
# # print(np.inner(np.array([[1,2,3]]), np.array([[0,1,0]])))
# a = np.array([[1,2], [3,4]])
# b = np.array([[11,12], [13, 14]])
# print(np.inner(a,b))
# a = np.array([[1,2], [3,4]])
# b = np.array([[11,12], [13, 14]])
# print(np.inner(a,b))

# #计算式：
# # 1*11+2*12, 1*13+2*14 
# # 3*11+4*12, 3*13+4*14


# numpy.matmul 函数返回两个数组的矩阵乘积，虽然它返回二维数组的正常乘积，
# 但如果任一参数的维数大于2，则将其视为存在于最后两个索引的矩阵的栈，
# 并进行相应的广播

# 如果任一参数是一维数组，则通过在其维度上附加1来将其提升为矩阵，
# 并在乘法之后被去除。

# a = [[1,0], [0,1]]
# b = [1,2]

# a = [[1,0], [0,1]]
# b = [[4,1], [2,2]]

a = np.arange(8).reshape(2,2,2)
print(a)
b = np.arange(4).reshape(2,2)
print(b)
print(np.matmul(a,b))
# print(np.matmul(b,a))


# In[54]:


import numpy as np

#计算矩阵的行列式
# numpy.linalg.det()
# 满足行列式的计算方式，对于2X2的行列式，主对角线与次对角线乘积之差
# 任意大的方阵，都可以看做是2X2的矩阵组合

# a = np.array([[1,2], [3,4]])
# print(np.linalg.det(a))

# #numpy.linalg.solve() 矩阵形式的线性方程的解
# a = [[1,1,1],[0,2,5],[2,5,-1]]
# b = [6,-4,27]
# print(np.linalg.solve(a,b))

#numpy.linalg.inv()  计算矩阵的乘法逆矩阵
# 逆矩阵，设A是数域上的一个n阶矩阵，若在相同数域上存在另一个n阶矩阵B
# 使得AB = BA = E，则我们称B是A的逆矩阵，A是可逆矩阵 E 单位矩阵

a = np.array([[1,2], [3,4]])
y = np.linalg.inv(a)
print(y)


# In[63]:


import numpy as np

# numpy 为ndarray对象引入了一个简单的文件格式 npy
# npy文件用于存储重建ndarray所需的数据、图形、dtype和其他信息

# numpy.save(file, arr, allow_pickle=True, fix_imports=True)  函数保存以.npy 为扩展名的文件中
# file:要保存的恩建 扩展名为.npy 如果文件路径末尾没有扩展名.npy
# 该扩展名会被自动加上
# arr 要保存的素组
# allow_pickle 可选，布尔值，允许使用python pickles 保存对象数组，
# python中的pickle用于在保存到的磁盘文件或从磁盘文件读取之前，
# 对对象进行序列化和反序列化
# fix_imports  为了方便python2中读取python3保存的数据

# a = np.array([1,2,3,4,5])
# np.save('outfile.npy', a)

# # 使用load函数 读取
# b = np.load('outfile.npy')
# print(b)

# # numpy.savez(file, *args, **kwds) 函数将多个数组保存到以npz为扩展名的文件中
# # file 要保存的文件，扩展名为.npz，如果文件路径末尾没有扩展名，该扩展名会被自动加上
# # args 要保存的数组，可以使用换尖子参数为数组起一个名字，
# # 非关键字参数传递的数组会自动起名为 arr_0, arr_1
# # kwds 要保存的数组使用关键字名称
# a = np.array([[1,2,3], [4,5,6]])
# b = np.arange(0, 1.0, 0.1)
# c = np.sin(b)
# np.savez("runoob.npz", a, b, sin_array = c)
# r = np.load("runoob.npz")
# print(r["arr_0"])
# print(r["arr_1"])
# print(r["sin_array"])

# 以简单的文本文件格式存储数据
# numpy.savetxt(FILENAME, dtype=int, delimiter='')  写数据 存
# numpy.loadtxt(FILENAME, arr, fmt="%d", delimiter='')   读数据

# 参数delimiter可以指定各种分隔符，针对特定列的转换器函数，需要跳过的行数
a = np.arange(0, 10, 0.5).reshape(4, -1) #这个-1代表的意思就是，我不知道可以分成多少列，用在行上面一样
np.savetxt("out.txt", a, fmt='%d', delimiter=",") # 以整数形式保存，用逗号分隔
b = np.loadtxt("out.txt", delimiter=",") #以存入的时候的分隔符一样
print(b)


# In[80]:


#-*-conding:utf-8-*-

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

# %matplotlib inline这个必须加，否则在jupyter无法出图
# fname 为 你下载的字体库路径，注意 SimHei.ttf 字体的路径
zhfont1 = matplotlib.font_manager.FontProperties(fname="SimHei.ttf") 


#%matplotlib具体作用是当你调用matplotlib.pyplot的绘图函数plot()进行绘图的时候，
#或者生成一个figure画布的时候，可以直接在你的python console里面生成图像。
#Matplotlib 是python的会图库，他可与NUmpy一起使用提供了一种有效的MatLab开源替代方案。
# 它也可以和图形工具包一起使用，如PyQt和wxPython

# x = np.arange(1, 11)
# y = 2*x + 5
# # fontproperties 设置中文显示，fontsize 设置字体大小
# plt.title("demo图形", fontproperties=zhfont1)
# plt.xlabel("x轴", fontproperties=zhfont1)
# plt.ylabel("y轴", fontproperties=zhfont1)
# plt.plot(x, y, "ob") # "ob" 表示图形显示的数字对应的符号 下图为蓝色的圆点
# plt.show() 

# x = np.arange(0, 3*np.pi, 0.1)
# y = np.sin(x)
# plt.title("sine wave form")
# plt.plot(x, y)
# plt.show()

# #subplot() 同一图中绘制不同的东西
# x = np.arange(0, 3*np.pi, 0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)
# #建立subplot网格，高为2，宽为1
# #激活第一个subplot
# plt.subplot(2,1,1)
# plt.plot(x, y_sin)
# plt.title('Sine')
# #将第二个subplot激活，并绘制第二个图像
# plt.subplot(2,1,2)
# plt.plot(x, y_cos)
# plt.title('Cosine')
# plt.show()


# #bar() 
# # pyplot子模块提供bar() 函数来生成条形图
# x = [5, 8, 10]
# y = [12, 16, 6]
# x2 = [6, 9, 11]
# y2 = [6, 15, 7]
# plt.bar(x, y, align = 'center')
# plt.bar(x2, y2, align = 'center', color = 'g')
# plt.show()


#numpy.histogram()
# 数据的频率分布的图形表示。水平尺寸相等的矩形对应于类间隔，称为bin，
# 变量height对应于频率
# 函数将输入数组和bin作为两个参数，bin数组中的连续元素用作每个bin的边界
# a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
# np.histogram(a,bins =  [0,20,40,60,80,100]) 
# hist,bins = np.histogram(a,bins =  [0,20,40,60,80,100])  
# print (hist) 
# print (bins)
# plt.plot(bins, hist)
# plt.show()

#plt()  将直方图的数字表示转换为图形。
# pyplot子模块的plt()函数包含数据和bin数组作为参数，并转换为直方图
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
plt.hist(a,bins =  [0,20,40,60,80,100]) 
plt.show()


# In[ ]:




