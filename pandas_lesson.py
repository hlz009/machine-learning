#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

#读取文件
# 查看类型
idCardInfo = pd.read_csv("idcard.csv", encoding="gbk")
print(type(idCardInfo))
print(idCardInfo.dtypes)


# In[7]:


# idCardInfo.head() #  不传参数默认取得前5条
idCardInfo.head(7)   #指定参数，获取前面的数据


# In[9]:


# idCardInfo.tail()  #获取后面的数据，不传递参数 则默认5条
idCardInfo.tail(3)  


# In[10]:


idCardInfo.columns  #获取数据的列


# In[11]:


idCardInfo.shape #获取数组的形状


# In[13]:


# idCardInfo.loc[0]  #获取第一行数据

idCardInfo.loc[1]  #获取第二行数据


# In[14]:


idCardInfo.loc[3:6]  #使用切片  选取下标第3个到第6个 
#（注意这个包含了右边索引的边界值）


# In[15]:


idCardInfo["name"] #获取列的元素


# In[16]:


idCardInfo[['name', 'confidence']] #读取两列值


# In[18]:


confidence = idCardInfo['confidence']/100
print(confidence)


# In[19]:


idCardInfo["confidence"].max() #求最大值

idCardInfo["confidence"].min() #求最小值

idCardInfo["confidence"].mean() # 求平均值


# In[29]:


'''
值排序
'''
idCardInfo.sort_values(["money", "confidence"], inplace = True, ascending = True)
print(idCardInfo)
# help(idCardInfo.sort_values)
# print(idCardInfo)


# In[31]:


idCardInfo1 = pd.read_csv("idcard01.csv", encoding="gbk")
print(idCardInfo1)


# In[32]:


pd.isnull(idCardInfo1["confidence"])


# In[33]:


#去掉空值 NAN
conf_is_null = pd.isnull(idCardInfo1["confidence"])
idCardInfo1["confidence"][conf_is_null == False]


# In[47]:


import numpy as np
#分组的概念

idCardInfo1.pivot_table(index="sex", values="money", aggfunc = np.sum)
# help(idCardInfo1.pivot_table)
# print(idCardInfo1)
# help(idCardInfo1.pivot)
# idCardInfo1.pivot(index="sex", values="money")


# In[66]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plt.plot()
# plt.show()
d = idCardInfo.sort_values("confidence", ascending=True).head()
print(d)
plt.xlabel("confidence")
plt.yticks(rotation=45)
plt.ylabel("money")
plt.title("confidence-money")
plt.plot(d["confidence"], d["money"])
plt.show()


# In[67]:


#matplotlib画多图

fig = plt.figure()
fig1 = fig.add_subplot(2,2,1)
fig2 = fig.add_subplot(2,2,4)
plt.plot()
plt.show()


# In[68]:


fig = plt.figure()
fig1 = fig.add_subplot(2,2,1)
fig2 = fig.add_subplot(2,2,4)
fig1.plot(idCardInfo["money"], idCardInfo["confidence"])
fig2.plot(idCardInfo["confidence"], idCardInfo["money"])
plt.show()


# In[70]:


import numpy as np

# figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)

# num:图像编号或名称，数字为编号 ，字符串为名称
# figsize:指定figure的宽和高，单位为英寸；
# dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
# facecolor:背景颜色
# edgecolor:边框颜色
# frameon:是否显示边框

fig = plt.figure(figsize=(16,6))
colors = ['red', 'blue', 'green', 'orange', 'black']
h = np.arange(10, 100, 10)
for i in range(5):
    v = 1000*i + np.random.randn(9,1)*1000
    label = str(i*10)
    plt.plot(h, v, c = colors[i], label=label)
plt.legend(loc='best') #显示折线图标签，即右下角的小图
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = np.random.randn(2, 100)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].hist(data[0]) #直方图
axs[1, 0].scatter(data[0], data[1]) #散点图 
axs[0, 1].plot(data[0], data[1]) #折线图
axs[1, 1].hist2d(data[0], data[1]) #二维矩阵图
plt.show()


# In[6]:


help(np.random.randn)


# In[11]:


np.random.randn(2, 100)


# In[ ]:




