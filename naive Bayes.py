from sklearn.datasets import load_iris
iris = load_iris()
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
X=iris['data'][:100,:]
Y=iris['target'][:100]
names=iris['target_names']
from sklearn.model_selection import train_test_split
#x为数据集的feature熟悉，y为label.
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

data=np.c_[x_train,y_train.T]
df=pd.DataFrame(data)
df[[4]]=df[[4]].astype(int)

# building naive Bayes model

def pro(df):
    """probability of y(ck),data is a dataframe"""
    y=df[4]
    from collections import Counter
    dic = Counter(y)
    for i in dic.keys():
        dic[i]=dic[i]/len(df)
    return (dic)

def find_probability(x,df,n,y):
    """df is a dataframe, n means which degree y stands for ck
    the return is a dictionary which contain probability of x(i) under condition of ck"""
    df_y=df[df[4]==y][n]
    from collections import Counter
    dic=Counter(df_y)
    for i in dic.keys():
        dic[i]=(dic[i]+1)/(len(df_y)+len(dic.keys()))
    for i in dic.keys():
        if i==x:
            return (dic[i])
        else:
            return (1/(len(df_y)+len(dic.keys())))



def probability_given(x,y,df):
    """x 为给值，y是ck，df为dataframe"""
    P=1
    for i in range(len(x)):
        pro=find_probability(x[i],df,i,y)
        P=P*pro
    return (P)


def find_best(x):
    dic = {}
    for j in range(2):
        dic[j] = probability_given(x, j, df)

    return (max(dic, key=dic.get))





# remark
pre=[]
for i in range(len(x_test)):
    y=find_best(x_test[i])
    pre.append(y)

num=0
for i in range(len(x_test)):
    if pre[i]==y_test[i]:
        num=num+1

accurancy=num/len(x_test)


# 总结： bayes 不适用于多分类问题，二分类问题的准确性大幅度上升
























