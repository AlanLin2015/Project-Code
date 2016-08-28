# -*- coding: utf-8-*-
from pandas import Series,DataFrame
from numpy import *
import  pandas as pd
import csv
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

def chanpin():
    chanpin=pd.read_csv('C:\Users\Alan Lin\Desktop\\chanpin.csv')
    tolist=chanpin['tid'].tolist()
    toset=set(tolist)
    list1=list(toset)
    allorder=[]
    for i in list1:
        everyorder=chanpin[chanpin['tid'] == i]['cpbm'].tolist()
        allorder.append(everyorder)
    return allorder

def leibie():
    leibie=pd.read_table('C:\Users\Alan Lin\Desktop\\leibie.txt',sep="\t",names=['tid','mc'],low_memory=False)
    tolist=leibie['tid'].tolist()
    toset=set(tolist)
    list1=list(toset)
    allleibie=[]
    for i in list1:
        everyorder=leibie[leibie['tid'] == i]['mc'].tolist()
        allleibie.append(everyorder)
    return allleibie
    
def createC1(dataset):
    sub=[]
    for line in dataset:
        for i in line:
            if [i] not in sub:
                sub.append([i])
    sub.sort()
    return map(frozenset,sub)
    
#C1=createC1(dataset)

def scanD(D,C1,minsupport=0.7):
    ssdict={}
    L=[]
    supportData={}
    for tid in D:
        for i in C1:
            if i.issubset(tid):
                if i not in ssdict:
                    ssdict[i] = 1
                else:
                    ssdict[i] += 1
    #print 'ssdict:',ssdict
    num=float(len(D))
    for key in ssdict:
       # print 'key:',key
        support=ssdict[key]/num
       # print 'support:',support
        if support >= minsupport:
            L.insert(0,key)
            supportData[key] = support
    return L,supportData
    
def apriorizuhe(lk,k):
    lenlk=len(lk)
    readlist=[]
    for i in range(lenlk):
        for j in range(i+1,lenlk):
            L1=list(lk[i])[:k-2];L2=list(lk[j])[:k-2]
            if L1 == L2:
                readlist.append(lk[i]|lk[j])
    return readlist


def main(dataset,minsupport=0.7):
    D=map(set,dataset)
    C1=createC1(dataset)
    L,supportData=scanD(D,C1,minsupport)
    L=[L]
    k=2
    while(len(L[k-2])>0):
        ck=apriorizuhe(L[k-2],k)
        L1,supportdata=scanD(D,ck,minsupport)
        L.append(L1)
        supportData.update(supportdata)
        k += 1
    return L,supportData


'''关联分析'''

def generateRules(L,supportData,minconf=0.7):  #minconf为可信度
    bigrulelist=[]   #新建列表用于储存关联信息
    for i in range(1,len(L)):   #从第二个开始遍历每一个由频繁项集组成的列表
        for freqset in L[i]:   #从列表里遍历每一个频繁项集
            H1=[frozenset([item]) for item in freqset]   #对频繁项集里的每个项提出来化为frozenset的形式储存在列表中，如[frozenset([1]),frozenset([2])]
            print 'H1:',H1
            if (i > 1):   #因为第二行的频繁项集里的项都只有2个，所以选择大于二行的进行迭代求解，第一行只有一个直接忽略
                H1=clacconf(freqset,H1,supportData,bigrulelist,minconf)   #先算第二层匹配
                rulesfromconseq(freqset,H1,supportData,bigrulelist,minconf)              
            else:
                clacconf(freqset,H1,supportData,bigrulelist,minconf)   #直接求每个频繁项作为后项的可信度，并保留可信度符合要求的项
    return bigrulelist

def clacconf(freqset,H,supportData,bigrulelist,minconf):   #输入频繁项集如frozenset([0,1])，H值作为后项，形式如[frozenset([0]),frozenset([1])]
    returnlist=[]
    for conseq in H:   #对频繁项集里的每个项都假设是后项，计算该可信度
        a=supportData[freqset]/supportData[freqset-conseq]
        if a >= minconf:   #若该可信度符合要求，则输出该后项
            print freqset-conseq,'-->',conseq, 'conf:',a
            bigrulelist.append([freqset-conseq,conseq,[a]])
            returnlist.append(conseq)
    return returnlist
    
def rulesfromconseq(freqset,H,supportData,bigrulelist,minconf):   #当频繁项集的内容大于1时，如frozenset([0,1,2,3]),其H值为[frozenset([0]),frozenset([1]),...frozenset([3])]
    if len(H) == 0:   #如果上一层没有匹配上则H为空集
        pass
    else:
        m=len(H[0])   #计算H值的第一个值的长度
        if (len(freqset) > (m+1)):   #若freqset的长度大于m+1的长度，则继续迭代
            hmp=apriorizuhe(H,m+1)   #将单类别加类别，如{0,1,2}转化为{0,1},{1,2}等
            print 'hmp:',hmp
            hmp=clacconf(freqset,hmp,supportData,bigrulelist,minconf)   #计算可信度
            if (len(hmp) > 1):   #如果后项的数量大于1，则还有合并的可能，继续递归
                rulesfromconseq(freqset,hmp,supportData,bigrulelist,minconf)
#处理bigrulelist，处理成列表形式，容易输出            
def bigrulelistchuli(bigrulelist):
    always=[]
    for i in bigrulelist:
        m=map(list,i)
        nn=[]
        for j in m:
            j=j[0]
            nn.append(j)
        always.append(nn)
    return always
#输出列表
def outputtxt(data):
    csvfile=file('C:\Users\Alan Lin\Desktop\\always.txt','wb')
    writer=csv.writer(csvfile)
   # writer.writerow(['a','b','c'])
    data=data
    writer.writerows(data)
    csvfile.close()

def outputcsv(data):
    csvfile=file('C:\Users\Alan Lin\Desktop\\always.csv','wb')
    writer=csv.writer(csvfile)
   # writer.writerow(['a','b','c'])
    data=data
    writer.writerows(data)
    csvfile.close()
