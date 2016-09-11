# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

print('初始化路径和变量')
names=['user_id','item_id','rating','timestamp']
training_file='C:\Users\Alan Lin\Desktop\machinelearningdata\data\\u3.base'
testset_file='C:\Users\Alan Lin\Desktop\machinelearningdata\data\\u3.test'
n_users=943
n_items=1682
ratings=np.zeros((n_users,n_items))   #建立评分矩阵
df=pd.read_csv(training_file,sep='\t',names=names)   #打开文件
print (df.head())   #打印该数据的前五行
for row in df.itertuples():   #遍历每一行
    ratings[row[1]-1,row[2]-1]=row[3]   #将每个user和item对应的评分加到评分矩阵里


def cal_sparsity():   #计算矩阵密度
    sparsity = float(len(ratings.nonzero()[0]))   #空余的或为零的位置个数
    sparsity /= (ratings.shape[0]*ratings.shape[1])   #行乘以列，总位置数
    sparsity *= 100
    print ('训练集矩阵密度为: {:4.2f}%'.format(sparsity))


def rmse(pred,actual):   #计算预测结果的rmse
    from sklearn.metrics import mean_squared_error
    pred=pred[actual.nonzero()].flatten()
    actual=actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred,actual))
    

    
''' 基线算法(baseline) '''
def cal_mean():
    global all_mean,user_mean,item_mean   #定义全局变量
    all_mean = np.mean(ratings[ratings!=0])   #计算全局均值
    user_mean = sum(ratings.T)/sum((ratings!=0).T)   #计算每个user的均值
    #print 'user_mean:',user_mean
    item_mean = sum(ratings)/sum((ratings!=0))   #计算每个item的均值
    #print 'item_mean:',item_mean
    
    user_mean_nan='是'
    item_mean_nan='是'
    
    if np.isnan(user_mean).any():   #如果不存在任何的缺值的话，返回“否”
        user_mean_nan='否'
    if np.isnan(item_mean).any():   #如果不存在任何的缺值的话，返回“否”
        user_mean_nan='否'
    print '是否存在User均值为NaN?',user_mean_nan
    print '是否存在Item均值为NaN?',item_mean_nan
    
    print '对NaN填充总体均值...'
    
    user_mean=np.where(np.isnan(user_mean),all_mean,user_mean)   #用全局均值来填充空缺位置
    item_mean=np.where(np.isnan(item_mean),all_mean,item_mean)
    
    if np.isnan(user_mean).any():
        user_mean_nan='否'
    if np.isnan(item_mean).any():
        user_mean_nan='否'
    print '是否存在User均值为NaN?',user_mean_nan
    print '是否存在Item均值为NaN?',item_mean_nan
    print '均值计算完成，总体打分均值为 %.4f' % all_mean
    
def predict_naive(user,item):   #每个点的评分用item+user-全局均值来计算
    prediction = item_mean[item] + user_mean[user] - all_mean
    return prediction
    

def testbaseline(testset_file,rmse=rmse,predict_naive=predict_naive):
    print('------ 基线算法(baseline) ------')
    cal_mean()
    print '载入测试集...'
    test_df = pd.read_csv(testset_file,sep='\t',names=names)
    test_df.head()
    predictions=[]
    targets=[]
    print ('测试集大小为 %d' % len(test_df))
    print ('采用基线算法进行预测...')
    for row in test_df.itertuples():
        user,item,actual = row[1]-1,row[2]-1,row[3]
        predictions.append(predict_naive(user,item))   #预测的得分
        targets.append(actual)   #实际的得分
    print ('测试结果的RMSE为 %.4f' % rmse(np.array(predictions),np.array(targets)))

''' item_based协同过滤算法（相似度未归一化） '''

def cal_similarity(ratings,kind,epsilon=1e-9):   #余弦距离相似度，epsilon防止分母为零
    if kind =='user':
        sim = ratings.dot(ratings.T) + epsilon   #两两用户矩阵相乘
    elif kind =='item':
        sim = ratings.T.dot(ratings) + epsilon   #两两物品矩阵相乘
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
    
def real_similarity(ratings):   #计算相似度矩阵并打印
    print ('计算相似度矩阵...')
    global user_similarity,item_similarity
    user_similarity = cal_similarity(ratings,kind = 'user')
    item_similarity = cal_similarity(ratings,kind = 'item')
    print ('计算完成')
    print ('相似度矩阵样例:(item-item)')
    print (np.round(item_similarity[:10,:10],3))
    
def predict_itemCF(user,item,k=100):   #item_based协同过滤算法
    nzero = ratings[user].nonzero()[0]
    prediction = ratings[user,nzero].dot(item_similarity[item,nzero])/sum(item_similarity[item,nzero])   #为何要除以相似度总和？
    print sum(item_similarity[item,nzero])
    return prediction

def test_item_based(testset_file,ratings):
    print('------ item-based协同过滤算法(相似度未归一化) ------')
    real_similarity(ratings)
    print ('载入测试集')
    test_df = pd.read_csv(testset_file,sep='\t',names=names)
    test_df.head()
    predictions=[]
    targets=[]
    print ('测试集大小为 %d' % len(test_df))
    print ('采用item-based协同过滤算法进行预测...')
    for row in test_df.itertuples():
        user,item,actual = row[1]-1,row[2]-1,row[3]
        predictions.append(predict_itemCF(user,item))   #预测的得分
        targets.append(actual)   #实际的得分
    print ('测试结果的RMSE为 %.4f' % rmse(np.array(predictions),np.array(targets)))
    
    
'''结合基线算法的item_based协同过滤算法(相似度未归一化) '''

def predict_itemCF_baseline(user,item,k=100):
    nzero = ratings[user].nonzero()[0]
    baseline = item_mean + user_mean[user] - all_mean
    prediction = (ratings[user, nzero] - baseline[nzero]).dot(item_similarity[item, nzero])\
                / sum(item_similarity[item, nzero]) + baseline[item]
    return  prediction
    
def test_item_based_baseline(testset_file,ratings):
    print('------ 结合基线算法的item-based协同过滤算法(相似度未归一化) ------')
    cal_mean()   #引入基线算法
    real_similarity(ratings)   #item-based协同过滤算法
    print ('载入测试集')
    test_df = pd.read_csv(testset_file,sep='\t',names=names)
    test_df.head()
    predictions=[]
    targets=[]
    print ('测试集大小为 %d' % len(test_df))
    print('采用结合baseline的item-item协同过滤算法进行预测...')
    for row in test_df.itertuples():
        user,item,actual = row[1]-1,row[2]-1,row[3]
        predictions.append(predict_itemCF_baseline(user,item))   #预测的得分
        targets.append(actual)   #实际的得分
    print ('测试结果的RMSE为 %.4f' % rmse(np.array(predictions),np.array(targets)))
    
    
''' user-based协同过滤算法(相似度未归一化) '''

def predict_userCF(user,item,k=100):   #user-user协同过滤算法，预测rating
    nzero = ratings[:,item].nonzero()[0]
    baseline = user_mean + item_mean[item] - all_mean
    prediction = ratings[nzero,item].dot(user_similarity[user,nzero]) / sum(user_similarity[user,nzero])
    if np.isnan(prediction):   #冷启动问题：该item暂时没有得分
        prediction = baseline[user]
    return prediction
    
def test_user_based(testset_file,ratings):
    print('------ user-based协同过滤算法(相似度未归一化) ------')
    cal_mean()   #引入基线算法
    real_similarity(ratings)   #item-based协同过滤算法
    print ('载入测试集')
    test_df = pd.read_csv(testset_file,sep='\t',names=names)
    test_df.head()
    predictions=[]
    targets=[]
    print ('测试集大小为 %d' % len(test_df))
    print('采用user-user协同过滤算法进行预测...')
    for row in test_df.itertuples():
        user,item,actual = row[1]-1,row[2]-1,row[3]
        predictions.append(predict_userCF(user,item))   #预测的得分
        targets.append(actual)   #实际的得分
    print ('测试结果的RMSE为 %.4f' % rmse(np.array(predictions),np.array(targets)))
    
''' 结合基线算法的的user-user协同过滤算法(相似度未归一化) '''

def predict_userCF_baseline(user,item,k=100):   #结合baseline的user-user协同过滤算法预测rating
    nzero = ratings[:,item].nonzero()[0]
    baseline = user_mean + item_mean[item] - all_mean
    prediction = (ratings[nzero,item] - baseline[nzero]).dot(user_similarity[user,nzero])\
                / sum(user_similarity[user,nzero]) + baseline[user]
    if np.isnan(prediction):
        prediction = baseline[user]
    return prediction
    
def test_user_based_baseline(testset_file,ratings):
    print('------ 结合基线算法的的user-user协同过滤算法(相似度未归一化) ------')
    cal_mean()   #引入基线算法
    real_similarity(ratings)   #item-based协同过滤算法
    print ('载入测试集')
    test_df = pd.read_csv(testset_file,sep='\t',names=names)
    test_df.head()
    predictions=[]
    targets=[]
    print ('测试集大小为 %d' % len(test_df))
    print('采用结合baseline的user-user协同过滤算法进行预测...')
    for row in test_df.itertuples():
        user,item,actual = row[1]-1,row[2]-1,row[3]
        predictions.append(predict_userCF_baseline(user,item))   #预测的得分
        targets.append(actual)   #实际的得分
    print ('测试结果的RMSE为 %.4f' % rmse(np.array(predictions),np.array(targets)))

'''经过上下界限修正后的item_based_baseline协同过滤'''

def predict_biasCF(user,item,k=100):   #结合基线算法的item_basedCF算法预测ratings
    nzero = ratings[user].nonzero()[0]
    baseline = item_mean + user_mean[user] - all_mean
    prediction = (ratings[user,nzero] - baseline[nzero]).dot(item_similarity[item,nzero])\
                / sum(item_similarity[item,nzero]) + baseline[item]
    if prediction >5:
        prediction = 5
    if prediction <1:
        prediction = 1
    return prediction
    
def test_biasCF(testset_file,ratings):
    print('------ 经过修正后的协同过滤 ------')
    cal_mean()   #引入基线算法
    real_similarity(ratings)   #item-based协同过滤算法
    print ('载入测试集')
    test_df = pd.read_csv(testset_file,sep='\t',names=names)
    test_df.head()
    predictions=[]
    targets=[]
    print ('测试集大小为 %d' % len(test_df))
    print('采用结合baseline的item-based协同过滤算法进行预测...')
    for row in test_df.itertuples():
        user,item,actual = row[1]-1,row[2]-1,row[3]
        predictions.append(predict_biasCF(user,item))   #预测的得分
        targets.append(actual)   #实际的得分
    print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))


'''Top-k协同过滤(item-based + baseline+修正+top-K)'''

def predict_topkCF(user,item,k=10):
    nzero = ratings[user].nonzero()[0]
    baseline = item_mean + user_mean[user] - all_mean
    choice = nzero[item_similarity[item,nzero].argsort()[::-1][:k]]
    print 'choice:',choice
    prediction = (ratings[user,choice] - baseline[choice]).dot(item_similarity[item,choice])\
                 / sum(item_similarity[item,choice]) + baseline[item]
    if prediction > 5:
        prediction = 5
    if prediction < 1:
        prediction = 1
    return prediction
    
def test_topkCF(testset_file,ratings):
    print('------ Top-k协同过滤(item-based + baseline)------')
    cal_mean()   #引入基线算法
    real_similarity(ratings)   #item-based协同过滤算法
    print ('载入测试集')
    test_df = pd.read_csv(testset_file,sep='\t',names=names)
    test_df.head()
    predictions=[]
    targets=[]
    print ('测试集大小为 %d' % len(test_df))
    print('采用top K协同过滤算法进行预测...')
    k = 20
    for row in test_df.itertuples():
        user,item,actual = row[1]-1,row[2]-1,row[3]
        predictions.append(predict_topkCF(user,item))   #预测的得分
        targets.append(actual)   #实际的得分
    print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))



'''超级大招:item_based + baseline + top-K + 修正 + 相似度矩阵归一化'''

def cal_similarity_normal(ratings,kind,epsilon=1e-9):   #pearson相似度矩阵计算
    if kind == 'user':   #对同一个user的打分归一化
        ratings_user_diff = ratings.copy()
        for i in range(ratings.shape[0]):
            nzero = ratings[i].nonzero()
            ratings_user_diff[i][nzero] = ratings[i][nzero] - user_mean[i]
        sim = ratings_user_diff.dot(ratings_user_diff.T) + epsilon
    elif kind == 'item':   #对同一个item的打分归一化
        ratings_item_diff = ratings.copy()
        for j in range(ratings.shape[1]):
            nzero = ratings[:,j].nonzero()
            ratings_item_diff[:,j][nzero] = ratings[:,j][nzero] - item_mean[j]
        sim = ratings_item_diff.T.dot(ratings_item_diff) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
def real_similarity_norm(ratings):
    print '计算归一化的相似度矩阵...'
    global user_similarity_normal,item_similarity_normal
    user_similarity_normal = cal_similarity_normal(ratings,kind = 'user')
    item_similarity_normal = cal_similarity_normal(ratings,kind = 'item')
    print '计算完成'
    print ('相似度矩阵样例:(item-item)')
    print (np.round(item_similarity_normal[:10,:10],3))
    
def predict_normal_CF(user,item,k=20):
    nzero = ratings[user].nonzero()[0]
    baseline = item_mean + user_mean[user] - all_mean
    choice = nzero[item_similarity_normal[item,nzero].argsort()[::-1][:k]]
    prediction = (ratings[user,choice] - baseline[choice]).dot(item_similarity_normal[item,choice])\
                 / sum(item_similarity_normal[item,choice]) + baseline[item]
    if prediction > 5 :
        prediction = 5
    if prediction < 1 :
        prediction =1

    return prediction
    
def test_normal_CF(testset_file,ratings):
    print('------ baseline + item-based + TopK + 归一化矩阵 ------')
    cal_mean()   #引入基线算法，baseline
    real_similarity_norm(ratings) #item-based协同过滤算法 + top-K +归一化矩阵+baseline
    print ('载入测试集')
    test_df = pd.read_csv(testset_file,sep='\t',names=names)
    test_df.head()
    predictions=[]
    targets=[]
    print ('测试集大小为 %d' % len(test_df))
    print('采用归一化矩阵方法，结合其它trick进行预测...')
    k = 13
    print('选取的K值为%d.' % k)
    for row in test_df.itertuples():
        user,item,actual = row[1]-1,row[2]-1,row[3]
        predictions.append(predict_normal_CF(user,item,k))   #预测的得分，加修正
        targets.append(actual)   #实际的得分
    print('测试结果的rmse为 %.4f' % rmse(np.array(predictions), np.array(targets)))
