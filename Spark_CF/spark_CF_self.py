# -*- coding: utf-8 -*-
import sys
import itertools   #循环器
from math import sqrt
from operator import add
from os.path import join,isfile,dirname   #路劲拼接，判断是否文件，返回路径
from pyspark import SparkConf,SparkContext
from pyspark.mllib.recommendation import ALS

def parseRating(line):   #解析u.data
    fields = line.strip().split('\t')
    return long(fields[3]) % 10,(int(fields[0]),int(fields[1]),float(fields[2]))
def parseRating2(line):   #解析personalRating.txt
    fields = line.strip().split(' ')
    return long(fields[3]) % 10,(int(fields[0]),int(fields[1]),float(fields[2]))
def parseMovie(line):   #解析u.item
    fields = line.strip().split('|')
    title = fields[1]
    movieid = (''.join(fields[5:23]))
    return int(movieid),title
    
def loadRatings(ratingfile):   #导入personalRating数据
    if not isfile(ratingfile):   #检测文件是否存在
        print 'file %s is down.'% ratingfile
        sys.exit(1)   #系统服务退出
    f = open(ratingfile,'r')
    ratings = filter(lambda r:r[2]>0,[parseRating2(line)[1] for line in f]) #filter过滤函数
    f.close()
    if not ratings:
        print 'No ratings provided.'
        sys.exit(1)
    else:
        return ratings
        
def computeRmse(model,data,n):   #计算均方根误差，此时test已经只取value部分
    predictions = model.predictAll(data.map(lambda x: (x[0],x[1])))   #计算预测得分
    predictionsAndRatings = predictions.map(lambda x: ((x[0],x[1]),x[2])).join(data.map(lambda x:((x[0],x[1]),x[2]))).values()
    #将预测得分和实际得分join起来，并且用values只取得分
    return sqrt(predictionsAndRatings.map(lambda x:(x[0] - x[1])**2).reduce(add)/float(n))   #计算预测得分的RMSE

if __name__ == '__main__':
    if (len(sys.argv) != 4):   #判断输入参数是否达标
        print 'argv is not enough.'
        sys.exit(1)

    #设定环境
    conf = SparkConf().setAppName('MovieLensALS').set('spark.executor.memory','2g')
    sc = SparkContext(conf = conf)
    #载入personal打分数据，并转换为RDD（pythonRDD）
    myRatings = loadRatings(sys.argv[3])
    myRatingsRDD = sc.parallelize(myRatings,1) 
    #引入data和movie数据文件
    datasys = sys.argv[1]
    itemsys = sys.argv[2]
    #导入数据
    ratings = sc.textFile(datasys).map(parseRating)   #导入data数据并解析
    movies = dict(sc.textFile(itemsys).map(parseMovie).collect())   #导入movie数据并解析并action显示转为字典
    #计算ratings总数目，user数目，movie数目
    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r:r[0]).distinct().count()   #distinct()属于pyspark，去重
    numMovies = ratings.values().map(lambda r:r[1]).distinct().count()
    print 'Got %d ratings from %d users on %d movies.' % (numRatings,numUsers,numMovies)
    
    #根据时间戳最后一位数字把数据集分成训练集（0.6），交叉验证集（0.2），和测试集（0.2）
    #每个集合都是（userId,movieId,ratings）格式的RDD
     
    numPartitions=4   #RDD重分区参数
    #取值，即(userId,movieId,ratings)   #加入myRatingsRDD的数据   #将shuffle为true   #缓存
    training = ratings.filter(lambda x: x[0] < 6) \
      .values() \
      .union(myRatingsRDD) \
      .repartition(numPartitions) \
      .cache()
      
      
    validation = ratings.filter(lambda x: x[0] >=6 and x[0] < 8 )\
                .values()\
                .repartition(numPartitions)\
                .cache()
                
    test = ratings.filter(lambda x:x[0] >= 8).values().cache()
    
    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()
    print 'Training: %d,validation: %d,test:%d'%(numTraining,numValidation,numTest)
    
    #开始交叉验证训练
    
    ranks = [8,12]   #隐语义模型里的k值
    lambdas = [0.1,10.0]
    numIters = [10,20]
    bestModel = None   #最好的模型
    bestValidationRmse = float('inf')   #最优的RMSE值
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1
    
    for rank,lmbda,numIter in itertools.product(ranks,lambdas,numIters):   #遍历每个参数组合
        model = ALS.train(validation,rank,numIter,lmbda)   #用每组参数组合来训练模型
        validationRmse = computeRmse(model,validation,numValidation)   #用交叉验证集来寻找最小的Rmse
        print 'RMSE(validation) = %f for the model trained with' % validationRmse +\
                'rank = %d,lambda = %0.1f, and numIter = %d.'%(rank,lmbda,numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter
    
    testRmse = computeRmse(bestModel,test,numTest)   #用最优的模型计算测试集的RMSE
    print 'The best model was trained with rank = %d and lambda = %.1f,' % (bestRank,bestLambda)\
            + 'and numIter = %d,and its RMSE on the test set is %f.' % (bestNumIter,testRmse)
    
    #将训练集和交叉验证集的ratings的mean作为baseline的模型计算RMSE，并计算提升率
    meanRating = training.union(validation).map(lambda x: x[2]).mean()
    baselineRmse = sqrt(test.map(lambda x:(meanRating - x[2]) ** 2 ).reduce(add)/numTest)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print 'The best model improves the baseline by %.2f' % (improvement) +'%.'
    
    myRateMovieIds = set([x[1] for x in myRatings])   #将personalRatings里的movieid导入
    candidates = sc.parallelize([m for m in movies if m not in myRateMovieIds])   #将movies字典里除了该用户看过的电影movieid都添加并转换RDD
    candidatesmap = candidates.map(lambda x:(0,x))
    predictions = bestModel.predictAll(candidatesmap)
    predictions = model.predictAll(data.map(lambda x: (x[0],x[1])))
    recommendations = sorted(predictions,key = lambda x:x[2],reverse = True)[:50]   #根据分数逆序排序，取前五十个数据
    
    print 'Movies recommended for you:'
    for i in xrange(len(recommendations)):
        print ('%2d:%s' % (i+1,movies[recommendations[i][1]])).encode('ascii','ignore')
        
    sc.stop()
    
    
    
        
    
    
    
    
    
    
        

    
    