import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt

def loadTrainSet(filepath):
    raw = np.loadtxt(filepath, delimiter=',', dtype=np.str, skiprows=1)
    X, y = raw[:,1:], raw[:,0]
    trainSet = np.hstack((X, y.reshape(-1,1)))
    return trainSet
    

def out(trainset):
    trainset = trainset
    X=trainset[:,:(trainset.shape[1]-1)]
    y=trainset[:,(trainset.shape[1]-1)]
    X=np.asarray(X)
    y=np.asarray(y.T)[0]
    return X,y


def gridsearchcv(X,y):
    accuracy=[]
    stdlist=[]
    classifier = RandomForestClassifier(verbose=2, n_jobs=1,oob_score=1)
    param_grid={'n_estimators':np.arange(1, 100, 10)}
#    param_grid={'n_estimators':np.arange(1, 202, 10)}
#    param_grid={'n_estimators':[200], 'criterion':['gini', 'entropy']}
#    param_grid={'n_estimators':[200], 'max_features':np.append(np.arange(28-20, 28, 1), np.arange(28, 28+20, 1))}
#    param_grid={'n_estimators':[200], 'max_depth':np.arange(40, 40+20, 1)}
#    param_grid={'n_estimators':[200], 'min_samples_split':np.arange(2, 2+10, 1)}
#    param_grid={'n_estimators':[200], 'min_samples_leaf':np.arange(1, 1+10, 1)}
#    param_grid={'n_estimators':[200], 'max_leaf_nodes':np.arange(3000, 3000+1000, 100)}

    grid = GridSearchCV(classifier , param_grid=param_grid)
    grid.fit(X,y)
    fig=plt.figure(1, figsize=(16, 12))
    plt.clf()
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)
    scores=grid.grid_scores_
    for i in range(len(scores)):
        accu=scores[i][1]
        stdnum=np.std(scores[i][2])
        accuracy.append(accu)
        stdlist.append(stdnum) 
    ax1.plot(np.arange(1, 100, 10),accuracy, linewidth=2)
    ax2.plot(np.arange(1, 100, 10),stdlist, linewidth=2)
    plt.axis('tight')
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('accuracy')
    ax2.set_xlabel('n_estimators')
    ax2.set_ylabel('std_accuracy')
    