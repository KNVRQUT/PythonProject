

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *

#DATA IS OLYMPIC DATA WITH 'Y' BEING MEDAL AND X BEING THE FOLLOWING COLUMNS: SEX, AGE, HEIGHT, WEIGHT

data1 = pd.read_csv('/users/knvrqut/desktop/OLYMPICS.CSV', encoding = 'latin-1')
data1

data1.info()
data1.count()
data1.describe()
data1.head(50)
data1.tail(50)
data1.ndim
data1.shape
data1.mean()
data1.median()
data1.mode()
data1.min()
data1.max()
data1.dtypes
data1.__len__()
#UNIQUE BY COLUMN
pd.unique(data1.iloc[:,1])
#COLUMNS
data1.keys()



data2 = data1.iloc[:,[2,3,4,5,14]]
data2

data3 = data2.dropna()
data3
data3.info()

#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data4 = le.fit_transform(data3.iloc[:,0])
data4

data5 = le.fit_transform(data3.iloc[:,4])
data5

#CHANGE TO A DATAFRAME
data4df = pd.DataFrame(data4, columns= ['Sex'])
data4df

data5df = pd.DataFrame(data5, columns= ['Medal'])
data5df

#CONCAT DATA
data6 = pd.concat([data3.iloc[:,[1,2,3]], data4df, data5df], axis=1)
data6

#DID ANOTHER DROPNA HERE
data7 = data6.dropna()
data7


#PLOTTING COMES NEXT
plt.plot(data7.iloc[:,1], linestyle = '--', color = 'blue', linewidth=1, alpha =0.8, label = 'Age')
plt.title('Olympic Data')
plt.xlabel()
plt.ylabel()
plt.legend(loc = 'upper right')
plt.xlim()
plt.ylim()
plt.tight_layout()
plt.grid('off')
plt.axis('off')
plt.show()


plt.scatter(data7.iloc[:,0], data7.iloc[:,1], s=50, color = 'orange',alpha=0.8, marker='+',  label = 'Olympic Data')
plt.xlabel()
plt.ylabel()
plt.xlim()
plt.ylim()
plt.legend(loc = 'lower left')
plt.grid('off')
plt.axis('on')


plt.hist(data7.iloc[:,0], color='b', alpha=0.8, label='Olympic Data', orientation='horizontal')
plt.xlabel()
plt.ylabel()
plt.xlim()
plt.ylim()
plt.tight_layout()
plt.legend(loc='upper right')
plt.grid('off')
plt.axis('on')


plt.bar(left=data7.iloc[:,0], height=data7.iloc[:,1], width=0.7, alpha=0.8, label='Olympic Data', color = 'black', align='edge')
plt.title()
plt.xlabel()
plt.ylabel()
plt.xlim()
plt.ylim()
plt.legend(loc='upper right')
plt.grid('off')
plt.axis('on')
plt.show()


plt.boxplot(data7.iloc[:,3], whis=.5, showfliers=True, sym='+', vert=False)
plt.title()
plt.xlabel()
plt.ylabel()
plt.xlim()
plt.ylim()
plt.grid('off')
plt.axis('on')
plt.tight_layout()
plt.show()


ggplot(aes(x='Age', y='Medal'), data=data7) + geom_line(color = 'r', alpha=0.8) + ggtitle('Olympic Data') +\
    xlab('Age') + ylab('Medal') + xlim() + ylim() + facet_wrap('Sex')

ggplot(aes(x='Age', y='Medal'), data=data7) + geom_point(color = 'r', alpha=0.8) + ggtitle('Olympic Data') +\
    xlab('Age') + ylab('Medal') + xlim() + ylim() + facet_wrap('Sex')

ggplot(aes(x='Age', y='Medal'), data=data7) + geom_histogram(color = 'r', alpha=0.8) + ggtitle('Olympic Data') +\
    xlab('Age') + ylab('Medal') + xlim() + ylim() + facet_wrap('Sex')

ggplot(aes(x='Age', y='Medal'), data=data7) + geom_bar(color = 'r', alpha=0.8) + ggtitle('Olympic Data') +\
    xlab('Age') + ylab('Medal') + xlim() + ylim() + facet_wrap('Sex')

ggplot(aes(x='Age', y='Medal'), data=data7) + geom_boxplot(whis=0.5, str='+', notch=True) + ggtitle('Olympic Data') +\
    xlab('Age') + ylab('Medal') + xlim() + ylim()



#X,Y
X = data7.iloc[:,[0,1,2,3]]
X
X.info()

y = data7.iloc[:,4]
y

#TRAIN, TEST, SPLIT
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.5)



#STD, NORM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

std = StandardScaler()
nrm = MinMaxScaler()

X_train_std = std.fit_transform(X_train)
X_train_std
X_test_std = std.fit(X_test)
X_test_std

X_train_nrm = nrm.fit_transform(X_train)
X_train_nrm
X_test_nrm = nrm.fit(X_test)



#PCA, LDA
from sklearn.lda import LDA

lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_train_lda
X_test_lda = lda.fit_transform(X_test, y_test)



#FIT AND PREDICT ALGORITHMSs
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

lr = LinearRegression()
logr = LogisticRegression(random_state=0, n_jobs=4, penalty='l2', max_iter=100)
perp = Perceptron(n_iter=10, eta0=0.1, n_jobs=4, random_state=0)
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
nb = GaussianNB()

lr.fit(X_train, y_train)
logr.fit(X_train, y_train)
perp.fit(X_train, y_train)
dtc.fit(X_train, y_train)
nb.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
logr_pred = logr.predict(X_test)
perp_pred = perp.predict(X_test)
dtc_pred = dtc.predict(X_test)
nb_pred = nb.predict(X_test)

lr.score(X_test, y_test)
logr.score(X_test, y_test)
perp.score(X_test, y_test)
dtc.score(X_test, y_test)
nb.score(X_test, y_test)


#SCORE
from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test, dtc_pred)
ac

print('Acc. Score: %.5f' % ac)

#CROSS VALIDATION SCORE

from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline

ppl = Pipeline([('dtc', DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0))])

cvs = cross_val_score(ppl, X_train, y_train, cv=10, n_jobs=4)
cvs
print('Cross Validation Score: Mean %.3f, STD %.3f' % (np.mean(cvs), np.std(cvs)))



#LEARNING CURVE
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline

ppl = Pipeline([( 'dtc', DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0))])
train_sizes, train_scores, test_scores = learning_curve(ppl, X, y, cv=10, n_jobs=4)

train_mean = np.mean(train_scores)
train_std = np.std(train_scores)
test_mean = np.mean(test_scores)
test_std = np.mean(test_scores)

plt.plot(train_sizes, train_scores, color = 'g', marker = 'o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, color = 'g', alpha=0.7)

plt.plot(train_sizes, test_scores, color = 'b', linestyle= '-', marker='s', markersize=5, label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, color = 'b', alpha=0.7)


plt.tight_layout()
plt.grid()
plt.title()
plt.legend(loc='upper right')



#GCV HYPERPARAMETERS
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

ppl = Pipeline([('dtc', DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0))])


'''Put the hyperparameters of the alg. with the highest score in the grid
'''
grid = {'dtc__criterion' : ['entropy', 'gini'],
        'dtc__max_depth' : [5,10,15],
        'dtc__random_state' :[0,5,10]}

gcv = GridSearchCV(ppl, grid, cv=10, n_jobs=4)

gcv.fit(X_train, y_train)

'''List of best hyperparameters listed for the alg. chose is best_estimator
'''
gcv.best_estimator_
gcv.best_score_

'''Best params as listed in your grid
'''
gcv.best_params_


'''Gives you with clf the actual percentage if you ran it with the best hyperparameters ... really great.
'''
clf = gcv.best_estimator_
clf.fit(X_train, y_train)
clf.score(X_test, y_test)



#RANDOMFOREST FOR FEATURES
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

rfc = RandomForestClassifier(n_estimators=1000, max_depth=5, criterion='entropy', random_state=0, n_jobs=4)

rfc.fit(X_train, y_train)


'''With this value you already know the ratings based on the X values ... very simple ... the rest is just graphing it
'''
importances = rfc.feature_importances_
importances

indices = np.argsort(importances)[::1]

labels = X.columns[0:]


plt.bar(left =range(X_train.shape[1]) , height=importances[indices], color = 'blue', align='center')

plt.xticks(range(X_train.shape[1]), labels[indices], rotation = 90)
plt.tight_layout()


#DTC
from graphviz import Source
from sklearn import tree

import numpy as np

graph = Source(tree.export_graphviz(dtc, out_file=None, feature_names=X.columns))

graph.format = 'png'


graph.render('dtree render', view=True)