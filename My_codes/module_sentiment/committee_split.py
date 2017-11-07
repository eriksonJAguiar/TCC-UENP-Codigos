#Metricas
from sklearn.metrics import accuracy_score

#Modelos de classificacao
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import nltk
import re
import csv
import json
import sys
import statistics
import math


classifiers = []


def read_csv(file):

	df1 = pd.DataFrame.from_csv('../files_extern/%s.csv'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

	df1 = df1.reset_index()

	return df1

def CMultinomialNV(train,y_train,test,y_test):

	count_vect = CountVectorizer()
	train = count_vect.fit_transform(train)
	test = count_vect.transform(test)


	parameters = {'alpha':[0.000001,0.00001,0.0001,0.001,0.1,1.0],'fit_prior':[True,False]}
		
	grid_nb = GridSearchCV(MultinomialNB(),parameters)

	ac_ = []


	for i in range(10):
		grid_nb.fit(train,y_train)
		#grid_nb.transform(test)
		pred = grid_nb.predict(test)
		ac_.append(accuracy_score(y_test,pred))

	
	ac = statistics.median(ac_)
		
	classifiers.append(grid_nb)	

	return ac,ac_

def CDecisionTree(train,y_train,test,y_test):

	count_vect = CountVectorizer()
	train = count_vect.fit_transform(train)
	test = count_vect.transform(test)
	
	parameters = {'criterion':('gini','entropy'),'splitter':('best','random')}

	grid_dt = GridSearchCV(tree.DecisionTreeClassifier(),parameters)

	ac_ = []

	for i in range(10):
		grid_dt.fit(train,y_train)
		pred = grid_dt.predict(test)
		ac_.append(accuracy_score(y_test,pred))

	
	ac = statistics.median(ac_)


	classifiers.append(grid_dt)
		
	#dt = tree.DecisionTreeClassifier(criterion)


	return ac,ac_

def CSuportVectorMachine(train,y_train,test,y_test):

	count_vect = CountVectorizer()
	train = count_vect.fit_transform(train)
	test = count_vect.transform(test)

	#parameters = {'kernel':('linear', 'rbf'), 'C':[10, 100]}

	parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000],'decision_function_shape':['ovr']}
		
	grid_svm = GridSearchCV(svm.SVC(),parameters)

	ac_ = []

	for i in range(10):
		grid_svm.fit(train,y_train)
		pred = grid_svm.predict(test)
		ac_.append(accuracy_score(y_test,pred))

	
	ac = statistics.median(ac_)

	classifiers.append(grid_svm)
		
	#csvm = svm.SVC(kernel=kernel,gamma=gamma,C=C,decision_function_shape=decision_function_shape)


	return ac,ac_

def CRandomForest(train,y_train,test,y_test):
		
	count_vect = CountVectorizer()
	train = count_vect.fit_transform(train)
	test = count_vect.transform(test)

	parameters = {'n_estimators':[1,5,10,20,30],'criterion':('gini','entropy')}

	grid_rf = GridSearchCV(RandomForestClassifier(),parameters)

	ac_ = []

	for i in range(10):
		grid_rf.fit(train,y_train)
		pred = grid_rf.predict(test)
		ac_.append(accuracy_score(y_test,pred))

	
	ac = statistics.median(ac_)	

	classifiers.append(grid_rf)

	#rf = RandomForestClassifier()

	
	return ac,ac_	

def CLogistRegression(train,y_train,test,y_test):

	count_vect = CountVectorizer()
	train = count_vect.fit_transform(train)
	test = count_vect.transform(test)

	parameters = {'penalty':['l2'],'C':[0.000001,0.00001,0.0001,0.001,0.1,1.0],'solver':['newton-cg','lbfgs','sag'],'multi_class':['ovr','multinomial']}
	#newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
	#'penalty':('l1'),'C':[0.000001,0.00001,0.0001,0.001,0.1,1.0],'solver':['lbfgs', 'liblinear', 'sag', 'saga']
		
	grid_lr = GridSearchCV(LogisticRegression(),parameters)

	ac_ = []

	for i in range(10):
		grid_lr.fit(train,y_train)
		pred = grid_lr.predict(test)
		ac_.append(accuracy_score(y_test,pred))

	
	ac = statistics.median(ac_)

	classifiers.append(grid_lr)

	#lr = LogisticRegression(penalty=penalty,multi_class=multi_class)

	return ac,ac_
	
def committee(pesos,train,y_train,test,y_test):

	
	count_vect = CountVectorizer()
	train = count_vect.fit_transform(train)
	test = count_vect.transform(test)

	model = VotingClassifier(estimators=[('nv', classifiers[0]), ('svm',classifiers[1]), ('dt',classifiers[2]) ,('rf', classifiers[3]), ('lr',classifiers[4])], weights=pesos,voting='hard')

	ac_ = []

	for i in range(10):
		model.fit(train,y_train)
		pred = model.predict(test)
		ac_.append(accuracy_score(y_test,pred))

	
	ac = statistics.median(ac_)


	return ac,ac_

def convert_df(df):
	new_df = []
	for d in df:
		if d == 'Positivo':
			new_df.append(1)
			
		elif d == 'Neutro':
			new_df.append(0)
			
		elif d == 'Negativo':
			new_df.append(-1)

	return new_df

def box_plot(results,names,title):

	fig = plt.figure()
	fig.suptitle(title)
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()

def calc_weigth(acc):

	ac = []
	soma = sum(acc)

	for i in range(len(acc)):
		ac.append(acc[i]/soma)

	return ac


if __name__ == '__main__':

	df_train = read_csv('twitters_treino')
	df_test = read_csv('twitters_teste')

	df_train['Classe'] = convert_df(df_train['Classe'])
	df_test['Classe'] = convert_df(df_test['Classe'])

	acc = []
	results = []

	nv_ac,r = CMultinomialNV(df_train['Texto'],df_train['Classe'],df_test['Texto'],df_test['Classe'])
	print('Nv = %f'%nv_ac)
	acc.append(nv_ac)
	results.append(r)

	dt_ac,r = CDecisionTree(df_train['Texto'],df_train['Classe'],df_test['Texto'],df_test['Classe'])
	print('Dt = %f'%dt_ac)
	acc.append(dt_ac)
	results.append(r)

	svm_ac,r = CSuportVectorMachine(df_train['Texto'],df_train['Classe'],df_test['Texto'],df_test['Classe'])
	print('SVM = %f'%svm_ac)
	acc.append(svm_ac)
	results.append(r)

	rf_ac,r = CRandomForest(df_train['Texto'],df_train['Classe'],df_test['Texto'],df_test['Classe'])
	print('RF = %f'%rf_ac)
	acc.append(rf_ac)
	results.append(r)


	rl_ac,r = CLogistRegression(df_train['Texto'],df_train['Classe'],df_test['Texto'],df_test['Classe'])
	print('RL = %f'%rl_ac)
	acc.append(rl_ac)
	results.append(r)

	pesos = calc_weigth(acc)

	cm_ac,r = committee(pesos,df_train['Texto'],df_train['Classe'],df_test['Texto'],df_test['Classe'])
	print('Committee = %f'%cm_ac)
	acc.append(cm_ac)
	results.append(r)

	box_plot(results,['nv','dt','svm','rf','rl','cm'],'Comparação dos algoritmos')





