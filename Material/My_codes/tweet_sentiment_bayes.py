##Modelos de classificação
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

##Metricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve,auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix

##Outros Sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, KFold,GroupKFold
from sklearn.preprocessing import label_binarize


import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import nltk
import re
import csv
import json
import sys
from pymongo import MongoClient
from datetime import datetime

client = MongoClient()
db = client.baseTweetsTCC

params = sys.argv[1:]

def populaBase(df):
	records = json.loads(df.T.to_json()).values()
	db.sentiment_train.insert(records)

def getSTrain():
	
	tweets = db['sentiment_train'].find({},{'_id':0, 'index':0})

	return tweets

def getSTest():
	
	tweets = db['tweetsProcessing1'].find({}).limit(100)

	return tweets

def read_csv():

	df1 = pd.DataFrame.from_csv('files_extern/tweets-1.csv',sep=';',index_col=0,encoding ='ISO-8859-1')

	df1 = df1.reset_index()

	df2 = pd.DataFrame.from_csv('files_extern/tweets-2.csv',sep=';',index_col=0,encoding ='ISO-8859-1')

	df2 = df2.reset_index()

	df_full = pd.concat([df1,df2])

	return df_full

def write_csv(data):
	df = pd.DataFrame(data)
	df.to_csv('files_extern/logs.csv', mode='a', sep=';',index=False, header=False)

def convert_df(df):
	new_df = []
	for d in df['opiniao']:
		if d == 'Positivo':
			new_df.append(1)
		
		elif d == 'Neutro':
			new_df.append(0)
		
		elif d == 'Negativo':
			new_df.append(-1)

	return new_df

def clean(dataframe):
	new_df = []
	for df in dataframe['tweet']:
		expr = re.sub(r"http\S+", "", df)
		expr = re.sub(r"@\S+","",expr)
		
		filtrado = [w for w in nltk.regexp_tokenize(expr.lower(),"[\S'#]+") if not w in nltk.corpus.stopwords.words('portuguese')]
		frase = ""
		for f in filtrado:
			frase += f + " "
		new_df.append(frase)

	return new_df

def init():
	dataframe = read_csv()

	dataframe['opiniao'] = convert_df(dataframe)

	dataframe['tweet'] = clean(dataframe)

	dataframe = dataframe.reset_index()

	populaBase(dataframe)

def split_base(base,target):
	test = []
	train = []
	target_train = []
	target_test = []
	percent = int(len(base)*0.1)
	
	it = int(len(base)/percent)
	
	for i in range(it):
		x_train, x_test, x_target_train,x_target_test = train_test_split(base,target,test_size=0.1)
		train.append(x_train)
		target_train.append(x_target_train)
		test.append(x_test)
		target_test.append(x_target_test)

	return train, target_train, test,target_test

def cross_apply_(model,train,target_train,test,target_test):
	count_vect = CountVectorizer()
	accuracy = []
	precision = []
	recall = []
	f1_score = []
	error = []
	it = len(train)
	for i in range(it):
		X_train = count_vect.fit_transform(train[i])
		X_test = count_vect.transform(test[i])
		model.fit(X_train,target_train[i])
		pred = model.predict(X_test)
		ac = accuracy_score(target_test[i], pred)
		p = precision_score(target_test[i], pred,average='weighted')
		r = recall_score(target_test[i], pred,average='weighted')
		f1 = (2*p*r)/(p+r)
		e = mean_squared_error(target_test[i], pred)
		accuracy.append(ac)
		precision.append(p)
		recall.append(r)
		f1_score.append(f1)
		error.append(e)

	ac_mean = sum(accuracy)/len(accuracy)
	p_mean = sum(precision)/len(precision)
	r_mean = sum(recall)/len(recall)
	f1_mean = sum(f1_score)/len(f1_score)
	e_mean = sum(error)/len(error)

	return ac_mean, p_mean, r_mean, f1_mean,e_mean

def cross_apply(model,train,target):

	count_vect = CountVectorizer()
	X = count_vect.fit_transform(train)
	kf = KFold(10, shuffle=True, random_state=1)
	#pred = cross_val_predict(model, X_train, target,cv=k_fold)
	ac_v = []
	cm_v = []
	p_v = []
	r_v = []
	f1_v = []
	e_v = []

	for train_index,teste_index in kf.split(X):
		X_train, X_test = X[train_index],X[teste_index]
		y_train, y_test = target[train_index], target[teste_index]
		model.fit(X_train,y_train)
		pred = model.predict(X_test)
		cm = confusion_matrix(y_test,pred)
		ac = accuracy_score(y_test, pred)
		p = precision_score(y_test, pred,average='weighted')
		r = recall_score(y_test, pred,average='weighted')
		f1 = (2*p*r)/(p+r)
		e = mean_squared_error(y_test, pred)
		cm_v.append(cm)
		ac_v.append(ac)
		p_v.append(p)
		r_v.append(r)
		f1_v.append(f1)
		e_v.append(e)

	ac = sum(ac_v)/len(ac_v)
	p = sum(p_v)/len(p_v)
	f1 = sum(f1_v)/len(f1_v)
	r = sum(r_v)/len(r_v)
	e = sum(e_v)/len(e_v)
	cm = sum(cm_v)/len(cm_v)
	cm = cm.astype('int')


	return ac,p,r,f1,e,cm

def roc(model,train,target):

	count_vect = CountVectorizer()
	X = count_vect.fit_transform(train)

	target_ = label_binarize(target, classes=[-1,0,1])
	n_classes = target_.shape[1]
	classifier = OneVsRestClassifier(model)
	kf = KFold(n_splits=10, random_state=1, shuffle=True)
	
	fpr_v = []
	tpr_v = []
	roc_v = []

	for train_index,teste_index in kf.split(X):
		X_train, X_test = X[train_index], X[teste_index]
		y_train, y_test = target_[train_index], target_[teste_index]
		y_score = classifier.fit(X_train,y_train).predict(X_test)
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(n_classes):
			fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
			roc_auc[i] = auc(fpr[i], tpr[i])

		fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
		roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


	return fpr[2],tpr[2],roc_auc[2]

#def roc_(model, df,target):
	count_vect = CountVectorizer()
	rd = []
	groups = 10
	for i in range(groups):
		
		rd.append(msk)
		transf = count_vect.fit_transform(train)

		target_ = label_binarize(target, classes=[-1,0,1])
		n_classes = target_.shape[1]

		X_train = count_vect.fit_transform(train[i])
		X_test = count_vect.transform(test[i])
		model.fit(X_train,target_train[i])
		pred = model.predict(X_test)
		
#def split_train_test_cross(df,k):
	group_test = []
	group_train = []
	group_t_test = []
	group_t_train = []
	
	for i in range(k):
		msk = np.random.rand(len(df)) < 0.9
		train, test = df[msk].copy(deep = True), df[~msk].copy(deep = True)
		target_train, target_test = target[msk], target[~msk]
	


def plot_roc(fpr,tpr,roc_auc):
	plt.figure()
	lw = 2
	plt.plot(fpr[0],tpr[0],color='red',lw=lw,label='UAC(nv = %0.2f)' % roc_auc[0])
	plt.plot(fpr[1],tpr[1],color='yellow',lw=lw,label='UAC(dt = %0.2f)' % roc_auc[1])
	plt.plot(fpr[2],tpr[2],color='blue',lw=lw,label='UAC(sgc = %0.2f)' % roc_auc[2])
	plt.plot(fpr[3],tpr[3],color='green',lw=lw,label='UAC(rf = %0.2f)' % roc_auc[3])
	plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Taxa de Falso Positivo')
	plt.ylabel('Taxa de Verdadeiro Positivo')
	plt.title('Grafico ROC')
	plt.legend(loc="lower right")
	plt.show()

def plot_confuse_matrix(cm):
	labels = ['Negativo', 'Neutro','Positivo']
	print(cm)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm)
	plt.title('Matriz de Confusao do Classificador')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + labels)
	ax.set_yticklabels([''] + labels)

	thresh = cm.max()/2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.xlabel('Predito')
	plt.ylabel('Verdadeiro')
	plt.show()

if __name__ == '__main__':

	if len(params) > 0:
		if params[0] == '1':
			init()
			print("Algoritmo inicializado com sucesso !!")
			exit(0)

	train_coll = getSTrain()

	train_df = pd.DataFrame(list(train_coll))

	array_train = train_df['tweet'].values

	target_train = train_df['opiniao'].values

	nb = MultinomialNB(alpha = 0.000000000001)

	dt = tree.DecisionTreeClassifier(criterion='gini')

	sgdc = SGDClassifier(loss="log", penalty="l2")

	rf = RandomForestClassifier()

	v_fpr = []
	v_tpr = []
	v_roc_auc = []
	df_metrics = pd.DataFrame(columns=['algoritmo', 'acuracia', 'precisao', 'recall', 'f1','erro','auc','data'])

	#colunas do vetor 'algoritmo', 'acuracia', 'precisao', 'recall', 'f1','erro','auc','data'
	l = []
	datas = []

	print("Mensurando Naive Bayes Multinominal...")
	ac,p,r,f1,e,cm = cross_apply(nb,array_train,target_train)
	print("Acuracia = %f"%(ac))
	print("Precisão = %f"%(p))
	print("Recall = %f"%(r))
	print("F1 Score = %f"%(f1))
	print("Erro = %f"%(e))
	fpr,tpr,roc_auc = roc(nb,array_train,target_train)
	v_fpr.append(fpr)
	v_tpr.append(tpr)
	v_roc_auc.append(roc_auc)
	#plot_confuse_matrix(cm)
	l = 'nv',ac,p,r,f1,e,roc_auc,str(datetime.now())
	datas.append(l)
	print("Calculo Naive Bayes Multinomina realizado com sucesso !")
	print('')
	
	print("Mensurando Arvore de Decisao...")
	ac,p,r,f1,e,cm = cross_apply(dt,array_train,target_train)
	print("Acuracia = %f"%(ac))
	print("Precisão = %f"%(p))
	print("Recall = %f"%(r))
	print("F1 Score = %f"%(f1))
	print("Erro = %f"%(e))
	fpr,tpr,roc_auc = roc(dt,array_train,target_train)
	v_fpr.append(fpr)
	v_tpr.append(tpr)
	v_roc_auc.append(roc_auc)
	l = 'dt',ac,p,r,f1,e,roc_auc,str(datetime.now())
	datas.append(l)
	print("Calculo Arvore de Decisao realizado com sucesso !")
	print('')

	print("Mensurando Gradiente Estocastico...")
	ac,p,r,f1,e,cm = cross_apply(sgdc,array_train,target_train)
	print("Acuracia = %f"%(ac))
	print("Precisão = %f"%(p))
	print("Recall = %f"%(r))
	print("F1 Score = %f"%(f1))
	print("Erro = %f"%(e))
	fpr,tpr,roc_auc = roc(sgdc,array_train,target_train)
	v_fpr.append(fpr)
	v_tpr.append(tpr)
	v_roc_auc.append(roc_auc)
	l = 'ge',ac,p,r,f1,e,roc_auc,str(datetime.now())
	datas.append(l)
	print("Calculo Gradiente Estocastico realizado com sucesso !")
	print('')


	print("Mensurando Random Forest...")
	ac,p,r,f1,e,cm = cross_apply(rf,array_train,target_train)
	print("Acuracia = %f"%(ac))
	print("Precisão = %f"%(p))
	print("Recall = %f"%(r))
	print("F1 Score = %f"%(f1))
	print("Erro = %f"%(e))
	fpr,tpr,roc_auc = roc(rf,array_train,target_train)
	v_fpr.append(fpr)
	v_tpr.append(tpr)
	v_roc_auc.append(roc_auc)
	l = 'rf',ac,p,r,f1,e,roc_auc,str(datetime.now())
	datas.append(l)
	print("Calculo Random Forest realizado com sucesso !")
	print('')

	write_csv(datas)
	plot_roc(v_fpr,v_tpr,v_roc_auc)

	#count_vect = CountVectorizer()

	#X_train = count_vect.fit_transform(array_train)

	#nb.fit(X_train,target_train)

	#t = ['estou muito feliz']

	#X_test = count_vect.transform(t)

	#pred = nb.predict(X_test)

	#print(X_train[0])

	#print(pred)

	print("Algoritmo processado com sucesso !!")
	