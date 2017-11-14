#Modelos de classificacao
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV

#Metricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix

#Outros Sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, KFold,GroupKFold
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfTransformer


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
from datetime import datetime
from class_roc import Roc
from unicodedata import normalize



class SentClassifiers():


	def read_csv(self,file):

		df1 = pd.DataFrame.from_csv('../files_external/%s'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

		df1 = df1.reset_index()

		return df1

	def write_csv(self,data,file):
		df = pd.DataFrame(data)
		df.to_csv('../files_external/'+file+'.csv', mode='a', sep=';',index=False, header=False)

	def getSTrain():
	
		tweets = db['sentiment_train'].find({},{'_id':0, 'index':0})

		return tweets

	def convert_df(self,df):
		new_df = []
		for d in df:
			if d == 'Positivo' or d =='Positive':
				new_df.append(1)
			
			elif d == 'Neutro' or d =='Neutral':
				new_df.append(0)
			
			elif d == 'Negativo' or d == 'Negative':
				new_df.append(-1)
	
		return new_df

	def clear(self,dataframe):
		new_df = []
		for df in dataframe:
			expr = re.sub(r"http\S+", "", df)
			expr = re.sub(r"[@#]\S+","",expr)
			#expr = normalize('NFKD',expr).encode('ASCII','ignore').decode('ASCII')
			filtrado = [w for w in nltk.regexp_tokenize(expr.lower(),"[\S]+") if not w in nltk.corpus.stopwords.words('portuguese')]
			frase = ""
			for f in filtrado:
				frase += f + " "
			
			new_df.append(frase)

		return new_df


	def initial(self,file):
		dataframe = self.read_csv(file)

		dataframe = dataframe.dropna()

		new_df = pd.DataFrame()

		new_df['opiniao'] = self.convert_df(dataframe['opiniao'])

		new_df['tweet'] = self.clear(dataframe['tweet'])

		new_df = new_df.reset_index()

		return new_df

	#construtor
	def __init__(self,file=None,dataframe=None):
		
		if dataframe is None:
			self.train_df = self.initial(file)
			
			self.array_train = self.train_df['tweet'].values

			self.target_train = self.train_df['opiniao'].values

			self.classifiers = []

			self.df_pred = pd.DataFrame()
		
		elif file is None:
			dataframe['tweet'] = self.clear(dataframe['tweet'])
			
			self.array_train = dataframe['tweet'].values
			
			self.target_train = dataframe['sentiment'].values
			
			self.classifiers = []
			
			self.df_pred = pd.DataFrame()
		
		else:
			print('parametro incorreto')

	def find_tweet(self):
		pos = self.read_csv('freq_pos3')['pt'].values
		neu = self.read_csv('freq_neu3')['pt'].values
		neg = self.read_csv('freq_neg3')['pt'].values


		df = pd.DataFrame()

		#self.array_train,self.target_train

		tupla = zip(neg,neu,pos)
		X = []
		y = []

		tweets = self.array_train

		for (ng,n,p) in tupla:
			for index in range(len(tweets)):
				text = self.array_train[index]
				target = self.target_train[index]
				
				if not(text.find(ng) == -1):
					X.append(text)
					y.append(target)
					#print('Text: %s, targ: %s'%(text,target))
				
				if not(text.find(n) == -1):
					X.append(text)
					y.append(target)
					#print('Text: %s, targ: %s'%(text,target))
				
				if not(text.find(p) == -1):
					X.append(text)
					y.append(target)
					#print('Text: %s, targ: %s'%(text,target))

		return X,y
		
	def validation_words(self,model,train,target):
		
		X_mod,y_mod = self.find_tweet()

		count_vect = CountVectorizer()
		X_train = count_vect.fit_transform(train)
		X_mod = count_vect.transform(X_mod)
		

		ac_v = []
		cm_v = []
		p_v = []
		r_v = []
		f1_v = []
		e_v = []
		fpr = []
		tpr = []
		roc_auc_ = []

		for i in range(5):
			model.fit(X_mod,y_mod)
			pred = model.predict(X_train)
			ac = accuracy_score(target, pred)
			p = precision_score(target, pred,average='weighted')
			r = recall_score(target, pred,average='weighted')
			f1 = (2*p*r)/(p+r)
			e = mean_squared_error(target, pred)
			cm = confusion_matrix(target,pred)
			cm_v.append(cm)
			ac_v.append(ac)
			p_v.append(p)
			r_v.append(r)
			f1_v.append(f1)
			e_v.append(e)

		ac = statistics.median(ac_v)
		p = statistics.median(p_v)
		f1 = statistics.median(f1_v)
		r = statistics.median(r_v)
		e = statistics.median(e_v)
		cm_median = self.matrix_confuse_median(cm_v)

		return ac,ac_v,p,r,f1,e,cm_median

	def cross_apply(self,model,train,target):

		count_vect = CountVectorizer()
		X = count_vect.fit_transform(train)
		kf = KFold(10, shuffle=True, random_state=1)

		ac_v = []
		cm_v = []
		p_v = []
		r_v = []
		f1_v = []
		e_v = []
		fpr = []
		tpr = []
		roc_auc_ = []
		predicts = []

		for train_index,teste_index in kf.split(X,target):
			X_train, X_test = X[train_index],X[teste_index]
			y_train, y_test = target[train_index], target[teste_index]
			model.fit(X_train,y_train)
			pred = model.predict(X_test)
			predicts += pred.tolist() 
			ac = accuracy_score(y_test, pred)
			p = precision_score(y_test, pred,average='weighted')
			r = recall_score(y_test, pred,average='weighted')
			f1 = (2*p*r)/(p+r)
			e = mean_squared_error(y_test, pred)
			cm = confusion_matrix(y_test,pred)
			cm_v.append(cm)
			ac_v.append(ac)
			p_v.append(p)
			r_v.append(r)
			f1_v.append(f1)
			e_v.append(e)


		ac = statistics.median(ac_v)
		p = statistics.median(p_v)
		f1 = statistics.median(f1_v)
		r = statistics.median(r_v)
		e = statistics.median(e_v)
		#cm_median = self.matrix_confuse_median(cm_v)
		cm_median = []

		return predicts,ac,ac_v,p,r,f1,e,cm_median

	def cross_apply_(self,model,train,target):

		count_vect = CountVectorizer()
		X = count_vect.fit_transform(train)
		#kf = KFold(10, shuffle=True, random_state=1)

		ac_v = []
		cm_v = []
		p_v = []
		r_v = []
		f1_v = []
		e_v = []
		fpr = []
		tpr = []
		roc_auc_ = []
		predicts = []
		j = 20
		for i in range(len(X)):
			index_test = [y for y in range(i,j)]
			index_train = [y for y in item if y not in index_test]
			X_train, X_test = np.array(X)[index_train],np.array(X)[index_test]
			y_train, y_test = np.array(target)[index_train], np.array(target)[index_test]
			

			model.fit(X_train,y_train)
			pred = model.predict(X_test)
			ac = accuracy_score(y_test, pred)
			p = precision_score(y_test, pred,average='weighted')
			r = recall_score(y_test, pred,average='weighted')
			f1 = (2*p*r)/(p+r)
			e = mean_squared_error(y_test, pred)
			cm = confusion_matrix(y_test,pred)
			cm_v.append(cm)
			ac_v.append(ac)
			p_v.append(p)
			r_v.append(r)
			f1_v.append(f1)
			e_v.append(e)


		ac = statistics.median(ac_v)
		p = statistics.median(p_v)
		f1 = statistics.median(f1_v)
		r = statistics.median(r_v)
		e = statistics.median(e_v)
		cm_median = self.matrix_confuse_median(cm_v)


		return predicts,ac,ac_v,p,r,f1,e,cm_median			
 
	
	def matrix_confuse_median(self,cm):
		it = (cm[0]).size
		n_class = (cm[0][0]).size

		cm_median = []

		for i in range(n_class):
			cm_median.append([])


		for i in range(it):
			median = []
			for j in range(len(cm)):
				median.append(cm[j].item(i))
			

			cm_median[int(i/3)].append(int(statistics.median(median)))


		array = np.asarray(cm_median)

		return array


		tab_aux = []
		for i in range(len(tab_pred[md[0]][0])):
			values = []
			for m in md:
				values.append(tab_pred[m][0][i])

			tab_aux.append(values)

		tab = dict()

		for m in md:
			tab[m] = []

		for tb in tab_aux:
			j = 0
			for m in md:
				tab[m].append(tb[j])
				j += 1

		return tab	

	def roc(self,cm):

		n_classes = 3
		#roc_auc = []
		fpr = [0,1]
		tpr = [0,1]

		for c in cm:
			
			re = []
			esp = []

			tp = 0
			sm = 0
			#sensibilidade
			for i in range(n_classes):
				tp = cm[i,i]
				for j in range(n_classes):
					sm += cm[i,j]

				s = tp/sm
				re.append(s)
				fpr.append(s)

			tn = 0
			smn = 0
			#Especificidade
			for i in range(n_classes):
				tn = cm[i,i]
				for j in range(n_classes):
					smn += cm[j,i]
				e = 1-(tn/smn)
				esp.append(e)	
				tpr.append(e)

		roc = Roc()

		fpr,tpr = np.array(fpr),np.array(tpr)
		roc.set_fpr(np.sort(fpr))
		roc.set_tpr(np.sort(tpr))

		roc.set_auc(auc(roc.get_fpr(),roc.get_tpr()))

		return roc

	def calc_weigth(self,acc):

		ac = []
		soma = sum(acc)

		for i in range(len(acc)):
			ac.append(acc[i]/soma)

		return ac

	def plot_roc(self,fpr,tpr,roc_auc,color,label):
		plt.figure()
		lw = 2
		plt.plot(fpr,tpr,color='red',lw=lw,label='UAC(%s = %0.2f)' % (label,roc_auc))
		plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('Taxa de Falso Positivo')
		plt.ylabel('Taxa de Verdadeiro Positivo')
		plt.title('Grafico ROC')
		plt.legend(loc="lower right")
		plt.show()

	def plot_roc_all(self,fpr,tpr,roc_auc,label):
		plt.figure()
		lw = 2
		plt.plot(fpr[0],tpr[0],color='red',lw=lw,label='UAC(%s = %0.2f)' % (label[0],roc_auc[0]))
		plt.plot(fpr[1],tpr[1],color='blue',lw=lw,label='UAC(%s = %0.2f)' % (label[1],roc_auc[1]))
		plt.plot(fpr[2],tpr[2],color='yellow',lw=lw,label='UAC(%s = %0.2f)' % (label[2],roc_auc[2]))
		plt.plot(fpr[3],tpr[3],color='green',lw=lw,label='UAC(%s = %0.2f)' % (label[3],roc_auc[3]))
		plt.plot(fpr[4],tpr[4],color='purple',lw=lw,label='UAC(%s = %0.2f)' % (label[4],roc_auc[4]))
		plt.plot(fpr[5],tpr[5],color='orange',lw=lw,label='UAC(%s = %0.2f)' % (label[5],roc_auc[5]))
		plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('Taxa de Falso Positivo')
		plt.ylabel('Taxa de Verdadeiro Positivo')
		plt.title('Grafico ROC')
		plt.legend(loc="lower right")
		plt.savefig('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Figuras/experimentos-final/roc.png')
		#plt.show()

	def plot_confuse_matrix(self,cm,title,file_name):
		labels = ['Negativo', 'Neutro','Positivo']
		cm = np.ceil(cm)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm)
		plt.title(title)
		fig.colorbar(cax)
		ax.set_xticklabels([''] + labels)
		ax.set_yticklabels([''] + labels)

		thresh = cm.max()/2
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.xlabel('Predito')
		plt.ylabel('Verdadeiro')
		plt.savefig('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Figuras/experimentos-final/%s.png'%(file_name))
		#plt.show()

	def box_plot(self,results,names,title):

		fig = plt.figure()
		fig.suptitle(title)
		ax = fig.add_subplot(111)
		plt.boxplot(results)
		ax.set_xticklabels(names)
		plt.savefig('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Figuras/boxplot-allan.png')
		#plt.show()
		

	def CMultinomialNV(self):

		parameters = {'alpha':[0.000001,0.00001,0.0001,0.001,0.1,1.0],'fit_prior':[True,False]}
		
		grid_nb = GridSearchCV(MultinomialNB(),parameters)

		#nb = MultinomialNB(alpha=0.000001)

		self.classifiers.append(grid_nb)

		#ac,ac_v,p,r,f1,e,cm = self.validation_words(grid_nb,self.array_train,self.target_train)
		pred,ac,ac_v,p,r,f1,e,cm = self.cross_apply(grid_nb,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)
		
		self.df_pred['nv'] = pred


		return ac,ac_v,p,r,f1,e,cm,roc_

	def CDecisionTree(self):

		parameters = {'criterion':('gini','entropy'),'splitter':('best','random'),'max_features':('auto','log2','sqrt')}

		grid_dt = GridSearchCV(tree.DecisionTreeClassifier(),parameters)

		
		#dt = tree.DecisionTreeClassifier(criterion='gini')

		self.classifiers.append(grid_dt)

		pred,ac,ac_v,p,r,f1,e,cm = self.cross_apply(grid_dt,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)

		self.df_pred['dt'] = pred


		return ac,ac_v,p,r,f1,e,cm,roc_

	def CSuportVectorMachine(self):

		#parameters = {'kernel':('linear', 'rbf'), 'C':[10, 100]}

		parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000],'decision_function_shape':['ovr','mutinomial']}
		
		grid_svm = GridSearchCV(svm.SVC(),parameters)

		
		#csvm = svm.SVC(kernel='linear',gamma=0.001,C=100,decision_function_shape='ovr')

		self.classifiers.append(grid_svm)

		pred,ac,ac_v,p,r,f1,e,cm = self.cross_apply(grid_svm,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)
		
		self.df_pred['svm'] = pred

		return ac,ac_v,p,r,f1,e,cm,roc_

	def CRandomForest(self):
		
		parameters = {'n_estimators':[1,5,10,20,30],'criterion':('gini','entropy')}

		grid_rf = GridSearchCV(RandomForestClassifier(),parameters)

		#rf = RandomForestClassifier(n_estimators=5,criterion='gini')

		self.classifiers.append(grid_rf)

		pred,ac,ac_v,p,r,f1,e,cm = self.cross_apply(grid_rf,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)

		self.df_pred['rf'] = pred

		return ac,ac_v,p,r,f1,e,cm,roc_	

	def CLogistRegression(self):

		parameters = {'penalty':['l2'],'C':[0.000001,0.00001,0.0001,0.001,0.1,1.0],'solver':['newton-cg','lbfgs','sag'],'multi_class':['ovr']}
		#newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
		#'penalty':('l1'),'C':[0.000001,0.00001,0.0001,0.001,0.1,1.0],'solver':['lbfgs', 'liblinear', 'sag', 'saga']
		
		grid_lr = GridSearchCV(LogisticRegression(),parameters)

		#lr = LogisticRegression(penalty='l2',multi_class='ovr')

		self.classifiers.append(grid_lr)

		pred,ac,ac_v,p,r,f1,e,cm = self.cross_apply(grid_lr,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)

		self.df_pred['lr'] = pred


		return ac,ac_v,p,r,f1,e,cm,roc_
	
	def committee(self,pesos):
		model = VotingClassifier(estimators=[('nv', self.classifiers[0]), ('svm',self.classifiers[1]), ('dt',self.classifiers[2]) ,('rf', self.classifiers[3]), ('lr',self.classifiers[4])], weights=pesos,voting='hard')


		pred,ac,ac_v,p,r,f1,e,cm_median = self.cross_apply(model,self.array_train,self.target_train)

		roc_ = Roc()

		roc_ = self.roc(cm_median)

		self.df_pred['cm'] = pred


		return ac,ac_v,p,r,f1,e,cm_median,roc_
	
	def write_dataframe(self):
    		self.write_csv(self.df_pred,'experimentos-final/predicoes')

	def pred_texts(self,dataset):

		test = self.clear(dataset)

		count_vect = CountVectorizer()
		X = count_vect.fit_transform(test)
		train = count_vect.transform(self.array_train)


		parameters = []

		parameters.append({'alpha':[0.000001,0.00001,0.0001,0.001,0.1,1.0],'fit_prior':[True,False]})
		parameters.append({'criterion':('gini','entropy'),'splitter':('best','random'),'max_features':('auto','log2','sqrt')})
		parameters.append({'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000],'decision_function_shape':['ovr','mutinomial']})	
		parameters.append({'n_estimators':[1,5,10,20,30],'criterion':('gini','entropy')})
		parameters.append({'penalty':['l2'],'C':[0.000001,0.00001,0.0001,0.001,0.1,1.0],'solver':['newton-cg','lbfgs','sag'],'multi_class':['ovr']})

		grid_nb = GridSearchCV(MultinomialNB(),parameters[0])
		grid_dt = GridSearchCV(tree.DecisionTreeClassifier(),parameters[1])
		grid_svm = GridSearchCV(svm.SVC(),parameters[2])
		grid_rf = GridSearchCV(RandomForestClassifier(),parameters[3])
		grid_lr = GridSearchCV(LogisticRegression(),parameters[4])

		pesos = []
		
		df_pesos = self.read_csv('pesos.csv')

		ac = df_pesos['ac'].values

		pesos = self.calc_weigth(ac)


		comite = VotingClassifier(estimators=[('nv', grid_nb), ('svm',grid_svm), ('dt',grid_dt) ,('rf', grid_rf), ('lr',grid_lr)], weights=pesos,voting='hard')

		comite.fit(train,self.target_train)

		pred = comite.predict(X)

		df = pd.DataFrame()

		df['tweet'] = dataset
		df['sentiment'] = pred

		return df


