#Modelos de classificacao
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

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
from datetime import datetime
from class_roc import Roc

class SentClassifiers():


	def read_csv(self,folder):

		df1 = pd.DataFrame.from_csv('%s/tweets-1.csv'%(folder),sep=';',index_col=0,encoding ='ISO-8859-1')

		df1 = df1.reset_index()

		df2 = pd.DataFrame.from_csv('%s/tweets-2.csv'%(folder),sep=';',index_col=0,encoding ='ISO-8859-1')

		df2 = df2.reset_index()

		df_full = pd.concat([df1,df2])

		return df_full

	def write_csv(self,data,file):
		df = pd.DataFrame(data)
		df.to_csv('files_extern/'+file+'.csv', mode='a', sep=';',index=False, header=False)

	def convert_df(self,df):
		new_df = []
		for d in df['opiniao']:
			if d == 'Positivo':
				new_df.append(1)
			
			elif d == 'Neutro':
				new_df.append(0)
			
			elif d == 'Negativo':
				new_df.append(-1)

		return new_df

	def clean(self,dataframe):
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

	def initial(self):
		dataframe = self.read_csv('../files_extern')

		dataframe['opiniao'] = self.convert_df(dataframe)

		dataframe['tweet'] = self.clean(dataframe)

		dataframe = dataframe.reset_index()

		#populaBase(dataframe)

		return dataframe

	#construtor
	def __init__(self):
		
		self.train_df = self.initial()
		
		self.array_train = self.train_df['tweet'].values

		self.target_train = self.train_df['opiniao'].values


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


		for train_index,teste_index in kf.split(X,target):
			X_train, X_test = X[train_index],X[teste_index]
			y_train, y_test = target[train_index], target[teste_index]
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

		ac = sum(ac_v)/len(ac_v)
		p = sum(p_v)/len(p_v)
		f1 = sum(f1_v)/len(f1_v)
		r = sum(r_v)/len(r_v)
		e = sum(e_v)/len(e_v)
		cm_mean = sum(cm_v)/len(cm_v)
		cm_mean.astype(int)

		return ac,p,r,f1,e,cm_mean

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

			#roc_auc.append(auc(re,esp))
			#fpr.append(re)
			#tpr.append(esp)

		fpr,tpr = np.array(fpr),np.array(tpr)
		fpr = np.sort(fpr)
		tpr = np.sort(tpr)

		roc_auc = auc(fpr,tpr)

		#print(fpr)
		#print(tpr)
		#print('roc = %f'%roc_auc)


		#write_csv(fpr,'roc_2')
		#write_csv(tpr,'roc_2')


		return fpr,tpr,roc_auc

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

	def plot_confuse_matrix(self,cm):
		labels = ['Negativo', 'Neutro','Positivo']
		cm = np.ceil(cm)
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

	def CMultinomialNV(self,alpha = 0.000001):

		nb = MultinomialNB(alpha)

		ac,p,r,f1,e,cm = self.cross_apply(nb,self.array_train,self.target_train)
		roc_  = Roc()
		fpr,tpr,auc = self.roc(cm)
		roc_.set_fpr(fpr)
		roc_.set_tpr(tpr)
		roc_.set_auc(auc)
		log = 'nv',ac,p,r,f1,e,str(datetime.now())

		return ac,p,r,f1,e,cm,roc_

	def CDecisionTree(self,criterion='gini'):

		dt = tree.DecisionTreeClassifier(criterion)

		ac,p,r,f1,e,cm = self.cross_apply(dt,self.array_train,self.target_train)
		roc_  = Roc()
		fpr,tpr,auc = self.roc(cm)
		roc_.set_fpr(fpr)
		roc_.set_tpr(tpr)
		roc_.set_auc(auc)
		log = 'dt',ac,p,r,f1,e,str(datetime.now())

		return ac,p,r,f1,e,cm,roc_

	def CSuportVectorMachine(self,gamma=0.001,C=100,decision_function_shape='ovr'):

		csvm = svm.SVC(gamma,C,decision_function_shape)

		ac,p,r,f1,e,cm = self.cross_apply(csvm,self.array_train,self.target_train)
		roc_  = Roc()
		fpr,tpr,auc = self.roc(cm)
		roc_.set_fpr(fpr)
		roc_.set_tpr(tpr)
		roc_.set_auc(auc)
		log = 'svm',ac,p,r,f1,e,str(datetime.now())


		return ac,p,r,f1,e,cm,roc_

	def CGradientDescEst(self,loss="log", penalty="l2"):

		sgdc = SGDClassifier(loss="log", penalty="l2")

		ac,p,r,f1,e,cm = self.cross_apply(sgdc,self.array_train,self.target_train)
		roc_  = Roc()
		fpr,tpr,auc = self.roc(cm)
		roc_.set_fpr(fpr)
		roc_.set_tpr(tpr)
		roc_.set_auc(auc)
		log = 'ge',ac,p,r,f1,e,str(datetime.now())
		self.write_csv(log,'log_2')
		
		return ac,p,r,f1,e,cm,roc_

	def CRandomForest(self):
		
		rf = RandomForestClassifier()

		ac,p,r,f1,e,cm = self.cross_apply(rf,self.array_train,self.target_train)
		roc_  = Roc()
		fpr,tpr,auc = self.roc(cm)
		roc_.set_fpr(fpr)
		roc_.set_tpr(tpr)
		roc_.set_auc(auc)
		log = 'rf',ac,p,r,f1,e,str(datetime.now())

		return ac,p,r,f1,e,cm,roc_
	

	def CLogistRegression(self,penalty="l2"):

		lr = LogisticRegression(penalty)
		ac,p,r,f1,e,cm = self.cross_apply(lr,self.array_train,self.target_train)
		roc_  = Roc()
		fpr,tpr,auc = self.roc(cm)
		roc_.set_fpr(fpr)
		roc_.set_tpr(tpr)
		roc_.set_auc(auc)
		log = 'rl',ac,p,r,f1,e,str(datetime.now())

		return ac,p,r,f1,e,cm,roc_



	
