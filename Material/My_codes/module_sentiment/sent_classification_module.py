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
import statistics
import math
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
		df.to_csv('../files_extern/'+file+'.csv', mode='a', sep=';',index=False, header=False)

	def getSTrain():
	
		tweets = db['sentiment_train'].find({},{'_id':0, 'index':0})

		return tweets

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

		return dataframe

	#construtor
	def __init__(self):
		
		self.train_df = self.initial()
		
		self.array_train = self.train_df['tweet'].values

		self.target_train = self.train_df['opiniao'].values

		self.acc_vet = []


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

		ac = statistics.median(ac_v)
		p = statistics.median(p_v)
		f1 = statistics.median(f1_v)
		r = statistics.median(r_v)
		e = statistics.median(e_v)
		cm_median = self.matrix_confuse_median(cm_v)

		return ac,ac_v,p,r,f1,e,cm_median

	def matrix_confuse_median(self,cm):
		pos = dict()

		for i in range(1,10):
			pos[i] = []

		for cf in cm:
			x = 1
			for i in range(0,3):
				for j in range(0,3):
					pos[x].append(cf[i][j])
					x += 1

		part1 = []
		part2 = []
		part3 = []

		for i in range(1,10):
			if i <= 3 :
				part1.append(math.ceil(statistics.median(pos[i])))	

			elif i > 3 and i <= 6:
				part2.append(math.ceil(statistics.median(pos[i])))

			else:
				part3.append(math.ceil(statistics.median(pos[i])))


		m_confuse = np.array([part1,part2,part3])

		return m_confuse

	def more_voted(self,votes):
		rank = 0
		if votes[0] > votes[1] and votes[0] > votes[2]:
			rank = -1

		elif votes[1] > votes[0] and votes[1] > votes[2]:
			rank = 0

		elif votes[2] > votes[0] and votes[2] > votes[1]:
			rank = 1

		return rank

	def votation(self,votes_i,weight):
		votes = []
		best = []
		for i in range(-1,2):
			v = votes_i.count(i)
			for j in range(len(votes_i)):
				if votes_i[j] == i:
					v *= weight[j]

			votes.append(v)

		return votes

	def percorre_pred(self,tab_pred,md):
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

	def cross_apply_best(self,k,models,train,target):

		count_vect = CountVectorizer()
		X = count_vect.fit_transform(train)
		kf = KFold(k, shuffle=True, random_state=1)
		k_esimo = 1

		m = ['nv','svm','dt','rf','gd','rl']

		pred_more_voted = dict()
		original_label = dict()

		for train_index,teste_index in kf.split(X,target):
			tab_pred = dict()
			tab_pred_aux = dict()
		
			tab_pred['nv'] = []
			tab_pred['svm'] = []
			tab_pred['dt'] = []
			tab_pred['rf'] = []
			tab_pred['gd'] = []
			tab_pred['rl'] = []

			tab_pred_aux = tab_pred

			for i in range(len(models['model'])):
				X_train, X_test = X[train_index],X[teste_index]
				y_train, y_test = target[train_index], target[teste_index]
				models['model'][i].fit(X_train,y_train)
				pred = models['model'][i].predict(X_test)
				tab_pred[m[i]].append(pred)

			tab_pred_aux = self.percorre_pred(tab_pred,m)
			more_voted = []

			for j in range(len(tab_pred_aux[m[0]])):
				vts = []
				for md in m:
					vts.append(tab_pred_aux[md][j]) 
					
				votes = self.votation(vts,models['peso'])

				more_voted.append(self.more_voted(votes))


			pred_more_voted[str(k_esimo)] = more_voted
			original_label[str(k_esimo)] = y_test.tolist()
			k_esimo += 1

		return pred_more_voted, original_label

	def committee(self,k,pesos):
		models = dict()
		models['model'] = []
		models['peso'] = pesos
		nv = MultinomialNB(alpha=0.000001)
		models['model'].append(nv)
		dt = tree.DecisionTreeClassifier(criterion='gini')
		models['model'].append(dt)
		csvm = svm.SVC(gamma=0.001,C=100,decision_function_shape='ovr')
		models['model'].append(csvm)
		sgdc = SGDClassifier(penalty="l2")
		models['model'].append(sgdc)
		rf = RandomForestClassifier()
		models['model'].append(rf)
		lr = LogisticRegression(penalty='l2')
		models['model'].append(lr)

		pred,original = self.cross_apply_best(k,models,self.array_train,self.target_train)

		return pred,original

	def mensure(self,k,tests,predicts):
		ac_v = []
		cm_v = []
		p_v = []
		r_v = []
		f1_v = []
		e_v = []

		#print(tests[str(i)])

		for i in range(1,k+1):
			ac = accuracy_score(tests[str(i)], predicts[str(i)])
			p = precision_score(tests[str(i)], predicts[str(i)],average='weighted')
			r = recall_score(tests[str(i)], predicts[str(i)],average='weighted')
			f1 = (2*p*r)/(p+r)
			e = mean_squared_error(tests[str(i)], predicts[str(i)])
			cm = confusion_matrix(tests[str(i)], predicts[str(i)])
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

		roc = Roc()

		fpr,tpr = np.array(fpr),np.array(tpr)
		roc.set_fpr(np.sort(fpr))
		roc.set_tpr(np.sort(tpr))

		roc.set_auc(auc(roc.get_fpr(),roc.get_tpr()))

		#print(fpr)
		#print(tpr)
		#print('roc = %f'%roc_auc)


		#write_csv(fpr,'roc_2')
		#write_csv(tpr,'roc_2')


		return roc

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

	def box_plot(self,results,names):

		fig = plt.figure()
		fig.suptitle('Algorithm Comparison')
		ax = fig.add_subplot(111)
		plt.boxplot(results)
		ax.set_xticklabels(names)
		plt.show()

	def CMultinomialNV(self,alpha = 0.000001):

		nb = MultinomialNB(alpha)

		ac,ac_v,p,r,f1,e,cm = self.cross_apply(nb,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)
		log = 'nv',ac,p,r,f1,e,str(datetime.now())

		return ac,ac_v,p,r,f1,e,cm,roc_

	def CDecisionTree(self,criterion='gini'):

		dt = tree.DecisionTreeClassifier(criterion)

		ac,ac_v,p,r,f1,e,cm = self.cross_apply(dt,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)

		log = 'dt',ac,p,r,f1,e,str(datetime.now())

		return ac,ac_v,p,r,f1,e,cm,roc_

	def CSuportVectorMachine(self,gamma=0.001,C=100,decision_function_shape='ovr'):

		csvm = svm.SVC(gamma=gamma,C=C,decision_function_shape=decision_function_shape)

		ac,ac_v,p,r,f1,e,cm = self.cross_apply(csvm,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)
		log = 'svm',ac,p,r,f1,e,str(datetime.now())


		return ac,ac_v,p,r,f1,e,cm,roc_

	def CGradientDescEst(self, penalty="l2"):

		sgdc = SGDClassifier(penalty=penalty)

		ac,ac_v,p,r,f1,e,cm = self.cross_apply(sgdc,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)

		log = 'ge',ac,p,r,f1,e,str(datetime.now())
		
		return ac,ac_v,p,r,f1,e,cm,roc_

	def CRandomForest(self):
		
		rf = RandomForestClassifier()

		ac,ac_v,p,r,f1,e,cm = self.cross_apply(rf,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)

		log = 'rf',ac,p,r,f1,e,str(datetime.now())

		return ac,ac_v,p,r,f1,e,cm,roc_	

	def CLogistRegression(self,penalty="l2"):

		lr = LogisticRegression(penalty=penalty)
		ac,ac_v,p,r,f1,e,cm = self.cross_apply(lr,self.array_train,self.target_train)
		roc_  = Roc()
		roc_ = self.roc(cm)

		log = 'rl',ac,p,r,f1,e,str(datetime.now())

		return ac,ac_v,p,r,f1,e,cm,roc_

	
