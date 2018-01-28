from sent_classification_module import *
from class_roc import Roc
import collections
from datetime import datetime as dt
import time
import pandas as pd
import os
import random


def read_file(file):

	df1 = pd.DataFrame.from_csv('datasets/Result/%s'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

	df1 = df1.reset_index()

	return df1

def write_csv(self,data,file):
		df = pd.DataFrame(data)
		df.to_csv(file+'.csv', mode='a', sep=';',index=False, header=False)

def predict(sent):
	
	files = []
	for _, _, f in os.walk('datasets'):
		files.append(f)


	for f in files[0]:
		df = sent.mread_csv(f)
		print('Realizando a predição para %s'%(f))
		print('Aguarde...')
		df_pred = sent.pred_texts(df['tweet'].values)

		df['sentiment'] = df_pred['sentiment']

		sent.write_csv(df,'datasets/Result/%s'%(f))

		print('Finalizado para o dataset %s'%(f))


def config_dataset():

	files = []

	for _, _, f in os.walk('datasets/Result'):
		files.append(f)

	twitter = pd.DataFrame(columns=['user','tweet','coordenada','horario','sentiment'])

	for f in files[0]:
		df = read_file(f)
		twitter = pd.concat([twitter,df])


	lines = twitter.count()['tweet']

	twt = []
	snt = []

	new_twitter = pd.DataFrame(columns=['tweet','sentiment'])

	for i in range(1,10):
		for i in range(1,20):
			r = random.randint(0,lines-1)
			t = twitter['tweet'].values
			s = twitter['sentiment'].values
			twt.append(t[r])
			snt.append(int(s[r]))

	new_twitter['tweet'] = twt
	new_twitter['sentiment'] = snt

	write_file(new_twitter,'datasets/Result/dataset-parcial')

	return twitter

def mensure(sent):

	results = []
	acuracias = []
	logs = []
	nv_roc = Roc()
	svm_roc = Roc()
	dt_roc = Roc()
	rf_roc = Roc()
	gd_roc = Roc()
	rl_roc = Roc()
	cm_roc = Roc()
	fpr = []
	tpr = []
	auc = []

	custos = pd.DataFrame()

	start = time.time()
	nv_acc,nv_ac,nv_p,nv_r,nv_f1,nv_e,nv_cm,nv_roc = sent.CMultinomialNV()
	end = time.time()
	custos['nv'] = [end-start]
	print('Naive')
	print('ac = %f'%nv_acc)
	print('p = %f'%nv_p)
	print('r = %f'%nv_r)
	print('f1 = %f'%nv_f1)
	print('e = %f'%nv_e)
	print("time %f"%(end-start))
	print('---------------')

	sent.plot_confuse_matrix(nv_cm,'Matriz de Confusao - Naive Bayes','matriz-nv')


	l = 'nv',nv_acc,nv_p,nv_r,nv_f1,nv_e,str(dt.now())
	logs.append(l)
	fpr.append(nv_roc.get_fpr())
	tpr.append(nv_roc.get_tpr())
	auc.append(nv_roc.get_auc())

	start = time.time()
	sgd_acc,sgd_ac,sgd_p,sgd_r,sgd_f1,sgd_e,sgd_cm,sgd_roc = sent.gradienteDesc()
	end = time.time()
	custos['sgd'] = [end-start]
	print('Gradiente')
	print('ac = %f'%sgd_acc)
	print('p = %f'%sgd_p)
	print('r = %f'%sgd_r)
	print('f1 = %f'%sgd_f1)
	print('e = %f'%sgd_e)
	print("time %f"%(end-start))
	print('---------------')

	sent.plot_confuse_matrix(sgd_cm,'Matriz de Confusao - SGD','matriz-sgd')


	l = 'sgd',sgd_acc,sgd_p,sgd_r,sgd_f1,sgd_e,str(dt.now())
	logs.append(l)

	fpr.append(sgd_roc.get_fpr())
	tpr.append(sgd_roc.get_tpr())
	auc.append(sgd_roc.get_auc())

	results.append(sgd_ac)
	results.append(nv_ac)

	sent.write_csv(custos,'Result/tempo-exe')

	sent.write_csv(logs,'Result/metricas')

	label = ['sgd','naive']
	
	sent.plot_roc_all(fpr,tpr,auc,label)

if __name__ == '__main__':

	#dataset = config_dataset()

	#sent = SentClassifiers(dataframe=dataset)

	#mensure(sent)

	sent = SentClassifiers('train/dataset-portuguese.csv')

	predict(sent)

	
	
	
