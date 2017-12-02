from sent_classification_module import *
from class_roc import Roc
import collections
from datetime import datetime as dt
import time
import pandas as pd
import os
import random


def read_file(file):

	df1 = pd.DataFrame.from_csv('datasets-allan/%s'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

	df1 = df1.reset_index()

	return df1

def write_file(data,file):
		df = pd.DataFrame(data)
		df.to_csv('Result/'+file+'.csv', mode='a', sep=';',index=False, header=False)


def predict():

	for _, _, f in os.walk('datasets-allan'):
		files.append(f)

	
	for f in files[0]:
		df = read_file('datasets-allan/%s'%(f))
		print('Realizando a predição para %s'%(f))
		print('Aguarde...')
		df_pred = sent.pred_texts(df['tweet'].values)

		df['sentiment'] = df_pred['sentiment']

		sent.write_csv(df,'datasets-allan/result/%s'%(f))

		print('Finalizado para o dataset %s'%(f))


def config_dataset():

	files = []

	for _, _, f in os.walk('datasets-allan'):
		print(f)
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
			snt.append(s[r])

	#new_twitter['tweet'] = twt
	#new_twitter['sentiment'] = snt

	write_file(new_twitter,'dataset-parcial')

	return twitter


if __name__ == '__main__':

	dataset = config_dataset()

	exit(0)

	sent = SentClassifiers(dataframe=dataset)

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
	fpr.append(nv_roc.get_fpr())
	tpr.append(nv_roc.get_tpr())
	auc.append(nv_roc.get_auc())

	l = 'nv',nv_acc,nv_p,nv_r,nv_f1,nv_e,str(dt.now())
	logs.append(l)

	start = time.time()
	svm_acc,svm_ac,svm_p,svm_r,svm_f1,svm_e,svm_cm,svm_roc = sent.CSuportVectorMachine()
	end = time.time()
	custos['svm'] = [end-start]
	print('SVM')
	print('ac = %f'%svm_acc)
	print('p = %f'%svm_p)
	print('r = %f'%svm_r)
	print('f1 = %f'%svm_f1)
	print('e = %f'%svm_e)
	print('time = %f'%(end-start))
	print('---------------')

	sent.plot_confuse_matrix(svm_cm,'Matriz de Confusao - SVM','matriz-svm')
	fpr.append(svm_roc.get_fpr())
	tpr.append(svm_roc.get_tpr())
	auc.append(svm_roc.get_auc())

	l = 'svm',svm_acc,svm_p,svm_r,svm_f1,svm_e,str(dt.now())
	logs.append(l)
	
	start = time.time()
	dt_acc,dt_ac,dt_p,dt_r,dt_f1,dt_e,dt_cm,dt_roc = sent.CDecisionTree()
	end = time.time()
	custos['dt'] = [end-start]
	print('Decisao')
	print('ac = %f'%dt_acc)
	print('p = %f'%dt_p)
	print('r = %f'%dt_r)
	print('f1 = %f'%dt_f1)
	print('e = %f'%dt_e)
	print('time = %f'%(end-start))
	print('---------------')

	sent.plot_confuse_matrix(dt_cm,'Matriz de Confusao - Arv. Decisao','matriz-dt')
	fpr.append(dt_roc.get_fpr())
	tpr.append(dt_roc.get_tpr())
	auc.append(dt_roc.get_auc())

	l = 'dt',dt_acc,dt_p,dt_r,dt_f1,dt_e,str(dt.now())
	logs.append(l)

	start = time.time()
	rf_acc,rf_ac,rf_p,rf_r,rf_f1,rf_e,rf_cm,rf_roc = sent.CRandomForest()
	end = time.time()
	custos['rf'] = [end-start]
	print('Forest')
	print('ac = %f'%rf_acc)
	print('p = %f'%rf_p)
	print('r = %f'%rf_r)
	print('f1 = %f'%rf_f1)
	print('e = %f'%rf_e)
	print('time = %f'%(end-start))
	print('---------------')

	sent.plot_confuse_matrix(rf_cm,'Matriz de Confusao - Rand. Forest','matriz-rf')
	fpr.append(rf_roc.get_fpr())
	tpr.append(rf_roc.get_tpr())
	auc.append(rf_roc.get_auc())

	l = 'rf',rf_acc,rf_p,rf_r,rf_f1,rf_e,str(dt.now())
	logs.append(l)

	start = time.time()
	rl_acc,rl_ac,rl_p,rl_r,rl_f1,rl_e,rl_cm,rl_roc = sent.CLogistRegression()
	end = time.time()
	custos['rl'] = [end-start]

	print('Logistic')
	print('ac = %f'%rl_acc)
	print('p = %f'%rl_p)
	print('r = %f'%rl_r)
	print('f1 = %f'%rl_f1)
	print('e = %f'%rl_e)
	print('time = %f'%(end-start))
	print('---------------')

	sent.plot_confuse_matrix(rl_cm,'Matriz de Confusao - Reg. Logistica','matriz-rl')
	fpr.append(rl_roc.get_fpr())
	tpr.append(rl_roc.get_tpr())
	auc.append(rl_roc.get_auc())

	l = 'rl',rl_acc,rl_p,rl_r,rl_f1,rl_e,str(dt.now())
	logs.append(l)

	results.append(nv_ac)
	results.append(svm_ac)
	results.append(dt_ac)
	results.append(rf_ac)
	results.append(rl_ac)

	acuracias.append(nv_acc)
	acuracias.append(svm_acc)
	acuracias.append(dt_acc)
	acuracias.append(rf_acc)
	acuracias.append(rl_acc)

	start = time.time()
	pesos = sent.calc_weigth(acuracias)


	names = ['naive','svm','tree','forest','logistic','committee']

	ac,cmm_ac,p,r,f1,e,cm_cm,cm_roc = sent.committee(pesos)
	end = time.time()
	custos['cm'] = [end-start]
	
	results.append(cmm_ac)

	print("Comitê")
	print("Acuracia %f"%ac)
	print("Precisao %f"%p)
	print("Recall %f"%r)
	print("F1 Score %f"%f1)
	print("Erro %f"%e)
	print('time = %f'%(end-start))
	print("--------------------------")

	sent.write_csv(custos,'custo-tempo')

	sent.plot_confuse_matrix(cm_cm,'Matriz de Confusao - Comite','matriz-cm')
	fpr.append(cm_roc.get_fpr())
	tpr.append(cm_roc.get_tpr())
	auc.append(cm_roc.get_auc())

	l = 'cm',ac,p,r,f1,e,str(dt.now())
	logs.append(l)

	sent.write_csv(logs,'metricas')

	sent.plot_roc_all(fpr,tpr,auc,names)
	sent.box_plot(results,names,'comparação entre os algoritmos','boxplot')
	
	
