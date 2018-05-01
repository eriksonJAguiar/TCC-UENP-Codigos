# -*- coding: utf-8 -*-
from sent_classification_module import *
from class_roc import Roc
import collections
from datetime import datetime as dt
import time
import pandas as pd



if __name__ == '__main__':

	sent = SentClassifiers(file='dataset-english-senders.csv')

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

	
	sent.plot_confuse_matrix(nv_cm,'Matriz de Confusao - Naive Bayes','matriz2-nv')
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

	sent.plot_confuse_matrix(svm_cm,'Matriz de Confusao - SVM','matriz-svm2')
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

	sent.plot_confuse_matrix(dt_cm,'Matriz de Confusao - Arv. Decisao','matriz2-dt')
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

	sent.plot_confuse_matrix(rf_cm,'Matriz de Confusao - Rand. Forest','matriz2-rf')
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

	sent.plot_confuse_matrix(rl_cm,'Matriz de Confusao - Reg. Logistica','matriz2-rl')
	fpr.append(rl_roc.get_fpr())
	tpr.append(rl_roc.get_tpr())
	auc.append(rl_roc.get_auc())

	l = 'rl',rl_acc,rl_p,rl_r,rl_f1,rl_e,str(dt.now())
	logs.append(l)

	start = time.time()
	sgd_acc,sgd_ac,sgd_p,sgd_r,sgd_f1,sgd_e,sgd_cm,sgd_roc = sent.CGradienteDesc()
	end = time.time()
	custos['sgd'] = [end-start]

	print('SGD')
	print('ac = %f'%sgd_acc)
	print('p = %f'%sgd_p)
	print('r = %f'%sgd_r)
	print('f1 = %f'%sgd_f1)
	print('e = %f'%sgd_e)
	print('time = %f'%(end-start))
	print('---------------')

	sent.plot_confuse_matrix(sgd_cm,'Matriz de Confusao - SGD','matriz-sgd')
	fpr.append(sgd_roc.get_fpr())
	tpr.append(sgd_roc.get_tpr())
	auc.append(sgd_roc.get_auc())

	sgd = 'sgd',sgd_acc,sgd_p,sgd_r,sgd_f1,sgd_e,str(dt.now())
	logs.append(sgd)

	results.append(nv_ac)
	results.append(svm_ac)
	results.append(dt_ac)
	results.append(rf_ac)
	results.append(rl_ac)
	results.append(sgd_ac)

	acuracias.append(nv_acc)
	acuracias.append(svm_acc)
	acuracias.append(dt_acc)
	acuracias.append(rf_acc)
	acuracias.append(rl_acc)

	start = time.time()
	pesos = sent.calc_weigth(acuracias)


	names = ['naive','svm','tree','forest','logistic','sgd','committee']

	ac,cmm_ac,p,r,f1,e,cm_cm,cm_roc = sent.committee(pesos)
	end = time.time()
	custos['cm'] = [end-start]
	
	results.append(cmm_ac)

	print("Comite")
	print("Acuracia %f"%ac)
	print("Precisao %f"%p)
	print("Recall %f"%r)
	print("F1 Score %f"%f1)
	print("Erro %f"%e)
	print('time = %f'%(end-start))
	print("--------------------------")

	sent.write_csv(custos,'Novos_Experimentos/custo-tempo')

	sent.plot_confuse_matrix(cm_cm,'Matriz de Confusao - Comite','matriz2-cm')
	fpr.append(cm_roc.get_fpr())
	tpr.append(cm_roc.get_tpr())
	auc.append(cm_roc.get_auc())

	l = 'cm',ac,p,r,f1,e,str(dt.now())
	logs.append(l)

	#df_ac = sent.read_csv('acuracias-pt-lexical')

	#results.append(df_ac['acuracia'])

	#sent.write_csv(lines,'committee')

	sent.write_csv(logs,'Novos_Experimentos/metricas')
	#sent.write_dataframe()

	sent.plot_roc_all(fpr,tpr,auc,names)
	sent.box_plot(results,names,'comparacao entre os algoritmos','boxplot1')
	
	
	

