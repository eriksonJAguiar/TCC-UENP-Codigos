from sent_classification_module import *
from class_roc import Roc
import collections
from datetime import datetime as dt

if __name__ == '__main__':

	sent = SentClassifiers('dataset-portuguese')

	results = []
	acuracias = []

	nv_acc,nv_ac,_,_,_,_,_,_ = sent.CMultinomialNV()
	dt_acc,dt_ac,_,_,_,_,_,_ = sent.CDecisionTree()
	svm_acc,svm_ac,_,_,_,_,_,_ = sent.CSuportVectorMachine()
	gd_acc,gd_ac,_,_,_,_,_,_ = sent.CGradientDescEst()
	rf_acc,rf_ac,_,_,_,_,_,_ = sent.CRandomForest()
	rl_acc,rl_ac,_,_,_,_,_,_ = sent.CLogistRegression()

	results.append(nv_ac)
	results.append(svm_ac)
	results.append(dt_ac)
	results.append(rf_ac)
	results.append(gd_ac)
	results.append(rl_ac)

	acuracias.append(nv_acc)
	acuracias.append(svm_acc)
	acuracias.append(dt_acc)
	acuracias.append(rf_acc)
	acuracias.append(gd_acc)
	acuracias.append(rl_acc)

	pesos = sent.calc_weigth(acuracias)

	k = 10

	pred,original = sent.committee(k,pesos)

	line = []

	names = ['naive','svm','tree','forest','gradient','logistic','committee']

	ac,cmm_ac,p,r,f1,e,_ = sent.mensure(k,pred,original)

	results.append(cmm_ac)

	print("Metricas")
	print("--------------------------")
	print("Acuracia %f"%ac)
	print("Precisao %f"%p)
	print("Recall %f"%r)
	print("F1 Score %f"%p)
	print("Erro %f"%e)
	print("--------------------------")

	l = ac,p,r,f1,e,str(dt.now())
	line.append(l)

	sent.write_csv(line,'committee')

	sent.box_plot(results,names)


	

