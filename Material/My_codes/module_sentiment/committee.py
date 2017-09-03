from sent_classification_module import *
from class_roc import Roc
import collections
from datetime import datetime as dt

if __name__ == '__main__':

	sent = SentClassifiers()

	roc = Roc()

	pesos = [1,1,1,1,1,1]

	k = 10

	pred,original = sent.committee(k,pesos)

	line = []

	names = ['naive','svm','tree','forest','gradient','logistic','committee']
	results = []

	ac,cmm_ac,p,r,f1,e,_ = sent.mensure(k,pred,original)
	_,nv_ac,_,_,_,_,_,_ = sent.CMultinomialNV()
	_,dt_ac,_,_,_,_,_,_ = sent.CDecisionTree()
	_,svm_ac,_,_,_,_,_,_ = sent.CSuportVectorMachine()
	_,gd_ac,_,_,_,_,_,_ = sent.CGradientDescEst()
	_,rf_ac,_,_,_,_,_,_ = sent.CRandomForest()
	_,rl_ac,_,_,_,_,_,_ = sent.CLogistRegression()

	results.append(nv_ac)
	results.append(svm_ac)
	results.append(dt_ac)
	results.append(rf_ac)
	results.append(gd_ac)
	results.append(rl_ac)
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


	

