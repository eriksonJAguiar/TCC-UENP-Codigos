from sent_classification_module import *
from class_roc import Roc


if __name__ == '__main__':

	sent = SentClassifiers('dataset-portuguese')

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
	acuracias = []

	nv_ac,_,nv_p,nv_r,nv_f1,nv_e,nv_cm,nv_roc = sent.CMultinomialNV()

	print("Naive")
	print('ac = %f'%nv_ac)
	print('p = %f'%nv_p)
	print('r = %f'%nv_r)
	print('f1 = %f'%nv_f1)
	print('e = %f'%nv_e)
	print('---------------')

	acuracias.append(nv_ac)
	fpr.append(nv_roc.get_fpr())
	tpr.append(nv_roc.get_tpr())
	auc.append(nv_roc.get_auc())

	#sent.plot_confuse_matrix(nv_cm)

	svm_ac,_,svm_p,svm_r,svm_f1,svm_e,svm_cm,svm_roc = sent.CSuportVectorMachine()

	print("SVM")
	print('ac = %f'%svm_ac)
	print('p = %f'%svm_p)
	print('r = %f'%svm_r)
	print('f1 = %f'%svm_f1)
	print('e = %f'%svm_e)
	print('---------------')

	acuracias.append(svm_ac)
	fpr.append(svm_roc.get_fpr())
	tpr.append(svm_roc.get_tpr())
	auc.append(svm_roc.get_auc())

	dt_ac,_,dt_p,dt_r,dt_f1,dt_e,dt_cm,dt_roc = sent.CDecisionTree()

	print("Arvore de Decisao")
	print('ac = %f'%dt_ac)
	print('p = %f'%dt_p)
	print('r = %f'%dt_r)
	print('f1 = %f'%dt_f1)
	print('e = %f'%dt_e)
	print('---------------')

	acuracias.append(dt_ac)
	fpr.append(dt_roc.get_fpr())
	tpr.append(dt_roc.get_tpr())
	auc.append(dt_roc.get_auc())

	rf_ac,_,rf_p,rf_r,rf_f1,rf_e,rf_cm,rf_roc = sent.CRandomForest()

	print("Radom Forest")
	print('ac = %f'%rf_ac)
	print('p = %f'%rf_p)
	print('r = %f'%rf_r)
	print('f1 = %f'%rf_f1)
	print('e = %f'%rf_e)
	print('---------------')
	

	acuracias.append(rf_ac)
	fpr.append(rf_roc.get_fpr())
	tpr.append(rf_roc.get_tpr())
	auc.append(rf_roc.get_auc())


	rl_ac,_,rl_p,rl_r,rl_f1,rl_e,rl_cm,rl_roc = sent.CLogistRegression()
	
	print('Regressao Logistica')
	print('ac = %f'%rl_ac)
	print('p = %f'%rl_p)
	print('r = %f'%rl_r)
	print('f1 = %f'%rl_f1)
	print('e = %f'%rl_e)
	print('---------------')

	acuracias.append(rl_ac)
	fpr.append(rl_roc.get_fpr())
	tpr.append(rl_roc.get_tpr())
	auc.append(rl_roc.get_auc())

	pesos = sent.calc_weigth(acuracias)

	k = 10

	#pred,original = sent.committee(k,pesos)

	pesos = sent.calc_weigth(acuracias)

	cm_ac,_,cm_p,cm_r,cm_f1,cm_e,cm_median,cm_roc = sent.committee(k,pesos)

	print('Comitê')
	print('ac = %f'%cm_ac)
	print('p = %f'%cm_p)
	print('r = %f'%cm_r)
	print('f1 = %f'%cm_f1)
	print('e = %f'%cm_e)
	print('---------------')

	#cm_roc = sent.roc(cm_mean)

	fpr.append(cm_roc.get_fpr())
	tpr.append(cm_roc.get_tpr())
	auc.append(cm_roc.get_auc())


	label = ['naive','svm','tree','forest','logistic','committee']


	sent.plot_roc_all(fpr,tpr,auc,label)

	#sent.plot_roc(roc.get_fpr(),roc.get_tpr(),roc.get_auc(),'red','nv')

	#sent.plot_confuse_matrix(nv_cm)
	