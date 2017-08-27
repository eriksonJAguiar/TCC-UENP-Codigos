from sent_classification_module import *
from class_roc import Roc


if __name__ == '__main__':

	sent = SentClassifiers()

	nv_roc = Roc()
	svm_roc = Roc()
	dt_roc = Roc()
	rf_roc = Roc()
	gd_roc = Roc()
	rl_roc = Roc()

	fpr = []
	tpr = []
	auc = []

	nv_ac,nv_p,nv_r,nv_f1,nv_e,nv_cm,nv_roc = sent.CMultinomialNV()

	fpr.append(nv_roc.get_fpr())
	tpr.append(nv_roc.get_tpr())
	auc.append(nv_roc.get_auc())

	svm_ac,svm_p,svm_r,svm_f1,svm_e,svm_cm,svm_roc = sent.CSuportVectorMachine()

	fpr.append(svm_roc.get_fpr())
	tpr.append(svm_roc.get_tpr())
	auc.append(svm_roc.get_auc())

	dt_ac,dt_p,dt_r,dt_f1,dt_e,dt_cm,dt_roc = sent.CDecisionTree()

	fpr.append(dt_roc.get_fpr())
	tpr.append(dt_roc.get_tpr())
	auc.append(dt_roc.get_auc())

	rf_ac,rf_p,rf_r,rf_f1,rf_e,rf_cm,rf_roc = sent.CRandomForest()

	
	fpr.append(rf_roc.get_fpr())
	tpr.append(rf_roc.get_tpr())
	auc.append(rf_roc.get_auc())

	gd_ac,gd_p,gd_r,gd_f1,gd_e,gd_cm,gd_roc = sent.CGradientDescEst()

	
	fpr.append(gd_roc.get_fpr())
	tpr.append(gd_roc.get_tpr())
	auc.append(gd_roc.get_auc())

	rl_ac,rl_p,rl_r,rl_f1,rl_e,rl_cm,rl_roc = sent.CLogistRegression()

	
	fpr.append(rl_roc.get_fpr())
	tpr.append(rl_roc.get_tpr())
	auc.append(rl_roc.get_auc())

	label = ['nv','svm','dt','rf','gd','rl']


	#sent.plot_roc_all(fpr,tpr,auc,label)


	#sent.plot_roc(roc.get_fpr(),roc.get_tpr(),roc.get_auc(),'red','nv')

	sent.plot_confuse_matrix(rl_cm)
	