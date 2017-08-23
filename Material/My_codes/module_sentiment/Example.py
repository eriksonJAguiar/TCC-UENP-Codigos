from sent_classification_module import *
from class_roc import Roc


if __name__ == '__main__':

	sent = SentClassifiers()

	roc = Roc()

	ac,p,r,f1,e,cm,roc = sent.CMultinomialNV()

	#sent.plot_roc(roc.get_fpr(),roc.get_tpr(),roc.get_auc(),'red','nv')

	sent.plot_confuse_matrix(cm)