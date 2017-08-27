from sent_classification_module import *
from class_roc import Roc
import collections

if __name__ == '__main__':

	sent = SentClassifiers()

	roc = Roc()

	pesos = [1,1,1,1,1,1]

	rank,original = sent.ranking(10,pesos)

	for keys,values in rank.items():
		print(keys)
		print(values)
