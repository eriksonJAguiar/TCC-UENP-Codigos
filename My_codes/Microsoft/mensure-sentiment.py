from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, KFold,GroupKFold
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import distance
import pandas as pd
import numpy as np
import nltk
import re
import itertools
import statistics


def read_csv(file):

		df1 = pd.DataFrame.from_csv('../files_extern/%s.csv'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

		df1 = df1.reset_index()

		return df1

def write_csv(data,file):
	df = pd.DataFrame(data)
	#df = df.set_index(['opiniao', 'tweet'])
	df.to_csv('../files_extern/%s.csv'%(file), mode='a', sep=';',index=False, header=False)


def clean(frase):
	expr = re.sub(r"http\S+", "", frase)
	#expr = re.sub(r"[@#]\S+","",expr)
	filtrado = [w for w in nltk.regexp_tokenize(expr.lower(),"[\w]+") if not w in nltk.corpus.stopwords.words('portuguese')]
	
	return filtrado


def convert_df(df):
	new_df = []
	for d in df:
		d = float(d.replace(',','.'))
		if d >= 0.6:
			new_df.append(1)
			
		elif d > 0.4 and d < 0.6:
			new_df.append(0)
			
		elif d <= 0.4:
			new_df.append(-1)
	
	return new_df

def convert_df_(df):
	new_df = []
	for d in df:
		if d == 'Positivo' or d =='Positive':
			new_df.append(1)
			
		elif d == 'Neutro' or d =='Neutral':
			new_df.append(0)
			
		elif d == 'Negativo' or d == 'Negative':
			new_df.append(-1)
	
	return new_df

def plot_confuse_matrix(cm,title,file_name):
		labels = ['Negativo', 'Neutro','Positivo']
		cm = np.ceil(cm)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm)
		plt.title(title)
		fig.colorbar(cax)
		ax.set_xticklabels([''] + labels)
		ax.set_yticklabels([''] + labels)

		thresh = cm.max()/2
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.xlabel('Predito')
		plt.ylabel('Verdadeiro')
		plt.savefig('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/Figuras/experimentos-final/%s.png'%(file_name))
		#plt.show()

def mensure_():

	X = read_csv("dataset-portuguese")
	y_test = np.array(convert_df_(X['opiniao']))
	Y = read_csv("dataset-microsoft")
	pred = np.array(convert_df(Y['Sentimento_Microsoft']))

	metricas = pd.DataFrame()
	
	ac = accuracy_score(y_test, pred)
	p = precision_score(y_test, pred,average='weighted')
	r = recall_score(y_test, pred,average='weighted')
	f1 = (2*p*r)/(p+r)
	e = mean_squared_error(y_test, pred)
	cm = confusion_matrix(y_test,pred)

	metricas['acuracia'] = [ac]
	metricas['precisao'] = [p]
	metricas['recall'] = [r]
	metricas['f1'] = [f1]
	metricas['erro'] = [e]

	plot_confuse_matrix(cm,'Matriz de confusao Microsoft','confuse-microsoft')

	write_csv(metricas,'experimentos-final/metricas-microsoft')

if __name__ == '__main__':


	mensure_()
	print("Finalizado")