from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, KFold,GroupKFold
from sklearn.metrics import confusion_matrix
from gensim.models import word2vec as w2v
from gensim.models.word2vec import LineSentence

import matplotlib.pyplot as plt
import distance
import pandas as pd
import numpy as np
import nltk
import re
import itertools
import statistics

def read_file_txt(file):
	text_file = open("%s.txt"%file, "r",encoding='utf-8')
	lines = text_file.readlines()

	words = []

	for l in lines:
		expr = re.sub(r"[\[\]]","",l)
		words.append(expr)

	return words

def write_file_txt(new_file,datas):
	file = open('%s.txt'%new_file, 'w')
	for item in datas:
  		file.write("%s\n" % item)

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

def sent_frase(frase_tokens):

	frase_sent = []

	#neg.sort()
	sentences_neg = read_file_txt("dataset/Negativo_")
	#sentences_neg = LineSentence("dataset/Negativo_.txt")
	#sentences_pos = LineSentence("dataset/Positivo_.txt")
	sentences_pos = read_file_txt("dataset/Positivo_")


	model_neg = w2v.Word2Vec(min_count=1)
	model_pos = w2v.Word2Vec(min_count=1)
	model_neg.build_vocab(sentences_neg)
	model_pos.build_vocab(sentences_pos)
	model_neg.train(sentences_neg, total_examples=model_neg.corpus_count, epochs=model_neg.iter)
	model_pos.train(sentences_pos, total_examples=model_pos.corpus_count, epochs=model_pos.iter)
	
	for f in frase_tokens:
		try:
			pos = model_pos.similar_by_vector(f, topn=10, restrict_vocab=None)
			neg = model_neg.similar_by_vector(f, topn=10, restrict_vocab=None)

			if pos[0] > neg[0]:
				frase_sent.append(1)

			else:
				frase_sent.append(-1)

		except Exception:
			frase_sent.append(0)
			continue
			

		
	return frase_sent

def calc_sent(frase_sent):
	p = frase_sent.count(1)
	n = frase_sent.count(-1)

	if p + n == 0:
		return 0

	np = (p/(p+n))
	nn = (n/(p+n))

	return int(nn-np)

def convert_df(df):
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

def init():
	words_neg = read_file_txt("dataset/Negativo")
	write_file_txt("dataset/Negativo_",words_neg)

	words_pos = read_file_txt("dataset/Positivo")
	write_file_txt("dataset/Positivo_",words_pos)

def sent_mensure(dataset):

	sent = []
	for f in dataset:
		frase = clean(f)
		frase_sent = sent_frase(frase)
		s = calc_sent(frase_sent)
		sent.append(s)

	return sent

def mensure():

	X = read_csv("dataset-portuguese")
	X['opiniao'] = convert_df(X['opiniao'])
	X_text = X['tweet']
	target = np.array(X['opiniao'])

	kf = KFold(10, shuffle=True, random_state=1)

	metricas = pd.DataFrame()
	metricas2 = pd.DataFrame()
	ac_v = []
	cm_v = []
	p_v = []
	r_v = []
	f1_v = []
	e_v = []
	predicts = []

	for train_index,teste_index in kf.split(X_text,target):
		#X_test = X_text[teste_index]
		y_test = target[teste_index]
		
		pred = sent_mensure(X_test)

		predicts += pred.tolist() 
		ac = accuracy_score(y_test, pred)
		p = precision_score(y_test, pred,average='weighted')
		r = recall_score(y_test, pred,average='weighted')
		f1 = (2*p*r)/(p+r)
		e = mean_squared_error(y_test, pred)
		cm = confusion_matrix(y_test,pred)
		cm_v.append(cm)
		ac_v.append(ac)
		p_v.append(p)
		r_v.append(r)
		f1_v.append(f1)
		e_v.append(e)

	metricas['acuracia'] = [statistics.median(ac_v)]
	metricas2['acuracia'] = ac_v
	metricas['precisao'] = [statistics.median(p_v)]
	metricas2['precisao'] = p_v
	metricas['recall'] = [statistics.median(r_v)]
	metricas2['recall'] = r_v
	metricas['f1'] = [statistics.median(f1_v)]
	metricas2['f1'] = f1_v
	metricas['erro'] = [statistics.median(e_v)]
	metricas2['erro'] = e_v
	cm_median = self.matrix_confuse_median(cm_v)

	plot_confuse_matrix(cm_median,'Matriz de Confusao TSViz','TSviz-confuse')
	
	write_csv(metricas,'experimentos-final/metricas-TSviz')
	write_csv(metricas,'experimentos-final/metricas-parciais-TSviz')
	write_csv(predicts,'experimentos-final/TSviz-predicoes')

def mensure_():

	X = read_csv("dataset-portuguese")
	y_test = np.array(convert_df(X['opiniao']))
	Y = read_csv("dataset-TSViz")
	pred = np.array(Y['opiniao'])

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

	plot_confuse_matrix(cm,'Matriz de confusao TSViz','confuse-TSviz')

	write_csv(metricas,'experimentos-final/metricas-TSviz')

if __name__ == '__main__':


	mensure_()
	print("Finalizado")
	

	


