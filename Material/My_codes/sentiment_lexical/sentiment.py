import distance
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.metrics import accuracy_score

def write_file(file):
	text_file = open("%s.txt"%file, "r",encoding='latin-1')
	lines = text_file.readlines()

	words = []

	for l in lines:
		expr = re.sub(r"\W+","",l)
		words.append(expr)

	return words

def read_csv(file):

		df1 = pd.DataFrame.from_csv('../files_extern/%s.csv'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

		df1 = df1.reset_index()

		return df1

def clean(frase):
	expr = re.sub(r"http\S+", "", frase)
	#expr = re.sub(r"[@#]\S+","",expr)
	filtrado = [w for w in nltk.regexp_tokenize(expr.lower(),"[\w]+") if not w in nltk.corpus.stopwords.words('portuguese')]
	
	return filtrado

def sent_frase(train_neg,train_pos,frase_tokens):

	frase_sent = []
	
	for f in frase_tokens:
		pos = []
		neg = []
		for word_pos in train_pos:
			s = distance.nlevenshtein(word_pos, f, method=1)
			#s = distance.hamming(word_pos, f, normalized=True)
			pos.append(s)
		
		for word_neg in train_neg:
			#s = distance.hamming(word_neg, f, normalized=True)
			s = distance.nlevenshtein(word_neg, f, method=1)
			neg.append(s)

		pos.sort()
		neg.sort()
		
		if pos[0] > neg[0]:
			frase_sent.append(1)

		else:
			frase_sent.append(-1)

		
	return frase_sent

def calc_sent(frase_sent):
	p = frase_sent.count(1)
	n = frase_sent.count(-1)

	np = (p/(p+n))
	nn = (n/(p+n))

	return int(nn-np)

def mensure(y_true,y_pred):

	ac = accuracy_score(y_true,y_pred)

	return ac


if __name__ == '__main__':

	train  = read_csv("dataset-portuguese")
	y_true = np.array(train['opiniao'])
	words_neg = write_file("dataset/Negativo")
	words_pos = write_file("dataset/Positivo")
	
	sent = []
	for f in train['tweet']:
		frase = clean(f)
		frase_sent = sent_frase(words_neg,words_pos,frase)
		s = calc_sent(frase_sent)
		sent.append(s)


	y_pred = np.array(sent)

	ac = mensure(y_true,y_pred)

	print("ac = "%ac)

