import distance
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.metrics import accuracy_score
from gensim.models import word2vec as w2v
from gensim.models.word2vec import LineSentence
from gensim.models import Doc2Vec

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
		#pos = []
		#neg = []
		#for word_pos in train_pos:
			#s = distance.nlevenshtein(word_pos, f, method=1)
			#s = distance.hamming(word_pos, f, normalized=True)
			#pos.append(s)
		
		#for word_neg in train_neg:
			#s = distance.hamming(word_neg, f, normalized=True)
			#s = distance.nlevenshtein(word_neg, f, method=1)
			#neg.append(s)

		#pos.sort()
		print(f)
		#try:
		pos = model_pos.similar_by_vector(f, topn=10, restrict_vocab=None)
		neg = model_neg.similar_by_vector(f, topn=10, restrict_vocab=None)

		if pos[0] > neg[0]:
				frase_sent.append(1)

		else:
			frase_sent.append(-1)

		#except Exception:
			#frase_sent.append(0)
			

		
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

def init():
	words_neg = read_file_txt("dataset/Negativo")
	write_file_txt("dataset/Negativo_",words_neg)

	words_pos = read_file_txt("dataset/Positivo")
	write_file_txt("dataset/Positivo_",words_pos)

if __name__ == '__main__':

	#train  = read_csv("dataset-portuguese")
	#y_true = np.array(train['opiniao'])
	#words_pos = write_file("dataset/Positivo")

	#init()

	s = "Jogar Futebol"

	s = clean(s)

	f_sent = sent_frase(s)
	#sent = calc_sent(f_sent)
	print(f_sent)
	#print(sent)

	#sent = []
	#for f in train['tweet']:
	#	frase = clean(f)
	#	frase_sent = sent_frase(words_neg,words_pos,frase)
	#	s = calc_sent(frase_sent)
	#	sent.append(s)


	#y_pred = np.array(sent)

	#ac = mensure(y_true,y_pred)

	#print("ac = "%ac)

