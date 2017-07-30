import nltk
from textblob import TextBlob
from pymongo import MongoClient
from nltk.twitter import Twitter
import sys
import time
from datetime import datetime
import csv
import re

def grava(db,campos1,campo2,campo3,campo4):
	db.insert_one(
				{
					'_id':campos1,
					'id_user':campo2,
					'text':campo3,
					'date': campo4
				}
			)

def AcessaBd(dataBase,collection):
	try:
		client = MongoClient()

		db = client[dataBase]

		result = db[collection]

		return result

	except Exception as inst:
		print(type(inst))

def getAll(collection):
	
	return collection.find()

def getLimit(coll,n):
	return coll.find().sort('text',1).limit(n)


def removeStopwords(db,base):
	#dados = list()
	for document in base:
		expr = re.sub(r"http\S+", "", document['text'])
		expr = re.sub(r"@\S+","",expr)
		try:
			filtrado = [w for w in nltk.regexp_tokenize(expr.lower(),"[\S'#]+") if not w in nltk.corpus.stopwords.words('portuguese')]
			frase = ""
			for f in filtrado:
				frase += f + " "
			grava(db,document['_id'],document['id_user'],frase,document['created_at'])
			#dados.append(filtrado)
			#print(frase)
		except Exception as inst:
			print(type(inst))

def steaming(db,base):

	stem_pt = nltk.stem.SnowballStemmer('portuguese')
	
	for words in base:
		try:
			frase = ""
			for t in words['text']: 
				filtrado = stem_pt.stem(t.lower())
				frase += filtrado + " "
			grava(db,words['_id'],words['id_user'],frase,words['date'])
		except Exception as inst:
			print(type(inst))

def pass_to_txt(document):
	i = 0
	j = 1
	f = open('texts.txt', 'w')
	for t in document:
		if i < 999:
			f.write('%s\n'%(t['text']))
			i +=1
		else:
			f.close()
			f = open('texts%d.txt'%(j), 'w')
			i = 0
			j += 1
	
	f.close()

if __name__ == '__main__':

	print('Buscando...\n')
	print('Isso Pode Demorar um Pouco !')

	col1 = AcessaBd('baseTweetsTCC','tweets')
	col2 = AcessaBd('baseTweetsTCC','tweetsProcessing1')
	col3 = AcessaBd('baseTweetsTCC','tweetsProcessing2')

	tweets = getAll(col1)
	tweets_stop = getAll(col2)

	#pass_to_txt(tweets_stop)
	
	#tweets = getLimit(col2,500)

	print("Removendo Stop Words, Aguarde...")
	removeStopwords(col2,tweets)

	print("Passando o Steamming, Aguarde...")
	steaming(col3,tweets_stop)

	print("Texto processado com sucesso")



