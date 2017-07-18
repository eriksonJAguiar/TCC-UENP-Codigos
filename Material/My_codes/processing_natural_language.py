import nltk
from textblob import TextBlob
from pymongo import MongoClient
import sys
import time
from datetime import datetime

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

def removeStopwords(db,base):
	dados = list()
	for document in base:
		try:
			filtrado = [w for w in nltk.regexp_tokenize(document['text'].lower(),"[\w'@#]+") if not w in nltk.corpus.stopwords.words('portuguese')]
			frase = ""
			for f in filtrado:
				frase += f + " "
			grava(db,document['_id'],document['id_user'],frase,document['created_at'])
			dados.append(filtrado)
		except Exception as inst:
			print(type(inst))


	return dados

def steaming(db,base):

	stem_pt = nltk.stem.SnowballStemmer('portuguese')

	stem_apply = list()

	stop_base = getAll(db['tweetsProcessing1'])

	i = 0
	
	for words in base:
		try:
			list_stem = list()
			frase = ""
			for t in words: 
				filtrado = stem_pt.stem(t.lower())
				list_stem.append(filtrado)
				frase += filtrado + " "
			stem_apply.append(list_stem)
			#grava(db,(i+1),stop_base[i]['id_user'],frase)
			i += 1

		except Exception as inst:
			print(type(inst))

	return stem_apply

	

def synonyms(db,base):
	
	dados = list()


	for words in base:
		#try:
		list_syn = list()
		for t in words: 
			pt_blob = TextBlob(u'%s'%t)
			en_blol = pt_blob.translate(to='pt')
			print(en_blol)

			#grava(db,i,list_stem)
			#i += 1

		#except Exception as inst:
		#	print(type(inst))

def classification():
	return 0

def getLimit(coll,n):
	return coll.find().sort('text',1).limit(n)

if __name__ == '__main__':

	print('Buscando...\n')
	print('Isso Pode Demorar um Pouco !')

	col1 = AcessaBd('baseTweetsTCC','tweets')
	col2 = AcessaBd('baseTweetsTCC','tweetsProcessing1')
	col3 = AcessaBd('baseTweetsTCC','tweetsProcessing2')
	#col4 = AcessaBd('baseTweetsTCC','tweetsProcessing3')

	tweets = getAll(col1)
	#tweets_stop = getAll(col2)
	#tweets_stem = getAll(col3)
	
	#tweets = getLimit(col2,500)

	sem_stop = removeStopwords(col2,tweets)

	#stem_apply = steaming(col3,sem_stop)

	#synonyms_apply = synonyms(col4,stem_apply)

	print("Texto processado com sucesso")



