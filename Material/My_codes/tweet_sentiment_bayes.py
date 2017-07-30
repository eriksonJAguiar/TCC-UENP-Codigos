from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import numpy as np
import pandas as pd
import nltk
import re
import json
import sys
from pymongo import MongoClient

client = MongoClient()
db = client.baseTweetsTCC

params = sys.argv[1:]

def insert_One(tweet,sentiment):
	##DataBase
	db.sentiment_train.insert_one({
			'text': tweet,
			'sentiment': sentiment
		})

def populaBase(df):
	records = json.loads(df.T.to_json()).values()
	#print(records)
	db.sentiment_train.insert(records)

def getSTrain():
	
	tweets = db['sentiment_train'].find({},{'_id':0, 'index':0})

	return tweets

def getSTest():
	
	tweets = db['tweetsProcessing1'].find({},{'_id':0, 'index':0})

	return tweets

def read_csv():

	df1 = pd.DataFrame.from_csv('files_extern/tweets-1.csv',sep=';',index_col=0,encoding ='ISO-8859-1')

	df1 = df1.reset_index()

	df2 = pd.DataFrame.from_csv('files_extern/tweets-2.csv',sep=';',index_col=0,encoding ='ISO-8859-1')

	df2 = df2.reset_index()

	df_full = pd.concat([df1,df2])

	return df_full

def convert_df(df):
	new_df = []
	for d in df['opiniao']:
		if d == 'Positivo':
			new_df.append(1)
		
		elif d == 'Neutro':
			new_df.append(0)
		
		elif d == 'Negativo':
			new_df.append(-1)

	return new_df

def clean(dataframe):
	new_df = []
	for df in dataframe['tweet']:
		expr = re.sub(r"http\S+", "", df)
		expr = re.sub(r"@\S+","",expr)
		
		filtrado = [w for w in nltk.regexp_tokenize(expr.lower(),"[\S'#]+") if not w in nltk.corpus.stopwords.words('portuguese')]
		frase = ""
		for f in filtrado:
			frase += f + " "
		new_df.append(frase)

	return new_df

def init():
	dataframe = read_csv()

	dataframe['opiniao'] = convert_df(dataframe)

	dataframe['tweet'] = clean(dataframe)

	dataframe = dataframe.reset_index()

	populaBase(dataframe)


def split_base(base,target):
	test = []
	train = []
	target_train = []
	target_test = []
	percent = int(len(base)*0.1)
	
	it = int(len(base)/percent)
	
	for i in range(it):
		x_train, x_test, x_target_train,x_target_test = train_test_split(base,target,test_size=0.1)
		train.append(x_train)
		target_train.append(x_target_train)
		test.append(x_test)
		target_test.append(x_target_test)

	return train, target_train, test,target_test

def cross_apply(naive,train,target_train,test,target_test):
	count_vect = CountVectorizer()
	accuracy = []
	it = len(train)
	for i in range(it):
		X_train = count_vect.fit_transform(train[i])
		X_test = count_vect.transform(test[i])
		naive.fit(X_train,target_train[i])
		pred = naive.predict(X_test)
		ac = accuracy_score(target_test[i], pred)
		accuracy.append(ac)

	ac_mean = sum(accuracy)/len(accuracy)

	return ac_mean



if __name__ == '__main__':

	if len(params) > 0:
		if params[0] == '1':
			init()
			print("Algoritmo inicializado com sucesso !!")
			exit(0)

	train_coll = getSTrain()

	train_df = pd.DataFrame(list(train_coll))

	array_train = train_df['tweet'].values

	target_train = train_df['opiniao'].values

	nb = MultinomialNB()

	if len(params) > 0:
		if params[0] == '2':
			print("Mensurando...")
			train, target_train, test,target_test = split_base(array_train,target_train)
			ac = cross_apply(nb,train,target_train,test,target_test)
			print("Acuracia = %f"%(ac))
			print("Calculo realizado com sucesso !")
			exit(0)

	count_vect = CountVectorizer()

	X_train = count_vect.fit_transform(array_train)

	#nb.fit(X_train,target_train)

	#t = ['estou muito feliz']

	#X_test = count_vect.transform(t)

	#pred = nb.predict(X_test)

	#print(X_train[0])

	#print(pred)

	print("Algoritmo processado com sucesso !!")
	