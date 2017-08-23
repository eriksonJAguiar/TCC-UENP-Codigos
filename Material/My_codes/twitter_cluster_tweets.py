from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from pymongo import MongoClient
import json


client = MongoClient()
db = client.baseTweetsTCC

def getBase():

	tweets = db['tweetsProcessing1'].find({})

	return tweets



if __name__ == '__main__':
	
	print("Iniciando o agrupamento, aguarde...")

	tweets = getBase()


	tweets_df = pd.DataFrame(list(tweets))

	text = tweets_df['text']

	t_df = pd.DataFrame()
	t_df['text'] = text

	count_vect = CountVectorizer()
	X = count_vect.fit_transform(text)	

	#som = SOM(input_data)
	
	#som.set_parameter(neighbor=0.1, learning_rate=0.2)

	#output_map = som.train(len(X))

	#plt.imshow(output_map,interpolation='gaussian')
	#plt.show()

	km = KMeans(n_clusters=10)

	k = km.fit(X)

	t_df['target'] = k.labels_

	records = json.loads(t_df.T.to_json()).values()

	db.groups_texts.insert(records)

	print("Grupos processados com sucesso")

	
