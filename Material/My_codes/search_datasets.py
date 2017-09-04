import pandas as pd
import numpy as np
from TwitterAPI import *


#Credencias de acesso App Twitter
consumer_key = "NBL0CtVrn2ajbpaGEWC1GBY2c"
consumer_secret = "2F5Uz5VYg0ONu4xTYYZsWkAGfc3TYXCkXLCsXMJ1eCKOfhBTfS"
access_token = "2345718031-we2K2PETQXkz7NCexjdGuvE2L2rnd5KfouzN3Up"
access_token_secret = "aEQPKGifu1y29Wbh3u6Z0YIcjAsBC8VeD4Y75CDL2r12o"

#acessa OAuth
# Referencia para API: https://dev.twitter.com/rest/reference
twitter = TwitterAPI(consumer_key, consumer_secret,auth_type='oAuth2')


def read_csv(file):
		df1 = pd.DataFrame.from_csv('files_extern/original-datasets/%s.csv'%(file),sep=',',index_col=0,encoding ='ISO-8859-1')

		df1 = df1.reset_index()

		return df1

def write_csv(data,file):
	df = pd.DataFrame(data)
	df.to_csv('files_extern/%s.csv'%(file), mode='a', sep=';',index=False, header=False)


if __name__ == '__main__':

	df = read_csv('English_Twitter_sentiment')

	dados = []

	for i in range(3000):
		try:
			tweet = twitter.request('statuses/show/:%d' % df['TweetID'][i])
			text = ''
			sent = df['HandLabel'][i]
			for item in tweet:
				text = item['text']
				print(text)

			line = text,sent

			dados.append(line)
		except Exception as inst:
			pass

	write_csv(dados,'dataset-eglish')


		
