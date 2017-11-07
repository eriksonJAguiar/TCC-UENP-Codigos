import pandas as pd
import numpy as np
import tweepy
import time
from datetime import datetime
import csv
from unicodedata import normalize

#Credencias de acesso App Twitter
consumer_key = "NBL0CtVrn2ajbpaGEWC1GBY2c"
consumer_secret = "2F5Uz5VYg0ONu4xTYYZsWkAGfc3TYXCkXLCsXMJ1eCKOfhBTfS"
access_token = "2345718031-we2K2PETQXkz7NCexjdGuvE2L2rnd5KfouzN3Up"
access_token_secret = "aEQPKGifu1y29Wbh3u6Z0YIcjAsBC8VeD4Y75CDL2r12o"

#acessa OAuth
# Referencia para API: https://dev.twitter.com/rest/reference
authentication = tweepy.OAuthHandler(consumer_key, consumer_secret)
authentication.set_access_token(access_token, access_token_secret)
api = tweepy.API(authentication)

def write_file(datas,filename):
	with open('%s.csv'%(filename), 'a', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=';')
		for row in datas:
			spamwriter.writerow(row)

def acents(text):
	return normalize('NFKD',text).encode('ASCII','ignore').decode('ASCII')


if __name__ == '__main__':

	log = []	
	while True:
		tags = []
		line = []
		trends = api.trends_place(23424768)
		data = trends[0]
		trend = data['trends']
		for item in trend:
			name = str(item['name'])
			name = acents(name)
			if name not in log:
				l = name,str(datetime.now())
				line.append(l)
				tags.append(name)

		log.extend(tags)
		print(log)
		print(len(log))
		write_file(line,'tags')
		print('-------------')
		time.sleep(60)

		
