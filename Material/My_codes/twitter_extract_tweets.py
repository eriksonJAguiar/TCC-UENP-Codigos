#!/usr/bin/python
# -*- coding: utf-8 -*-

from TwitterAPI import *
from datetime import *
from pymongo import MongoClient

import sys
import json
import os.path
import time


#timeout
timeout = 60
timeout_start = time.time()

#Credencias de acesso App Twitter
consumer_key = "NBL0CtVrn2ajbpaGEWC1GBY2c"
consumer_secret = "2F5Uz5VYg0ONu4xTYYZsWkAGfc3TYXCkXLCsXMJ1eCKOfhBTfS"
access_token = "2345718031-we2K2PETQXkz7NCexjdGuvE2L2rnd5KfouzN3Up"
access_token_secret = "aEQPKGifu1y29Wbh3u6Z0YIcjAsBC8VeD4Y75CDL2r12o"

#acessa OAuth
# Referencia para API: https://dev.twitter.com/rest/reference
twitter = TwitterAPI(consumer_key, consumer_secret,auth_type='oAuth2')


##DataBase 

client = MongoClient()
db = client.baseTweetsTCC

def saveTrends(tag,date):
	try:
		db.trends.insert_one(
						{
							'tag':tag,
							'date':date
						}
					)
	except Exception as inst:
		pass

result_max = 10000
result_cont = 0
dh = datetime.now()
#tags = ['hiv','aids','viagra','tinder','menopausa','dst','ist','sifilis','usecamisinha','hpv','camisinha']
tags = []
#param = sys.argv[1:]
#print(param[0])

trends_br = twitter.request('trends/place', {'id':	23424768})
trends_eua = twitter.request('trends/place', {'id':	23424977})
trends_eng = twitter.request('trends/place', {'id': 24554868})

n_trends = 10

i = 0

for br in trends_br.get_iterator():
	tags.append(br['name'])
	saveTrends(br['name'],dh.now())
	i += 1
	if i > n_trends: break

i = 0
for eua in trends_eua.get_iterator():
	tags.append(eua['name'])
	saveTrends(eua['name'],dh.now())
	if i > n_trends: break
	i += 1

i = 0
for eng in trends_eua.get_iterator():
	tags.append(eng['name'])
	saveTrends(eng['name'],dh.now())
	if i > n_trends: break
	i += 1

#print(tags)


while result_cont < result_max:
	#print('Buscando...\n')
	#print('Isso Pode Demorar Um Pouco..\n')
	tag_cont = 0
	while tag_cont < len(tags):
		r = twitter.request('search/tweets', {'q': tags[tag_cont]})
		for item in r.get_iterator():
			#tweet = 'ID: %d, Usuario: %s, texto: %s, Horario: %s, Criado: %s \n'%(item['id'],item['user']['screen_name'],item['text'],dh.now(),item['created_at'])
			#print(item['text'])
			try:
				db.tweets.insert_one(
					{
						'_id':item['id'],
						'id_user':item['user']['id'],
						'name':item['user']['screen_name'],
						'text':item['text'],
						'hourGet':dh.now(),
						'created_at':item['created_at'],
						'location':item['user']['location'],
						'retweets_count':item['retweet_count']
					}
				)
			
				result_cont += 1
			except Exception as inst:
				#print(type(inst))
				pass
		
		tag_cont += 1
		#print("%d tweets capturados"%result_cont)

	if time.time() >= timeout_start + timeout:
		break
				
#print('Resultados = %d \n'%(result_cont))
#print('Coleta Relalizada com Sucesso! \n')

 







