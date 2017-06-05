#!/usr/bin/python
# -*- coding: utf-8 -*-

from TwitterAPI import *
from datetime import *
from pymongo import MongoClient

import sys
import json
import os.path
import time


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

result_max = 10000
result_cont = 0
dh = datetime.now()
tags = ['hiv','aids','viagra','tinder','menopausa','carnaval','dst','sifilis','usecamisinha','hpv','camisinha']

while result_cont < result_max:
	print('Buscando...\n')
	print('Isso Pode Demorar Um Pouco..\n')
	tag_cont = 0
	while tag_cont < len(tags):
		r = twitter.request('search/tweets', {'q': tags[tag_cont], 'lang':'pt-br','locale':'br', 'count':'10000', 'since':'2017-04-02', 'until':'2017-06-03'})
		for item in r.get_iterator():
			#tweet = 'ID: %d, Usuario: %s, texto: %s, Horario: %s, Criado: %s \n'%(item['id'],item['user']['screen_name'],item['text'],dh.now(),item['created_at'])
			#print(tweet)
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
				print(type(inst))
		
		tag_cont += 1
		print(result_cont+" tweets capturados")
				
print('Resultados = %d \n'%(result_cont))
print('Coleta Relalizada com Sucesso! \n')


 







