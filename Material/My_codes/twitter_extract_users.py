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
consumer_key = "2qQNn8rY6EPxIAmXCbYWu8xHF"
consumer_secret = "vX22tdAiZRg7wDP4jMf0vP4IL1dzncoTRV05BZiq5xDEzG1J7L"
access_token = "2345718031-eEnUqUP5ZSivgDbnZ15dpeXre1lFCiNsplHbDEV"
access_token_secret = "UQZeDtoet45JKmbdbiXPFwTsweQ2a2MvXf2JPihXeQ55W"

#acessa OAuth
# Referencia para API: https://dev.twitter.com/rest/reference
twitter = TwitterAPI(consumer_key, consumer_secret,auth_type='oAuth2')


##DataBase 

client = MongoClient()
db = client.baseTweetsTCC

tweets = db.find()

for document in tweets:
	print('Buscando...\n')
	print('Isso Pode Demorar Um Pouco..\n')
	while tag_cont < len(tags):
		r = twitter.request('users/show', {'user_id':document['id_user'] , 'screen_name':document['name']})
		for item in r.get_iterator():
			user = 'ID: %d, Usuario: %s, location: %s, Horario: %s, Perfil_criado: %s \n'%(item['id'],item['user']['screen_name'],item['text'],dh.now(),item['created_at'])
			print(user)
			#try:
			#	db.tweets.insert_one(
			#		{
			#			'_id':item['id'],
			#			'id_user':item['user']['id'],
			#			'name':item['user']['screen_name'],
			#			'text':item['text'],
			#			'hourGet':dh.now(),
			#			'created_at':item['created_at'],
			#			'location':item['user']['location'],
			#			'retweets_count':item['retweet_count']
			#		}
			#	)
			#except Exception as inst:
			#	print(type(inst))

		result_cont += 1
			
	print('Resultados = %d \n'%(result_cont))

	#if result_cont > 0:
	#	print('aguarde...')
	#	time.sleep(60)

	print('Coleta Relalizada com Sucesso! \n')


 







