#!/usr/bin/python
# -*- coding: utf-8 -*-

from TwitterAPI import *
from pymongo import MongoClient
from genderize import Genderize

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

tweets = db.tweets.find()

users_count = 0

for document in tweets:
	print('Buscando...\n')
	print('Isso Pode Demorar Um Pouco..\n')
	try:
		r = twitter.request('users/show', {'user_id':document['id_user'] , 'screen_name':document['name']})
		for item in r.get_iterator():
			#gender = Genderize().get(['nome'])

			db.usersTwitter.insert_one(
				{
					'_id':item['id'],
					'id_user':item['user']['id'],
					'name':item['user']['screen_name'],
					#'gender': gender,
					#'age': 'idade',
					'location':item['location'],
					'friends_number':item['friends_count'],
					'created_at':item['created_at']
				}
			)

			users_count += 1

	except Exception as inst:
			print(type(inst))
			
	print('Numero de Usuarios = %d \n'%(users_count))
	
	print('Coleta Relalizada com Sucesso! \n')


 







