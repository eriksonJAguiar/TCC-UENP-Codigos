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


##DataBase s
client = MongoClient()
db = client.baseTweetsTCC

tweets = db.tweets.find()

users_count = 0

print('Buscando...\n')
print('Isso Pode Demorar Um Pouco..\n')

for document in tweets:
	try:
		r = twitter.request('users/show', {'user_id':document['id_user'] , 'screen_name':document['name']})
		for item in r.get_iterator():
			
			name = item['name'].split(' ')
			gender = Genderize().get([name[0]])

			db.usersTwitter.insert_one(
				{
					'_id':item['id'],
					'twitter_name':item['screen_name'],
					'name':item['name'],
					'gender': gender[0]['gender'],
					'location':item['location'],
					'friends_number':item['friends_count'],
					'followers_number':item['followers_count'],
					'listed_number': item['listed_count'],
					'statuses_count':item['statuses_count'],
					'language':item['lang'],
					'profile_background_color':item['profile_background_color'],
					'profile_background_image_url':item['profile_background_image_url'],
					'profile_image_url': item['profile_image_url'],
					'created_at':item['created_at']
				}
			)

			users_count += 1

	except Exception as inst:
			print(type(inst))
			
print('Numero de Usuarios = %d \n'%(users_count))
	
print('Coleta Relalizada com Sucesso! \n')


 







