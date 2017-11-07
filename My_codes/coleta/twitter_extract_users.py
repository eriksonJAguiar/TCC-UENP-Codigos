#!/usr/bin/python
# -*- coding: utf-8 -*-

from TwitterAPI import *
from pymongo import MongoClient
#from genderize import Genderize

import sys
import json
import os.path
import time
import requests



#Credencias de acesso App Twitter
consumer_key = "NBL0CtVrn2ajbpaGEWC1GBY2c"
consumer_secret = "2F5Uz5VYg0ONu4xTYYZsWkAGfc3TYXCkXLCsXMJ1eCKOfhBTfS"
access_token = "2345718031-we2K2PETQXkz7NCexjdGuvE2L2rnd5KfouzN3Up"
access_token_secret = "aEQPKGifu1y29Wbh3u6Z0YIcjAsBC8VeD4Y75CDL2r12o"


#acessa OAuth
# Referencia para API: https://dev.twitter.com/rest/reference
twitter = TwitterAPI(consumer_key, consumer_secret,auth_type='oAuth2')


##DataBase s
client = MongoClient()
db = client.baseTweetsTCC

tweets = db.tweets.find()

users_count = 0

#print('Buscando...\n')
#print('Isso Pode Demorar Um Pouco..\n')
it = 1

for document in tweets:
	try:
		r = twitter.request('users/show', {'user_id':document['id_user'] , 'screen_name':document['name']})
		for item in r.get_iterator():

			db.usersTwitter.insert_one(
				{
					'_id':item['id'],
					'twitter_name':item['screen_name'],
					'name':item['name'],
					'gender': 'NaN',
					#'age': 'NaN',
					'location':item['location'],
					'friends_number':str(item['friends_count']),
					'followers_number':str(item['followers_count']),
					'listed_number': str(item['listed_count']),
					'statuses_count':str(item['statuses_count']),
					'language':item['lang'],
					'profile_background_color':str(item['profile_background_color']),
					#'profile_background_image_url':item['profile_background_image_url'],
					#'profile_image_url': item['profile_image_url'],
					'created_at':str(item['created_at']),
					'sentiment_mean':'NaN'
				}
			)

			users_count += 1


			#print('Numero de Usuarios = %d \n'%(users_count))
			
	except Exception as inst:
			#print(type(inst))
			pass
	
print('Numeros de usuarios coletados = %d'%(users_count))
print('Coleta Relalizada com Sucesso! \n')


 







