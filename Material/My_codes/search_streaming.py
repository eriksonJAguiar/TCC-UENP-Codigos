import pandas as pd
import numpy as np
from TwitterAPI import *
import tweepy


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


if __name__ == '__main__':

	tags = []	
	while True:
		trends = api.trends_place(23424768)
		data = trends[0]
		trend = data['trends']
		l = []
		for item in trend:
			name = str(item['name'])
			l.append(name)

		tags.extend(l)
		print(tags)
		print('-------------')

		
