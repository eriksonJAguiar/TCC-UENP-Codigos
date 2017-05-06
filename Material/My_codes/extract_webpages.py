import requests
import sys
import json
import os.path

from bs4 import BeautifulSoup
from requests_oauthlib import OAuth1

consumer_key = "2qQNn8rY6EPxIAmXCbYWu8xHF"
consumer_secret = "vX22tdAiZRg7wDP4jMf0vP4IL1dzncoTRV05BZiq5xDEzG1J7L"
access_token = "2345718031-eEnUqUP5ZSivgDbnZ15dpeXre1lFCiNsplHbDEV"
access_token_secret = "UQZeDtoet45JKmbdbiXPFwTsweQ2a2MvXf2JPihXeQ55W"


oauth = OAuth1(consumer_key,
                   client_secret=consumer_secret,
                   resource_owner_key=access_token,
                   resource_owner_secret=access_token_secret)

r = requests.get(url="https://twitter.com/search?l=pt&q=iads%20OR%20hiv%20since%3A2013-02-01%20until%3A2017-04-28&src=typd", auth=oauth)

print(r.json())

    except ValueError:
        print("Operacao Invalida")
