from sklearn.cluster import KMeans
import numpy as np
from pymongo import MongoClient



##DataBase
client = MongoClient()
db = client.baseTweetsTCC

users = db.usersTwitter.find()

#X é um array transformado com o numpy

#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

#for user in users:



	
