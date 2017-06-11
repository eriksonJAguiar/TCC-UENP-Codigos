from sklearn.cluster import KMeans
import numpy as np
from pymongo import MongoClient



##DataBase s
client = MongoClient()
db = client.baseTweetsTCC

users = db.usersTwitter.find()

for user in users:

	
