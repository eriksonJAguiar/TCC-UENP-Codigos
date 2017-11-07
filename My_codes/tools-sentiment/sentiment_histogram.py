import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import math


def getSentimentsMoth(db,mes):
    return db['usersSentiment'].find({'date':{"$regex": u"%s"%(mes)}})

def getCountTweets(db):
	return db['usersSentiment'].find({}).count()

#def getSentimentsDays(db,min,max,mes):
#    datas = []
#    i = min
    
#    while i <= max:
#    	data = db['usersSentiment'].find({'date':{"$regex": u"%s %d "%(mes,i)}})

#    return data

client = MongoClient()
db = client.baseTweetsTCC
sentiments = getSentimentsMoth(db,'Jun')
#sentiments2 = getSentimentsMoth(db,'May')
#sentiments3 = getSentimentsMoth(db,'Jun')
#sentiments = getSentimentsDays(db,20,30,'Apr')
h = list()
n = getCountTweets(db)
#h2 = list()
#h3 = list()
#n = 0
#n2 = 0
#n3 = 0

for sentiment in sentiments:
	h.append(sentiment['sentiment'])

#for sentiment2 in sentiments2:
#	h2.append(sentiment2['sentiment'])
#	n2 += 1

#for sentiment3 in sentiments3:
#	h3.append(sentiment3['sentiment'])
#	n3 += 1

#k = 10 pela regra de sturges

k = math.ceil(1 + 3.32*(math.log(n)))
#k2 = math.ceil(1 + 3.32*(math.log(n2)))
#k3 = math.ceil(1 + 3.32*(math.log(n3)))

plt.hist(h,bins=k,color='blue')
#plt.hist(h2,bins=k2,color='red')
#plt.hist(h3,bins=k3,color='gray')

plt.title('Padrao de Abril',fontsize=16, color='black')
plt.xlabel('Sentimento',fontsize=14, color='black')
plt.ylabel('Num de tweets',fontsize=14, color='black')
plt.show()
