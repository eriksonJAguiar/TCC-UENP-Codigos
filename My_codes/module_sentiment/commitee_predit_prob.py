from sent_classification_module import *
from pymongo import MongoClient

##DataBase
client = MongoClient()
db = client.baseTweetsTCC

def getLimit(n):
	return db.tweets.find().limit(n)


if __name__ == '__main__':

	sent = SentClassifiers('dataset-portuguese')

	tweets = getLimit(100)


	for tweet in tweets:
		#print(tweet['text'])
		classe, prob = sent.committee_prob(tweet['text'])
		print("%s, classe = %d prob = %f"%(tweet['text'],classe,prob))
