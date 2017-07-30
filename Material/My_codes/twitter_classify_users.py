from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import svm
import numpy as np
import pandas as pd
from pymongo import MongoClient
import json


#def update():
#	for u in users:
#	db.usersTwitter.update_one({'_id': u['_id']},{'$set': {'profile_background_color': int(u['profile_background_color'],16)}}, upsert=False)

def date_transform(dates):
	i = 0
	dt = []
	for d in dates:
		year = d.year
		month = d.month
		day = d.day
		hour = d.hour
		minute = d.minute
		second = d.second

		val = year*month*day*hour*minute*second

		dt.append(val)

		i += 1

	return dt

def convertions(dataframe):
	#convert  Dates
	dataframe['created_at'] = pd.to_datetime(dataframe.created_at)

	dataframe['created_at'] = date_transform(dataframe['created_at'])

	#convert location
	enc = LabelEncoder()

	enc.fit(dataframe['location'])

	dataframe['location'] = enc.transform(dataframe['location'])

	return dataframe


if __name__ == '__main__':
	
	##DataBase
	client = MongoClient()
	db = client.baseTweetsTCC

	users = db.usersTwitter.find({},{'sentiment_mean':0,'gender':0, 'language':0,'name':0,'twitter_name':0})

	#for u in users:
	#	if type(u['profile_background_color']) is str:
	#		db.usersTwitter.update_one({'_id': u['_id']},{'$set': {'profile_background_color': int(u['profile_background_color'],16)}}, upsert=False)

	users_df = pd.DataFrame(list(users))

	users_df_cp = users_df.copy()

	users_df_cp = convertions(users_df_cp).astype(int)
	
	u = np.array(users_df_cp)
	
	target = np.array([x for x in users_df['_id']])

	users_df_cp['target'] = k.labels_

	users_df_cp['created_at'] = users_df['created_at']
	users_df_cp['location'] = users_df['location']

	records = json.loads(users_df_cp.T.to_json()).values()

	#print(records)

	db.kmeans.insert(records)

	target = np.array([x for x in users_df['target']])

	train,test,target_train,target_test = train_test_split(u,target,test_size=0.3, random_state=42)

	nv = GaussianNB()
	nv.fit(train,target_train)
	prediction = nv.predict(test)

	ac = accuracy_score(target_test, prediction)

	#print(ac)
	#print(p)


	
