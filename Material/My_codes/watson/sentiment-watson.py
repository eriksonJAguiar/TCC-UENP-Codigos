import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score
from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 as Features

def read_csv(file):
	df = pd.DataFrame.from_csv('../files_extern/%s.csv'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

	df = df.reset_index()

	return df
def write_csv(data,file):
		df = pd.DataFrame(data)
		df.to_csv('../files_extern/'+file+'.csv', mode='a', sep=';',index=False, header=False)

def credencials(user,password):
	nlu = NaturalLanguageUnderstandingV1(username=user,password=password,version="2017-02-27")

	return nlu

def get_sent():
	try:
		nlu = credencials('71c0717e-b420-4d34-b5ef-03e68f4826d4','LMcHaglQ6iW3')

		df= read_csv('dataset-portuguese')

		target = df['opiniao'].values
		y_true = []
		y_pred = []
		df_out = pd.DataFrame()
		out = []

		for t in target:
			y_true.append(convert_number(t))

		for txt in df['tweet']:
			out = []
			response = nlu.analyze(
		  	text=txt,
		  	features=[
		  		Features.Sentiment()
		  	],language='pt')
			json_value = json.dumps(response, indent=2)
			json_value = json.loads(json_value)
			sent = json_value['sentiment']['document']['label']
			l = txt,sent
			out.append(l)
			df_out = out
			print('text: %s, sent: %s'%(txt,sent))
			y_pred.append(convert_number(sent))
			write_csv(df_out,'dataset-watson')

	except Exception as inst:
		print(inst)
		pass

def convert_number(target):
	if target == 'negative' or target == 'Negativo':
		return -1
	elif target == 'neutral' or target == 'Neutro':
		return 0
	elif target == 'positive' or target == 'Positivo' :
		return 1

if __name__ == '__main__':

	df = read_csv('dataset-portuguese')
	df_w = read_csv('dataset-watson')

	#tuplas = zip(df['opiniao'].values,df_w['opiniao'].values)

	y_true = []
	y_pred = []

	for t in df['opiniao']:
		y_true.append(convert_number(t))
	
	for p in df_w['opiniao']:
		y_pred.append(convert_number(p))

	
	print(accuracy_score(y_true,y_pred))
