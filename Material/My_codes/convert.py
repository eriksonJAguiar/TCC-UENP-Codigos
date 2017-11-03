import matplotlib.pyplot as plt
import distance
import pandas as pd
import numpy as np
import nltk
import re
import itertools
import statistics


def read_csv(file):

		df1 = pd.DataFrame.from_csv('files_extern/%s.csv'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

		df1 = df1.reset_index()

		return df1

def write_csv(data,file):
	df = pd.DataFrame(data)
	#df = df.set_index(['opiniao', 'tweet'])
	df.to_csv('files_extern/%s.csv'%(file), mode='a', sep=';',index=False, header=False)


def convert_df(df):
	new_df = []
	for d in df:
		d = float(d.replace(',','.'))
		if d >= 0.6:
			new_df.append(1)
			
		elif d > 0.4 and d < 0.6:
			new_df.append(0)
			
		elif d <= 0.4:
			new_df.append(-1)
	
	return new_df

def convert_df_(df):
	new_df = []
	for d in df:
		if d == 'Positivo' or d =='positive':
			new_df.append(1)
			
		elif d == 'Neutro' or d =='neutral':
			new_df.append(0)
			
		elif d == 'Negativo' or d == 'negative':
			new_df.append(-1)
	
	return new_df


if __name__ == '__main__':

	X = read_csv("dataset-watson")
	Y = read_csv("dataset-microsoft")
	target_x = convert_df_(X['opiniao'])
	target_y = convert_df(Y['Sentimento_Microsoft'])

	df_watson = pd.DataFrame()
	df_microsoft = pd.DataFrame()
	print(target_x)
	df_watson['opiniao'] = target_x
	df_watson['tweet'] = X['tweet']
	df_microsoft['opiniao'] = target_y
	df_microsoft['tweet'] = Y['Texto']

	write_csv(df_watson,'experimentos-final/pred-watson')
	write_csv(df_microsoft,'experimentos-final/pred-microsoft')

