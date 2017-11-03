import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
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

def plot_confuse_matrix(cm,title,file_name):
	labels = ['Negativo', 'Neutro','Positivo']
	cm = np.ceil(cm)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm)
	plt.title(title)
	fig.colorbar(cax)
	ax.set_xticklabels([''] + labels)
	ax.set_yticklabels([''] + labels)

	thresh = cm.max()/2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.xlabel('Predito')
	plt.ylabel('Verdadeiro')
	plt.savefig('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/Figuras/experimentos-final/%s.png'%(file_name))
	#plt.show()

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


def mensure():

	X = read_csv("dataset-portuguese")
	X['opiniao'] = convert_df(X['opiniao'])
	X_text = X['tweet']
	target = np.array(X['opiniao'])

	kf = KFold(10, shuffle=True, random_state=1)

	metricas = pd.DataFrame()
	metricas2 = pd.DataFrame()
	ac_v = []
	cm_v = []
	p_v = []
	r_v = []
	f1_v = []
	e_v = []
	predicts = []

	for train_index,teste_index in kf.split(X_text,target):
		#X_test = X_text[teste_index]
		y_test = target[teste_index]
		
		pred = sent_mensure(X_test)

		predicts += pred.tolist() 
		ac = accuracy_score(y_test, pred)
		p = precision_score(y_test, pred,average='weighted')
		r = recall_score(y_test, pred,average='weighted')
		f1 = (2*p*r)/(p+r)
		e = mean_squared_error(y_test, pred)
		cm = confusion_matrix(y_test,pred)
		cm_v.append(cm)
		ac_v.append(ac)
		p_v.append(p)
		r_v.append(r)
		f1_v.append(f1)
		e_v.append(e)

	metricas['acuracia'] = [statistics.median(ac_v)]
	metricas2['acuracia'] = ac_v
	metricas['precisao'] = [statistics.median(p_v)]
	metricas2['precisao'] = p_v
	metricas['recall'] = [statistics.median(r_v)]
	metricas2['recall'] = r_v
	metricas['f1'] = [statistics.median(f1_v)]
	metricas2['f1'] = f1_v
	metricas['erro'] = [statistics.median(e_v)]
	metricas2['erro'] = e_v
	cm_median = self.matrix_confuse_median(cm_v)

	plot_confuse_matrix(cm_median,'Matriz de Confusao TSViz','TSviz-confuse')
	
	write_csv(metricas,'experimentos-final/metricas-TSviz')
	write_csv(metricas,'experimentos-final/metricas-parciais-TSviz')
	write_csv(predicts,'experimentos-final/TSviz-predicoes')



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

	metricas = []

	metricas.append(accuracy_score(y_true,y_pred))
	p = precision_score(y_true, y_pred,average='weighted')
	metricas.append(p)
	r = recall_score(y_true, y_pred,average='weighted')
	metricas.append(r)
	f1 = (2*p*r)/(p+r)
	metricas.append(f1)
	e = mean_squared_error(y_true, y_pred)
	metricas.append(e)
	cm = confusion_matrix(y_true,y_pred)

	#plot_confuse_matrix(cm,'Matriz de confusao Watson','watson-confuse')

	#write_csv(metricas,'metricas-watson')