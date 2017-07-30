from pymongo import MongoClient


def AcessaBd(dataBase,collection):
	try:
		client = MongoClient()

		db = client[dataBase]

		result = db[collection]

		return result

	except Exception as inst:
		print(type(inst))

def getAll(collection):
	return collection.find()

def pass_to_txt(document):
	f = open('hastags.txt', 'w')
	for t in document:
		f.write('%s\n'%(t))
	
	f.close()

def getLimit(coll,n):
	return coll.find().sort('text',1).limit(n)

def getHastags(documents):
	tags = ['hiv','aids','viagra','tinder','menopausa','dst','ist','sifilis','usecamisinha','hpv','camisinha']
	hastags = list()
	for d in documents:
		text = d['text'].split(' ')
		for t in text:
			if len(t) > 0:
				if t[0] == '#':
					if not(t.split('#')[1] in tags):
						if not(t in hastags):
							hastags.append(t)

	return hastags


if __name__ == '__main__':

	print('Buscando...\n')
	print('Isso Pode Demorar um Pouco !')

	colllection = AcessaBd('baseTweetsTCC','tweetsProcessing1')

	tweets = getAll(colllection)
	
	hashtags = getHastags(tweets)

	pass_to_txt(hashtags)

	print("Texto processado com sucesso")



