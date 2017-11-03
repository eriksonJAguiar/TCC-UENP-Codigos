import nltk
import pandas as pd
import re
from googletrans import Translator
from unicodedata import normalize


def read_csv(file):
    
    df1 = pd.DataFrame.from_csv('files_extern/%s.csv'%(file),sep=';',index_col=0,encoding ='ISO-8859-1')

    df1 = df1.reset_index()

    return df1

def write_csv(data,file):
    df = pd.DataFrame(data)
    df.to_csv('files_extern/'+file+'.csv', mode='w', sep=';',index=False, header=False,encoding='utf8')

def clear(dataframe):
    new_df_tweet = []
    new_df_sent = []
    zipped = zip(dataframe['tweet'],dataframe['opiniao'])
    for (df,opiniao) in zipped:
        expr = re.sub(r"http\S+", "", df)
        #expr = re.sub(r"[@#]\S+","",expr)
        expr = normalize('NFKD',expr).encode('ASCII','ignore').decode('ASCII')
        filtrado = [w for w in nltk.regexp_tokenize(expr.lower(),"[^0-9\W_]+") if not w in nltk.corpus.stopwords.words('portuguese')]
        for f in filtrado:
            if len(f) >= 2:
                #print(f)
                #print(opiniao)
                new_df_tweet.append(f)
                new_df_sent.append(opiniao)

    new_df = pd.DataFrame()

    new_df['tokens'] = new_df_tweet
    new_df['sentimento'] = new_df_sent
        
    return new_df

def convert_df(df):
    new_df = []
    for d in df:
        if d == 'Positivo':
            new_df.append(1)
        elif d == 'Neutro':
            new_df.append(0)
        elif d == 'Negativo':
            new_df.append(-1)
    
    return new_df

def exlusivos(vet_neg,vet_neu,vet_pos):
	ex_pos = []
	ex_neg = []
	ex_neu = []
	tupla = zip(vet_neg,vet_neu,vet_pos)
	for (neg,neu,pos) in tupla:
		if  not (neg in vet_pos or neg in vet_neu):
			ex_neg.append(neg)
		if  not (neu in vet_neg or neu in vet_pos):
			ex_neu.append(neu)
		if not (pos in vet_neg or pos in vet_neu):
			ex_pos.append(pos)

	print(ex_neg)
	print(ex_neu)
	print(ex_pos)

	return ex_neg, ex_neu, ex_pos

def bigram(frases,vet_neg, vet_neu,vet_pos):
    bi_neg = []
    bi_neu = []
    bi_pos = []
    for f in frases:
        if f.find()


if __name__ == '__main__':
    
    df_tweets = read_csv('dataset-portuguese')  
    df_tweets['opiniao'] = convert_df(df_tweets['opiniao'])
    df_words = clear(df_tweets)
    
    neg = df_words.loc[df_words['sentimento'] == -1]
    neu = df_words.loc[df_words['sentimento'] == 0]
    pos = df_words.loc[df_words['sentimento'] == 1]

    neg_freq = nltk.FreqDist(neg['tokens'])
    neu_freq = nltk.FreqDist(neu['tokens'])
    pos_freq = nltk.FreqDist(pos['tokens'])

    vet_neg = []
    vet_neu = []
    vet_pos = []

    #neg_freq.plot(50, cumulative=False)
    #neu_freq.plot(50, cumulative=False)
    #pos_freq.plot(50, cumulative=False)

    #print(neg_freq.most_common(30))
    #print('------------------------')
    #print(neu_freq.most_common(30))
    #print('------------------------')
    #print(pos_freq.most_common(30))

    tupla = zip(neg_freq.most_common(len(neg)),neu_freq.most_common(len(neu)),pos_freq.most_common(len(pos)))

    df_neg = pd.DataFrame()
    df_neu = pd.DataFrame()
    df_pos = pd.DataFrame()

    words_neg = dict()
    words_neu = dict()
    words_pos = dict()

    words_neg['pt'] = []
    words_neg['en'] = []
    words_neg['es'] = []
    words_neu['pt'] = []
    words_neu['en'] = []
    words_neu['es'] = []
    words_pos['pt'] = []
    words_pos['en'] = []
    words_pos['es'] = []

    #neg_freq.plot(30, cumulative=False)
    
    translator = Translator(service_urls=['translate.google.com','translate.google.com.br'])

    for (ng,nu,ps) in tupla:
    	vet_neg.append(ng[0])
    	vet_neu.append(nu[0])
    	vet_pos.append(ps[0])

    vet_neg, vet_neu,vet_pos = exlusivos(vet_neg,vet_neu,vet_pos)

    tupla = zip(vet_neg[:50],vet_neu[:50],vet_pos[:50])

    for (ng,nu,ps) in tupla:
    	words_neg['pt'].append(ng)
    	en=translator.translate(ng, dest='en').text
    	words_neg['en'].append(en)
    	words_neg['es'].append(translator.translate(en, dest='es').text)

    	words_neu['pt'].append(nu)
    	en=translator.translate(nu, dest='en').text
    	words_neu['en'].append(en)
    	words_neu['es'].append(translator.translate(en, dest='es').text)

    	words_pos['pt'].append(ps)
    	en=translator.translate(ps, dest='en').text
    	words_pos['en'].append(en)
    	words_pos['es'].append(translator.translate(en, dest='es').text)

    	

    df_neg['pt'] = words_neg['pt']
    df_neg['en'] = words_neg['en']
    df_neg['es'] = words_neg['es']

    df_neu['pt'] = words_neu['pt']
    df_neu['en'] = words_neu['en']
    df_neu['es'] = words_neu['es']

    df_pos['pt'] = words_pos['pt']
    df_pos['en'] = words_pos['en']
    df_pos['es'] = words_pos['es']

    write_csv(df_neg,'bigram_neg')
    write_csv(df_neu,'bigram_neu')
    write_csv(df_pos,'bigram_pos')

