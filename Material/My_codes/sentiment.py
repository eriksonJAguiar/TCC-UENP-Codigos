import http.client, urllib.request, urllib.parse, urllib.error, base64, requests, json
from pymongo import MongoClient
import time




def getAllUsers(db):
    return db['usersTwitter'].find()

def getUserTweets(db,id):
    return db['tweetsProcessing1'].find({'id_user':id})

def getAllTweets(db,id):
    return db['tweetsProcessing1'].find({'_id':id})

def insertSentiment(db,id,score):
    db['usersSentiment'].insert_one(
                {
                    #'_id':index,
                    'id_user':id,
                    'sentiment': score,
                    #'date': date
                }
            )

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '0d01744a65c84cb79550486ebba1fa72',
}


client = MongoClient()
db = client.baseTweetsTCC
users = getAllUsers(db)

i = 0
count = 0

print("Iniciando o algoritmo...\n")
print("Isso pode demorar um pouco...\n")

val = []

for user in users:

        tweets = getUserTweets(db,user['_id'])

        for tweet in tweets:
            
            val.append({ "language": "pt","id": tweet['_id'],"text": tweet['text'] })
            
            i += 1
            count += 1

            if i > 900:  
                try:
                    body = {
                        "documents": val
                        #[
                        #    {
                        #      "language": "pt",
                        #      "id": tweet['_id'],
                        #      "text": tweet['text']
                        #    } 
                        #]    
                    }
                    r=requests.post('https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment',data=json.dumps(body),headers=headers)
                    text_json = json.loads(r.text)
                    #score = text_json['documents'][0]['score']
                    #id = text_json['documents'][0]['id']

                    #print(text_json)
                        
                    #insertSentiment(db,i+1,id,score,tweet['date'])

                        
                    for text in text_json['documents']:
                        id = text['id']
                        score = text['score']
                        insertSentiment(db,id,score)

                    
                    i = 0

                except Exception as inst:
                    #print("Ocorreu um erro no processamento")
                    print("[Errno {0}] {1}".format(e.errno, e.strerror))
                    #print(type(inst))

print("Processamento Concluido com sucesso")
