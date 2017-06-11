########### Python 3.2 #############
import http.client, urllib.request, urllib.parse, urllib.error, base64, requests, json

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '0d01744a65c84cb79550486ebba1fa72',
}

#params = urllib.parse.urlencode({
    # Request parameters
#    'numberOfLanguagesToDetect': "1",
#    'lang':'pt'
#})

body = {
    "documents": [
    {
      "language": "pt",
      "id": "1",
      "text": "olá mundo"
    }
  ]
}

try:
    #conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
    #conn.request("POST" "/text/analytics/v2.0?%s"% headers,params,body)
    #response = conn.getresponse()
    #data = response.read()
    #print(data)
    #conn.close()

    r=requests.post('https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment',data=json.dumps(body),headers=headers)
    text_json = json.loads(r.text)
    score = text_json['documents'][0]['score']
    id = text_json['documents'][0]['id']
    print(score)
    print(id)
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

####################################