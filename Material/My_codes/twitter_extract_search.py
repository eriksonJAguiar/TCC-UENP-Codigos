from TwitterSearch import *
try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_locale('br')
    tso.set_keywords(['hiv', 'aids']) # let's define all words we would like to have a look for
    tso.set_language('pt') # we want to see German tweets only
    #tso.setUntil('2015-07-19')
    #tso.setCount(10000)


    # it's about time to create a TwitterSearch object with our secret tokens

    #Credencias de acesso App Twitter
    consumer_key = "NBL0CtVrn2ajbpaGEWC1GBY2c"
    consumer_secret = "2F5Uz5VYg0ONu4xTYYZsWkAGfc3TYXCkXLCsXMJ1eCKOfhBTfS"
    access_token = "2345718031-we2K2PETQXkz7NCexjdGuvE2L2rnd5KfouzN3Up"
    access_token_secret = "aEQPKGifu1y29Wbh3u6Z0YIcjAsBC8VeD4Y75CDL2r12o"

    ts = TwitterSearch(consumer_key, consumer_secret,access_token,access_token_secret)

     # this is where the fun actually starts :)
    for tweet in ts.search_tweets_iterable(tso):
        print( '@%s tweeted: %s criado: %s' % ( tweet['user']['screen_name'], tweet['text'], tweet['created_at'] ) )
        
except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)