from TwitterSearch import *
try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_locale('br')
    tso.set_keywords(['hiv', 'aids']) # let's define all words we would like to have a look for
    tso.set_language('pt') # we want to see German tweets only
    tso.setUntil('2015-07-19')
    tso.setCount(10000)


    # it's about time to create a TwitterSearch object with our secret tokens

    consumer_key = "2qQNn8rY6EPxIAmXCbYWu8xHF"
    consumer_secret = "vX22tdAiZRg7wDP4jMf0vP4IL1dzncoTRV05BZiq5xDEzG1J7L"
    access_token = "2345718031-eEnUqUP5ZSivgDbnZ15dpeXre1lFCiNsplHbDEV"
    access_token_secret = "UQZeDtoet45JKmbdbiXPFwTsweQ2a2MvXf2JPihXeQ55W"

    ts = TwitterSearch(consumer_key, consumer_secret,access_token,access_token_secret)

     # this is where the fun actually starts :)
    for tweet in ts.search_tweets_iterable(tso):
        print( '@%s tweeted: %s criado: %s' % ( tweet['user']['screen_name'], tweet['text'], tweet['created_at'] ) )

except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)