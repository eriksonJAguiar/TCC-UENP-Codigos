# Install and Activate Packages
library(twitteR)
library(RCurl)
library(RJSONIO)
library(stringr)

#Credencias de acesso App Twitter
consumer_key <- "2qQNn8rY6EPxIAmXCbYWu8xHF"
consumer_secret <- "vX22tdAiZRg7wDP4jMf0vP4IL1dzncoTRV05BZiq5xDEzG1J7L"
access_token <- "2345718031-eEnUqUP5ZSivgDbnZ15dpeXre1lFCiNsplHbDEV"
access_token_secret <- "UQZeDtoet45JKmbdbiXPFwTsweQ2a2MvXf2JPihXeQ55W"

# Create Twitter Connection
setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_token_secret)


# Use the searchTwitter function to only get tweets within 50 miles of Los Angeles
#tweets_geolocated <- searchTwitter("hiv",n = 9999, lang="pt", since="2014-08-20", until="2017-04-26", resultType = 'popular')
#tweets_geolocated.df <- twListToDF(tweets_geolocated)



t<-searchTwitter('hiv', lang = 'pt', n=100000, since="2016-02-20",until="2017-04-28", retryOnRateLimit = 1)

list <- twListToDF(t)

