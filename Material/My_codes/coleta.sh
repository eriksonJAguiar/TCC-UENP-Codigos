#!/bin/sh 


while true
do
	if python3.5 /media/erikson/BackupLinux/Documentos/UENP/4\ º\ ano/TCC/TCC/Material/My_codes/twitter_extract_tweets.py $(date +'%Y-%m-%d')
	then
	  	#gravo na base
		while true
		do
	  		if mongodump -d baseTweetsTCC -o /home/erikson/Dropbox/backupMongo
			then
			  break
	  		fi
		done

		break
	fi
done

