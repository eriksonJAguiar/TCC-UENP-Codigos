file <- read.table('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/My_codes/files_extern/experimentos-final/predicoes.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
watson <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/pred-watson.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
microsoft <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/pred-microsoft.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
tsviz <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/pred-TSviz.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)


nv <- file$nv
svm <- file$svm
dt <- file$dt
rf <- file$rf
rl <- file$rl
cm <- file$cm


#value <- c(nv,svm,dt,rf,rl,cm,watson$opiniao,microsoft$opiniao,tsviz$opiniao)

value <-c(cm,rf,rl,svm,nv,dt,watson$opiniao,microsoft$opiniao) #c(dt,nv,svm,rl,rf,cm)

#grupo <- c(rep('cm',k),rep('rf',k),rep('rl',k),rep('svm',k),rep('nv',k),rep('dt',k))

n <- 8
k <- length(value)/n
len <- length(value)

z <- gl(n,k,len,labels = c("cm","rf","rl","svm","nv","dt","watson","t. analytics"))

m <- matrix(value,
       nrow = k,
       ncol=n,
       byrow = TRUE,
       dimnames = list(1 : k,c("cm","rf","rl","svm","nv","dt","watson","t. analytics") )) #c("dt","nv","svm","rl","rf","cm")


f <- friedman.test(m) 

fp <- posthoc.friedman.nemenyi.test(m)

nt <- NemenyiTest(value,z, out.list = TRUE)




