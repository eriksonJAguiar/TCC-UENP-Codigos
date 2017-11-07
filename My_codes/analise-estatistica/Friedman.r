file <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/predicoes.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
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

value <- c(nv,svm,dt,rf,rl,cm)

k <- length(value)/6

grupo <- c(rep('nv',k),rep('svm',k),rep('dt',k),rep('rf',k),rep('rl',k),rep('cm',k))

m <-  matrix(grupo,value)

f <- friedman.test.