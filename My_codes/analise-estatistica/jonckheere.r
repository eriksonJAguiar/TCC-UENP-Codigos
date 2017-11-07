file <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/predicoes-testes.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
watson <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/pred-watson.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
microsoft <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/pred-microsoft.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
tsviz <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/pred-TSviz.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)

#watson$opiniao,microsoft$opiniao,tsviz$opiniao


#k <- length(value)/6

# 1 - nv, 2 - smv, 3 - dt, rf - 4, rl - 5 e cm - 6

#grupoj <- c(rep(1,k),rep(2,k),rep(3,k),rep(4,k),rep(5,k),rep(6,k))

j <- jonckheere.test(file$valor, file$grupo)