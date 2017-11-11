#Calcular o teste de Wilcoxon:
  
file <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/My_codes/files_extern/experimentos-final/predicoes.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
#watson <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/pred-watson.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
#microsoft <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/pred-microsoft.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
#tsviz <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/pred-TSviz.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)


nv <- file$nv
svm <- file$svm
dt <- file$dt
rf <- file$rf
rl <- file$rl
cm <- file$cm

#value <- c(nv,svm,dt,rf,rl,cm,watson$opiniao,microsoft$opiniao,tsviz$opiniao)

value <-c(cm,rf,rl,svm,nv,dt)

#n <- 9
#k <- length(value)/9
#len <- length(value)

n <- 6
k <- length(value)/6
len <- length(value)

#Criar os níveis:
#1º específicais quantos e as ordens. Exemplo:
 # z <- gl(5, 30, 150)
#z <- gl(n,k,len,labels = c("nv","svm","dt","rf","rl","cm","watson","microsoft","tsviz"))
z <- gl(n,k,len,labels = c("cm","rf","rl","svm","nv","dt"))
#resulta em números de 1 até 5, repetidos 30 vezes cada, resultando em um total de 150.
#z <- gl(X, Y, Z) onde, X é quantidade de níveis; Y quantas vezes devem ser repetidos
#cada um antes do próximo; Z o total de ocorrencias

#2º Definir as legendas de cada nível. ATENÇÃO: A quatidade de legendas deve ser igual a
#quantidade de específicadas em X no comando anteior. Exemplo:
  #levels(z) <- (c("Part3Ite20", "Part5Ite20", "Part10Ite20", "Part15Ite20", "Part20Ite20"))
  #levels(z) <- (c("nv","svm","dt","rf","rl","cm"))
# Oranizar os dados:
#3º Os dados devem ser todos organizados em um vetor na ordem de suas expecíficas legendas, 
#definidos anteriormente. Exemplo:
  #values <- c(Resultados_Otimizacao$Part3Ite20, Resultados_Otimizacao$Part5Ite20, Resultados_Otimizacao$Part10Ite20, Resultados_Otimizacao$Part15Ite20, Resultados_Otimizacao$Part20Ite20)
#Devem ficar organizados em um vetor de maneira simples. 

#Realizar o calculo
#4º Realizar o Teste de Wilcoxon pareado:
  #w <- pairwise.wilcox.test(value, z)
  #p_values <- w$p.value
#Caso os valores sejam muito extensos pode-se usar o parâmetro "exact = FALSE" para evitar erros
#Exemplo:
 w <- pairwise.wilcox.test(value, z, exact = FALSE)
 p_values <- w$p.value