file <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/My_codes/files_extern/experimentos-final/predicoes.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
watson <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/My_codes/files_extern/experimentos-final/pred-watson.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
microsoft <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/My_codes/files_extern/experimentos-final/pred-microsoft.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)
tsviz <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/My_codes/files_extern/experimentos-final/pred-TSviz.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)

nv <- file$nv
svm <- file$svm
dt <- file$dt
rf <- file$rf
rl <- file$rl
cm <- file$cm


sh_nv <- shapiro.test(nv)
nv_p <- sh_nv$p.value

sh_svm <- shapiro.test(svm)
svm_p <- sh_svm$p.value 

sh_dt <- shapiro.test(dt)
dt_p <- sh_dt$p.value

sh_rf <- shapiro.test(rf)
rf_p <- sh_rf$p.value

sh_rl <- shapiro.test(rl)
rl_p <- sh_rl$p.value

sh_cm <- shapiro.test(cm)
cm_p <- sh_cm$p.value

sh_watson <- shapiro.test(watson$opiniao)
watson_p <- sh_watson$p.value

sh_microsoft <- shapiro.test(microsoft$opiniao)
microsoft_p <- sh_microsoft$p.value

sh_tsviz <- shapiro.test(tsviz$opiniao)
tsviz_p <- sh_tsviz$p.value

shapiroframe <- data.frame('nv' = nv_p,'svm' = svm_p, 'dt' = dt_p, 'rf' = rf_p, 'rl' = rl_p, 'cm' = cm_p, 'watson' = watson_p, 'microsoft' = microsoft_p, 'tsviz' = tsviz_p)


#write.csv(shapiroframe,file='/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/shapiro-wilk.csv')

