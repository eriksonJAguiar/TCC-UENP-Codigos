#wilcoxon pareado
file <- read.csv('/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/predicoes.csv', header=TRUE, sep=';', stringsAsFactors=FALSE)

nv <- file$nv[0:(length(file$nv)*0.1)]
svm <- file$svm[0:(length(file$svm)*0.1)]
dt <- file$dt[0:(length(file$dt)*0.1)]
rf <- file$rf[0:(length(file$rf)*0.1)]
rl <- file$rl[0:(length(file$rl)*0.1)]
cm <- file$cm[0:(length(file$cm)*0.1)]

value <- c(nv,svm,dt,rf,rl,cm)

#comparando os demais com o comitê
nv_w <- wilcox.test(nv,cm,paired = TRUE)
nv_p <- nv_w$p.value

svm_w <- wilcox.test(svm,cm,paired = TRUE)
svm_p <- svm_w$p.value

dt_w <- wilcox.test(dt,cm,paired = TRUE)
dt_p <- dt_w$p.value

rf_w <- wilcox.test(rf,cm,paired = TRUE)
rf_p <- rf_w$p.value

rl_w <- wilcox.test(rl,cm,paired = TRUE)
rl_p <- rf_w$p.value


wilcoxframe = data.frame('nv' = nv_p,'svm' = svm_p, 'dt' = dt_p, 'rf' = rf_p, 'rl' = rl_p, 'cm' = cm_p)

write.csv(wilcoxframe,file='/media/erikson/BackupLinux/Documentos/UENP/4 º ano/TCC/TCC-UENP-Codigos/Material/My_codes/files_extern/experimentos-final/wilcoxon.csv')


