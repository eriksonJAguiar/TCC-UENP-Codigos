library(ROCR)

FPR<-read.csv2("files_extern/roc.csv", sep=",", dec = ".")[1:30,2:7]
TPR<-read.csv2("files_extern/roc.csv", sep=",",dec = ".")[31:60,2:7]

ROCR::plot(x=FPR[,1], y=TPR[,1], type="l", ylim=c(0,1), xlim=c(0,1), ylab="Taxa de Verdadeiro Positivo", xlab="Taxa de Falso Positivo")
par(new=TRUE)
ROCR::plot(x=FPR[,2], y=TPR[,2], type="l", ylim=c(0,1), xlim=c(0,1), col=2, axes=FALSE, ann=FALSE)
par(new=TRUE)
ROCR::plot(x=FPR[,3], y=TPR[,3], type="l", ylim=c(0,1), xlim=c(0,1), col=3, axes=FALSE, ann=FALSE)
par(new=TRUE)
ROCR::plot(x=FPR[,4], y=TPR[,4], type="l", ylim=c(0,1), xlim=c(0,1), col=4, axes=FALSE, ann=FALSE)
par(new=TRUE)
ROCR::plot(x=FPR[,5], y=TPR[,5], type="l", ylim=c(0,1), xlim=c(0,1), col=5, axes=FALSE, ann=FALSE)
par(new=TRUE)
ROCR::plot(x=FPR[,6], y=TPR[,6], type="l", ylim=c(0,1), xlim=c(0,1), col=6, axes=FALSE, ann=FALSE)
