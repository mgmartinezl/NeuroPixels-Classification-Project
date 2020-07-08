# Correlation matrix for features

library(dplyr)
library(ggbiplot)
library(rgl)
library(RColorBrewer)
library(scales)
library(pvclust)
library(factoextra)
library(fpc)
library(NbClust)
library(Hmisc)

data = read.csv("Neuropixels_Features.csv")

df <- select(data,
             MFR = MFRBlockHz,
             MIFR = tf_MIFRBlockHz,
             Med_Isi = tf_MedIsi,
             Mode_Isi = tf_ModeIsi,
             Perc5_Isi = tf_Perc5Isi,
             Entropy = tf_Entropy,
             CV2_Mean = tf_CV2Mean,
             CV2_Median = tf_CV2Median,
             CV = tf_CV,
             Ir = tf_Ir,
             Lv = tf_Lv,
             LvR = tf_LvR,
             LcV = tf_LcV,
             Si = tf_Si,
             skw = tf_skw,
             MaxAmp = wf_MaxAmp,
             MaxAmp_Time = wf_MaxAmpTime,
             MinAmp = wf_MinAmp,
             MinAmp_Time = wf_MinAmpTime,
             Duration = wf_Duration,
             PosHw_Duration = wf_PosHwDuration,
             NegHw_Duration = wf_NegHwDuration,
             Onset = wf_Onset,
             Onset_Time = wf_OnsetTime,
             End = wf_End,
             End_Time = wf_EndTime,
             PkTr_Ratio = wf_PkTrRatio,
             Depolarization_Slope = wf_DepolarizationSlope,
             Repolarization_Slope = wf_RepolarizationSlope,
             Recovery_Slope = wf_RecoverySlope,
             Rise_Time = wf_RiseTime,
             Pos_Decay_Time = wf_PosDecayTime,
             Fall_Time = wf_FallTime,
             Neg_Decay_Time = wf_NegDecayTime,
             End_Slope_Tau = wf_EndSlopeTau)


res <- cor(df, method = "pearson", use = "complete.obs")
res2 <- rcorr(as.matrix(df), type=c("pearson"))
res2

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

flattenCorrMatrix(res2$r, res2$P)

library(corrplot)
# corrplot(res, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)

corrplot(res, method = "color", tl.srt = 90, tl.col = "black",  type = "upper",
         cl.cex = 0.5, tl.cex = 0.55)

df2 = read.csv("WardClustering_ACGs_and_OtherFeatures_5Classes_Comparison.csv")

#qplot(df2$wf_Duration, df2$SmallCluster, color=df2$SmallCluster)

library(grDevices)

# c1 <- rainbow(10)
c1 <- c("#FF62BC", "#C77CFF", "#00A9FF", "#E7B800", "#00AFBB", "#FC4E07")
c2 <- c("#FF62BC33", "#C77CFF33", "#00A9FF33", "#E7B80033", "#00AFBB33", "#FC4E0733")
# c2 <- rainbow(10, alpha=0.2)
# c3 <- rainbow(10, v=0.7)

# adjustcolor("#00AFBB", alpha.f = 0.2)

df2$Cluster2 <- factor(df2$Cluster , levels=c("Pk", "CS", "MLI", "Gr", "GoMF", "Unknown"))

boxplot(df2$MFRBlockHz ~ df2$Cluster2,
        col=c2, medcol=c1, 
        whiskcol=c1, 
        staplecol=c1, 
        boxcol=c1, 
        outcol=c1, 
        pch=16, cex=0.5,
        xlab='Cluster', ylab='Mean Firing Rate', frame=FALSE) + theme_minimal() 

boxplot(df2$tf_skw ~ df2$Cluster2,
        col=c2, medcol=c1, 
        whiskcol=c1, 
        staplecol=c1, 
        boxcol=c1, 
        outcol=c1, 
        pch=16, cex=0.5,
        xlab='Cluster', ylab='Skewness', frame=FALSE) + theme_minimal() 


library(ggpubr)
ggboxplot(df2, x = "Cluster2", y = "MFRBlockHz", color = "Cluster2", palette = c1, fill = c2,
          font.label = list(size = 12, face = "plain", color ="red"), frame=FALSE) + 
  stat_compare_means() + theme(legend.position="none")

#sp2<-ggplot(df2, aes(x=MFRBlockHz, y=tf_Entropy, color=SmallCluster)) + geom_point()
#sp2+scale_color_gradient(low="blue", high="red")
#sp2+scale_color_gradient(colours = rainbow(6))

ggplot(df2, aes(x=tf_Entropy, y=MFRBlockHz, color=SmallClusterName)) + 
  geom_point(alpha=1,aes(shape = SmallClusterName, color = SmallClusterName), size = 2) + 
  xlab("Entropy") + ylab("MFR") + theme(legend.position="top")  + theme_minimal()

ggplot(df2, aes(x=tf_CV2Mean, y=MFRBlockHz, color=SmallClusterName)) + 
  geom_point(alpha=1,aes(shape = SmallClusterName, color = SmallClusterName), size = 2) + 
  xlab("Mean Cv2") + ylab("MFR") + theme(legend.position="top")  + theme_minimal()

ggplot(df2, aes(x=tf_CV2Mean, y=tf_Entropy, color=SmallClusterName)) + 
  geom_point(alpha=1,aes(shape = SmallClusterName, color = SmallClusterName), size = 2) + 
  xlab("Mean Cv2") + ylab("Entropy") + theme(legend.position="top")  + theme_minimal()

ggplot(df2, aes(x=tf_CV, y=tf_Entropy, color=SmallClusterName)) + 
  geom_point(alpha=1,aes(shape = SmallClusterName, color = SmallClusterName), size = 2) + 
  xlab("Cv") + ylab("Entropy") + theme(legend.position="top")  + theme_minimal()

ggplot(df2, aes(x=tf_CV, y=MFRBlockHz, color=SmallClusterName)) + 
  geom_point(alpha=1,aes(shape = SmallClusterName, color = SmallClusterName), size = 2) + 
  xlab("Cv") + ylab("MFR") + theme(legend.position="top")  + theme_minimal()

ggplot(df2, aes(x=tf_Entropy, y=wf_Duration, color=SmallClusterName)) + 
  geom_point(alpha=1,aes(shape = SmallClusterName, color = SmallClusterName), size = 2) + 
  xlab("Entropy") + ylab("Duration") + theme(legend.position="top")  + theme_minimal()

ggplot(df2, aes(x=MFRBlockHz, y=wf_Duration, color=SmallClusterName)) + 
  geom_point(alpha=1,aes(shape = SmallClusterName, color = SmallClusterName), size = 2) + 
  xlab("MFR") + ylab("Duration") + theme(legend.position="top")  + theme_minimal()


library(ggplot2)
ggplot(df2, aes(x=SmallClusterName, y=MFRBlockHz, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") 

ggplot(df2, aes(x=SmallClusterName, y=wf_Duration, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Duration")

ggplot(df2, aes(x=SmallClusterName, y=tf_MedIsi, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Median Isi")

ggplot(df2, aes(x=SmallClusterName, y=tf_CV2Mean, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Mean CV2")

ggplot(df2, aes(x=SmallClusterName, y=tf_LvR, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("LvR")

ggplot(df2, aes(x=SmallClusterName, y=wf_RiseTime, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Rise time")

ggplot(df2, aes(x=SmallClusterName, y=wf_PosDecayTime, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Positive decay time")

ggplot(df2, aes(x=SmallClusterName, y=wf_NegDecayTime, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Negative decay time")

ggplot(df2, aes(x=SmallClusterName, y=wf_FallTime, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Fall time")

ggplot(df2, aes(x=SmallClusterName, y=wf_PosHwDuration, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Positive half-width duration")

ggplot(df2, aes(x=SmallClusterName, y=wf_NegHwDuration, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Negative half-width duration")

ggplot(df2, aes(x=SmallClusterName, y=wf_PkTrRatio, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Peak-trough ratio")

ggplot(df2, aes(x=SmallClusterName, y=wf_DepolarizationSlope, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Depolarization slope")

ggplot(df2, aes(x=SmallClusterName, y=wf_RepolarizationSlope, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Repolarization slope")

ggplot(df2, aes(x=SmallClusterName, y=wf_RecoverySlope, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Recovery slope")

ggplot(df2, aes(x=SmallClusterName, y=wf_Onset, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Onset")

ggplot(df2, aes(x=SmallClusterName, y=wf_End, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("End")

ggplot(df2, aes(x=SmallClusterName, y=MeanAmpBlock, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Mean Amplitude")

ggplot(df2, aes(x=SmallClusterName, y=ACGViolationRatio, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("ACG Violation Ratio")

ggplot(df2, aes(x=SmallClusterName, y=tf_MIFRBlockHz, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("MIFR")

ggplot(df2, aes(x=SmallClusterName, y=tf_ModeIsi, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Mode Isi")

ggplot(df2, aes(x=SmallClusterName, y=tf_Perc5Isi, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("5th perc Isi")

ggplot(df2, aes(x=SmallClusterName, y=tf_CV2Median, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Median CV2")

ggplot(df2, aes(x=SmallClusterName, y=tf_CV, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("CV")

ggplot(df2, aes(x=SmallClusterName, y=tf_Ir, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Ir")

ggplot(df2, aes(x=SmallClusterName, y=tf_Lv, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Lv")

ggplot(df2, aes(x=SmallClusterName, y=tf_Si, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Si")

ggplot(df2, aes(x=SmallClusterName, y=tf_skw, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("Skewness")

ggplot(df2, aes(x=SmallClusterName, y=wf_EndSlopeTau, fill=SmallClusterName)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("EndSlope Tau")

