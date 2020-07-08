library(dplyr)
library(ggbiplot)
library(rgl)
library(RColorBrewer)
library(scales)
library(pvclust)
library(factoextra)
library(fpc)
library(NbClust)

data = read.csv("Neuropixels_Features.csv")

df <- select(data,
             wf_MaxAmp,
             wf_MaxAmpTime,
             wf_MinAmp,
             wf_MinAmpTime,
             wf_Duration,
             wf_PosHwDuration,
             wf_NegHwDuration,
             wf_Onset,
             wf_OnsetTime,
             wf_End,
             wf_EndTime,
             wf_PkTrRatio,
             wf_DepolarizationSlope,
             wf_RepolarizationSlope,
             wf_RecoverySlope,
             wf_RiseTime,
             wf_PosDecayTime,
             wf_FallTime,
             wf_NegDecayTime,
             wf_EndSlopeTau)

# PCA results
df.pca <- prcomp(df, center = TRUE, scale = TRUE)
summary(df.pca) 
ggbiplot(df.pca)

pcs <- data.frame(df.pca$x[,1:8])
plot3d(pcs$PC4, pcs$PC5, pcs$PC6)
fviz_eig(df.pca)

# Optimal number of clusters
wss <- (nrow(df)-1)*sum(apply(df,2,var))
for (i in 2:9) wss[i] <- sum(kmeans(df, centers=i)$withinss)
plot(1:9, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# Ward Hierarchical Clustering
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=3) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 3, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# Visualize dendrograms
fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)



df_ward <- mutate(df, 
                  Sample=data$Sample, 
                  Unit=data$Unit, 
                  RespPk=data$RespPk,
                  RespGo=data$RespGo, 
                  RespGr=data$RespGr,
                  RespMF=data$RespMF,
                  RespMLI=data$RespMLI,
                  Cluster = groups)

sort(table(df_ward$Cluster))

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

boxplot(df_ward$wf_Duration ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='Duration', 
        main='Duration by Cluster', frame=FALSE)

boxplot(df_ward$wf_RiseTime ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='RiseTime', 
        main='RiseTime by Cluster', frame=FALSE)

boxplot(df_ward$wf_FallTime ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='FallTime', 
        main='FallTime by Cluster', frame=FALSE)

boxplot(df_ward$wf_PosHwDuration ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='PosHwDuration', 
        main='PosHwDuration by Cluster', frame=FALSE)

boxplot(df_ward$wf_NegHwDuration ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='NegHwDuration', 
        main='NegHwDuration by Cluster', frame=FALSE)

boxplot(df_ward$wf_PkTrRatio ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='PkTrRatio', 
        main='PkTrRatio by Cluster', frame=FALSE)

boxplot(df_ward$wf_DepolarizationSlope ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='DepolarizationSlope', 
        main='DepolarizationSlope by Cluster', frame=FALSE)

boxplot(df_ward$wf_RepolarizationSlope ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='RepolarizationSlope', 
        main='RepolarizationSlope by Cluster', frame=FALSE)

boxplot(df_ward$wf_RecoverySlope ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='RecoverySlope', 
        main='RecoverySlope by Cluster', frame=FALSE)

boxplot(df_ward$wf_PosDecayTime ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='PosDecayTime', 
        main='PosDecayTime by Cluster', frame=FALSE)

boxplot(df_ward$wf_OnsetTime ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='OnsetTime', 
        main='OnsetTime by Cluster', frame=FALSE)

boxplot(df_ward$wf_EndTime ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='EndTime', 
        main='EndTime by Cluster', frame=FALSE)

boxplot(df_ward$wf_NegDecayTime ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='NegDecayTime', 
        main='NegDecayTime by Cluster', frame=FALSE)


