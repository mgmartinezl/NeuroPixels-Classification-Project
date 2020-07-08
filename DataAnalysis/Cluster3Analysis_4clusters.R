library(dplyr)
library(ggbiplot)
library(rgl)
library(RColorBrewer)
library(scales)
library(pvclust)
library(factoextra)
library(fpc)
library(NbClust)
library(Rtsne)

data = read.csv("WardClustering_WithNonQualityResponsiveUnits_3Classes_5Responsive_AllFeatures.csv")

cluster_3 = data %>% filter(Cluster == 2)

df <- select(cluster_3,
             MFRBlockHz,
             tf_MedIsi,
             tf_Entropy,
             tf_CV2Mean,
             tf_LvR)

# df <- select(cluster_3,
#              MFRBlockHz,
#              tf_MedIsi,
#              tf_Entropy,
#              tf_CV2Mean,
#              tf_LvR,
#              wf_RiseTime,
#              wf_PosDecayTime,
#              wf_FallTime,
#              wf_NegDecayTime,
#              wf_MaxAmpNorm,
#              wf_Duration,
#              wf_PosHwDuration,
#              wf_NegHwDuration,
#              wf_Onset,
#              wf_End,
#              wf_Crossing,
#              wf_Pk10,
#              wf_Pk90,
#              wf_Pk50,
#              wf_PkTrRatio,
#              wf_DepolarizationSlope,
#              wf_RepolarizationSlope,
#              wf_RecoverySlope)

# PCA results
df.pca <- prcomp(df, center = TRUE, scale = TRUE)
summary(df.pca) 
ggbiplot(df.pca)

# PCs
pcs <- data.frame(df.pca$x[,1:5])
# plot3d(pcs$PC3, pcs$PC4, pcs$PC5)
# plot(pcs$PC1, pcs$PC2)
# fviz_eig(df.pca)


# Optimal number of clusters
#wss <- (nrow(df)-1)*sum(apply(df,2,var))
#for (i in 2:9) wss[i] <- sum(kmeans(df, centers=i)$withinss)
#plot(1:9, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")


# tSNE
# df <- unique(df) # Required by tSNE
# tsne <- Rtsne(df, dims=2, perplexity=80, verbose=TRUE, max_iter=500)
# plot(tsne$Y, col='gray', ylab='tSNE y2', xlab='tSNE y1')

# tSNE suggests the presence of 3 clusters!

# Ward Hierarchical Clustering
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=4) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 4, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# Visualize dendrograms
fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

# Apply k-means 
# k <- kmeans(pcs, 3, nstart=25, iter.max=1000)
# 
# library(RColorBrewer)
# library(scales)
# palette(alpha(brewer.pal(9,'Set1'), 0.5))
# plot(pcs, col=k$clust, pch=16)
# plot(pcs$PC1, pcs$PC2, col=k$clust, pch=8, xlab='PC1', ylab='PC2')
# plot3d(pcs$PC1, pcs$PC2, pcs$PC3, col=k$clust)
# 
# sort(table(k$clust))
# clust <- names(sort(table(k$clust)))
# 
# boxplot(df$MFRBlockHz ~ k$cluster,
#         xlab='Cluster', ylab='MFR',
#         main='MFR by Cluster')
# 
# # Append cluster assignment
# df_kmeans <- data.frame(df, Cluster = k$cluster)
# df_kmeans <- data.frame(df_kmeans, Sample=cluster_3$Sample, Unit=cluster_3$Unit,
#                         RespPk=cluster_3$RespPk,
#                         RespGo=cluster_3$RespGo, RespGr=cluster_3$RespGr,
#                         RespMF=cluster_3$RespMF)
# write.csv(df_kmeans, row.names = F, 'KMeansCluster3_Go_MF.csv')

# We need to try this very same exercise with waveform-based features

# Let's apply DBSCAN >> not useful!
# db <- fpc::dbscan(pcs, eps = 0.6, MinPts = 5)
# 
# dbscan::kNNdistplot(pcs, k =  5)
# abline(h = 1, lty = 2)
# 
# # Plot DBSCAN results
# fviz_cluster(db, data = pcs, stand = FALSE,
#              ellipse = FALSE, show.clust.cent = FALSE,
#              geom = "point", palette = "jco", ggtheme = theme_classic())
# 

df_ward <- mutate(df, 
                  Sample=cluster_3$Sample, 
                  Unit=cluster_3$Unit, 
                  RespPk=cluster_3$RespPk,
                  RespGo=cluster_3$RespGo, 
                  RespGr=cluster_3$RespGr,
                  RespMF=cluster_3$RespMF,
                  RespMLI=cluster_3$RespMLI,
                  wf_RiseTime=cluster_3$wf_RiseTime, 
                  wf_PosDecayTime=cluster_3$wf_PosDecayTime,
                  wf_FallTime=cluster_3$wf_FallTime,
                  wf_NegDecayTime=cluster_3$wf_NegDecayTime,
                  wf_MaxAmpNorm=cluster_3$wf_MaxAmpNorm,
                  wf_Duration=cluster_3$wf_Duration,
                  wf_PosHwDuration=cluster_3$wf_PosHwDuration,
                  wf_NegHwDuration=cluster_3$wf_NegHwDuration,
                  wf_Onset=cluster_3$wf_Onset,
                  wf_End=cluster_3$wf_End,
                  wf_Crossing=cluster_3$wf_Crossing,
                  wf_Pk10=cluster_3$wf_Pk10,
                  wf_Pk90=cluster_3$wf_Pk90,
                  wf_Pk50=cluster_3$wf_Pk50,
                  wf_PkTrRatio=cluster_3$wf_PkTrRatio,
                  wf_DepolarizationSlope=cluster_3$wf_DepolarizationSlope,
                  wf_RepolarizationSlope=cluster_3$wf_RepolarizationSlope,
                  wf_RecoverySlope=cluster_3$wf_RecoverySlope,
                  Cluster = groups)

write.csv(df_ward, row.names = F, 'WardClustering_GranularMolecularLayer_4clusters.csv')

# Data visualization

# myColors <- ifelse(df_ward$Cluster==3 , rgb(1,0.4,0.6,0.6), 
#                    ifelse(df_ward$Cluster==4, rgb(1,1,0.2,0.6),
#                           "gray90"))

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)
c4 <- c("#FF0099FF", "#CC00FFFF", "#3300FFFF", "#00FFFFFF")

boxplot(df_ward$MFRBlockHz ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='MFR', 
        main='MFR by Cluster', frame=FALSE)

boxplot(df_ward$tf_Entropy ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='Entropy',
        main='Entropy by Cluster', frame=FALSE)

boxplot(df_ward$tf_CV2Mean ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='CV2 Mean',
        main='CV2 Mean by Cluster', frame=FALSE)

boxplot(df_ward$tf_LvR ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='LvR',
        main='LvR by Cluster', frame=FALSE)

boxplot(df_ward$tf_MedIsi ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='Median ISI',
        main='Median ISI by Cluster', frame=FALSE)

boxplot(df_ward$wf_Duration ~ df_ward$Cluster,
        col=c2, medcol=c3, whiskcol=c1, staplecol=c3, boxcol=c3, outcol=c3, pch=23, cex=2,
        xlab='Cluster', ylab='Duration',
        main='Duration by Cluster', frame=FALSE)

# Let's evaluate the significance of the difference between the means
# Kruskal Wallis Test One Way Anova by Ranks
# Still assumes same variability

kruskal.test(df_ward$MFRBlockHz ~ df_ward$Cluster)  # p-value < 2.2e-16
kruskal.test(df_ward$tf_Entropy ~ df_ward$Cluster)  # p-value < 2.2e-16
kruskal.test(df_ward$tf_CV2Mean ~ df_ward$Cluster)  # p-value < 2.2e-16
kruskal.test(df_ward$tf_LvR ~ df_ward$Cluster)  # p-value < 2.2e-16
kruskal.test(df_ward$tf_MedIsi ~ df_ward$Cluster)  # p-value < 2.2e-16
kruskal.test(df_ward$wf_Duration ~ df_ward$Cluster)  # p-value < 2.2e-16
