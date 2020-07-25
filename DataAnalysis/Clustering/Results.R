library(dplyr)
library(ggbiplot)
library(rgl)
library(RColorBrewer)
library(scales)
library(pvclust)
library(factoextra)
library(fpc)
library(NbClust)
library(ggplot2)
library(ggpubr)
library(RColorBrewer)
library(clValid)
library(clusterSim)
library(amap)

data = read.csv("Data_v1.csv")

df <- select(data,
             MFR = MFRBlockHz,
             MeanAmp = MeanAmpBlock,
             MIFR = tf_MIFRBlockHz,
             MedIsih = tf_MedIsi,
             ModeIsih= tf_ModeIsi,
             Burstiness = tf_Perc5Isi,
             Entropy = tf_Entropy,
             MeanCV2 = tf_CV2Mean,
             MedianCV2 = tf_CV2Median,
             CV = tf_CV,
             Ir = tf_Ir,
             Lv = tf_Lv,
             LvR = tf_LvR,
             Si = tf_Si,
             skw = tf_skw)

# Main contributing features
df <- select(data,
             MFR = MFRBlockHz,
             MedIsih = tf_MedIsi,
             Entropy = tf_Entropy,
             MeanCV2 = tf_CV2Mean,
             LvR = tf_LvR,)

# df <- select(data,
#              MFR = MFRBlockHz,
#              MeanAmp = MeanAmpBlock,
#              MIFR = tf_MIFRBlockHz,
#              MedIsih = tf_MedIsi,
#              ModeIsih= tf_ModeIsi,
#              Burstiness = tf_Perc5Isi,
#              Entropy = tf_Entropy,
#              MeanCV2 = tf_CV2Mean,
#              MedianCV2 = tf_CV2Median,
#              CV = tf_CV,
#              Ir = tf_Ir,
#              Lv = tf_Lv,
#              LvR = tf_LvR,
#              Si = tf_Si,
#              skw = tf_skw,
#              RiseTime= wf_RiseTime,
#              PosDecayTime = wf_PosDecayTime,
#              FallTime = wf_FallTime,
#              NegDecayTime = wf_NegDecayTime,
#              MaxAmp = wf_MaxAmpNorm,
#              Duration = wf_Duration,
#              PosHwDuration = wf_PosHwDuration,
#              NegHwDuration = wf_NegHwDuration,
#              Onset = wf_Onset,
#              End = wf_End,
#              Crossing = wf_Crossing,
#              Pk10 = wf_Pk10,
#              Pk90 = wf_Pk90,
#              Pk50 = wf_Pk50,
#              PkTrRatio = wf_PkTrRatio,
#              DepolarizationSlope = wf_DepolarizationSlope,
#              RepolarizationSlope = wf_RepolarizationSlope,
#              RecoverySlope = wf_RecoverySlope,
#              EndSlopeTau = wf_EndSlopeTau)


# PCA decomposition
df.pca <- prcomp(na.omit(df), center = TRUE, scale = TRUE)

# PCA visualization
summary(df.pca)  
ggbiplot(df.pca)
fviz_eig(df.pca, addlabels = TRUE, barfill='salmon', 
         barcolor='salmon', ggtheme = theme_classic())

# See variables
fviz_pca_var(df.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# Elbow method >> 3 PCs
plot(df.pca, type='l')
plot(df.pca) 
fviz_eig(df.pca)
fviz_pca(df.pca)
fviz_pca_var(df.pca)
fviz_pca_ind(df.pca, addEllipses = FALSE, addlabels=FALSE, col.ind = "gray", ggtheme = theme_classic())

fviz_pca_var(df.pca, col.var = "contrib",
             ggtheme = theme_minimal())

fviz_pca_var(df.pca, col.var = "contrib", addlabels=FALSE,
             gradient.cols = c("blue", "salmon"),
             ggtheme = theme_classic())


c1 <- c("#FF62BC", "#C77CFF", "#00A9FF", "#E7B800", "#00AFBB", "#FC4E07")
c2 <- c("#FF62BC33", "#C77CFF33", "#00A9FF33", "#E7B80033", "#00AFBB33", "#FC4E0733")

# Some exploratory plots
ggplot(data, aes(x=wf_Duration, y=tf_MIFRBlockHz, fill=RespPk)) + 
  geom_boxplot(alpha=0.8) + theme(legend.position="none") + xlab("") + ylab("MIFR")

plot(data$tf_CV2Mean)
plot(pcs$PC1, pcs$PC2,col=k$clust)

fviz_nbclust(pcs, hcut, method = c("silhouette", "wss", "gap_stat"))
fviz_nbclust(pcs, hcut, method = c("gap_stat"))

# Let's choose from 3 to 5 PCs
pcs <- data.frame(df.pca$x[,1:5])

# 3D analysis
plot3d(pcs$PC1, pcs$PC2, pcs$PC3)

# Even when only fist 2 PCs is already clear the presence of two clusters
plot(pcs$PC1, pcs$PC2)

# Determine number of clusters >> 2? 3?
wss <- (nrow(df)-1)*sum(apply(df,2,var))
for (i in 2:10) wss[i] <- sum(kmeans(df, centers=i)$withinss)
plot(1:10, wss, type="l", xlab="Number of Clusters", ylab="Within Sum of Squares")

# Elbow method for k-means
fviz_nbclust(df, kmeans, method = "silhouette", linecolor ='salmon') +
  geom_vline(xintercept = 2, linetype = 2)

# Compute matrix of distances
distances <- dist(pcs)

# Apply k-means 
k <- kmeans(pcs, 6, nstart=1, iter.max=1000)

# K-Means projected on PCA
ggplot(pcs, aes(PC1, PC2)) + geom_point(aes(color = k$clust)) + 
  scale_color_viridis(option = "C") + scale_fill_viridis(option = "C") + theme_classic()

ggplot(pcs, aes(PC1, PC2)) + geom_point(aes(color = data$RespPk)) + theme_classic()

# Append cluster assignment
df_kmeans <- data.frame(df, cluster = k$cluster, RespPk=data$RespPk, RespCS=data$RespCS, Group=data$SubCluster)

# Compute Davies Bouldin index >> the lower the better
print(index.DB(pcs, df_kmeans$cluster, distances, centrotypes="medoids"))

# Compute dunn index >> the higher the better
print(dunn(distances, df_kmeans$cluster))

out <- vector(mode = "list", length = 10)
for(i in seq_along(out)) {
  out[[i]] <- Kmeans(pcs, 2, iter.max=500, nstart=1, method="euclidean")
}

for(i in seq_along(out[-1])) {
  print(all.equal(out[[i]], out[[i+1]]))
}

# Apply Ward Hierarchical Clustering
fit <- hclust(distances, method="ward")
groups <- cutree(fit, k=6) 

# Append cluster assignment
df_hierarchical <- data.frame(df, cluster = groups, RespPk=data$RespPk)

# Compute Davies Bouldin index  >> the lower the better
print(index.DB(pcs, df_hierarchical$cluster, distances, centrotypes="medoids"))

# Compute dunn index >> the higher the better
print(dunn(distances, df_hierarchical$cluster))

# Compare results
table(df_kmeans$cluster, data$SubCluster)
sort(table(data$SubCluster))
sort(table(k$clust))
sort(table(df_hierarchical$cluster))
clust <- names(sort(table(k$clust)))

row.names(df[k$clust==clust[1],])
row.names(df[k$clust==clust[2],])
row.names(df[k$clust==clust[3],])

boxplot(df$wf_FallTime ~ k$cluster,
        xlab='Cluster', ylab='MFR',
        main='wf_FallTime by Cluster')

# Append cluster assignment
df_kmeans <- data.frame(df, cluster = k$cluster, RespPk=data$RespPk)

# Nice visualization of Ward Hierarchical Clustering
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward")
plot(fit) 
groups <- cutree(fit, k=2) 
rect.hclust(fit, k=2, border="red")

hc <- as.dendrogram(eclust(pcs, "hclust", k = 6, hc_metric = "euclidean", 
                           hc_method = "ward.D", graph = FALSE))

fviz_dend(hc, 
          k = 6,                
          color_labels_by_k = TRUE, 
          ggtheme = theme_classic(),     
          show_labels = FALSE,
          rect_fill=TRUE
)


hc.res <- eclust(pcs, "hclust", k = 2, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# Visualize dendrograms
fviz_dend(hc.res, show_labels = FALSE,
          palette = "jco", as.ggplot = TRUE)


# Export preliminary results
df_ward <- mutate(df, ResponsivePk = data$ResponsiveUnit_Pk, 
                  Sample = data$Sample, Unit = data$Unit, Cluster = groups)

write.csv(df_ward, row.names = F, 'WardClustering.csv')