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

# df <- select(data,
#              MFRBlockHz, 
#              MeanAmpBlock, 
#              tf_MIFRBlockHz,
#              tf_MedIsi,
#              tf_ModeIsi,
#              tf_Perc5Isi,
#              tf_Entropy,
#              tf_CV2Mean,
#              tf_CV2Median,
#              tf_CV,
#              tf_Ir,
#              tf_Lv,
#              tf_LvR,
#              tf_LcV,
#              tf_Si,
#              tf_skw)

df <- select(data,
             MFRBlockHz, 
             tf_MedIsi,
             tf_Entropy,
             tf_CV2Mean,
             tf_LvR)

# df <- select(data, 
#              MFRBlockHz, 
#              MeanAmpBlock, 
#              tf_MIFRBlockHz,
#              tf_MedIsi,
#              tf_ModeIsi,
#              tf_Perc5Isi,
#              tf_Entropy,
#              tf_CV2Mean,
#              tf_CV2Median,
#              tf_CV,
#              tf_Ir,
#              tf_Lv,
#              tf_LvR,
#              tf_LcV,
#              tf_Si,
#              tf_skw,
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

# Correlations between variables?
#plot(df[,1:16])

# Plot PCA >> Elbow method >> 3 PCs explain 72%, 5 PCs explain 87% of variance
plot(df.pca, type='l')
plot(df.pca) 
fviz_eig(df.pca)
fviz_pca(df.pca)
fviz_pca_ind(df.pca, addEllipses = FALSE, col.ind = "gray")

# Let's choose from 3 to 5 PCs
pcs <- data.frame(df.pca$x[,1:3])

# 3D analysis
plot3d(pcs$PC1, pcs$PC2, pcs$PC3)

# Even when only fist 2 PCs is already clear the presence of two clusters
plot(pcs$PC1, pcs$PC2)

# Determine number of clusters >> 2? 3?
wss <- (nrow(df)-1)*sum(apply(df,2,var))
for (i in 2:10) wss[i] <- sum(kmeans(df, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# Apply k-means 
k <- kmeans(pcs, 3, nstart=25, iter.max=1000)

library(RColorBrewer)
library(scales)
palette(alpha(brewer.pal(9,'Set1'), 0.5))
plot(pcs, col=k$clust, pch=16)

plot3d(pcs$PC1, pcs$PC2, pcs$PC3, col=k$clust)
plot3d(pcs$PC2, pcs$PC3, pcs$PC4, col=k$clust)

sort(table(k$clust))
clust <- names(sort(table(k$clust)))

row.names(df[k$clust==clust[1],])
row.names(df[k$clust==clust[2],])
row.names(df[k$clust==clust[3],])

boxplot(df$wf_FallTime ~ k$cluster,
        xlab='Cluster', ylab='MFR',
        main='wf_FallTime by Cluster')

# Append cluster assignment
df_kmeans <- data.frame(df, cluster = k$cluster)

# Ward Hierarchical Clustering
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward")
plot(fit) 
groups <- cutree(fit, k=2) 
rect.hclust(fit, k=2, border="red")

#library(dendextend)
#avg_dend_obj <- as.dendrogram(fit)
#avg_col_dend <- color_branches(avg_dend_obj, h = 5)
#plot(avg_col_dend)


# Ward Hierarchical Clustering with Bootstrapped p values
# fit <- pvclust(pcs, method.hclust="ward",method.dist="euclidean") # Bootstrapping
# plot(fit) # dendogram with p values
# add rectangles around groups highly supported by the data
# pvrect(fit, alpha=.95)

hc.res <- eclust(pcs, "hclust", k = 2, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# Visualize dendrograms
fviz_dend(hc.res, show_labels = FALSE,
          palette = "jco", as.ggplot = TRUE)

# See variables
fviz_pca_var(df.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# Export preliminary results
df_ward <- mutate(df, ResponsivePk = data$ResponsiveUnit_Pk, Sample = data$Sample, Unit = data$Unit, Cluster = groups)
write.csv(df_ward, row.names = F, 'WardClustering.csv')
