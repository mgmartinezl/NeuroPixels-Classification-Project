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

ward = read.csv("WardClustering.csv")

data <- mutate(data, Cluster=ward$Cluster)

ward_PkC = data %>% filter(Cluster == 1)
ward_others = data %>% filter(Cluster == 2)

df <- select(
        ward_others, 
        wf_RepolarizationSlope,
        wf_RecoverySlope,
        wf_PkTrRatio,
        wf_Duration,
        MeanAmpBlock,
        MFRBlockHz, 
        tf_MedIsi,
        tf_Entropy,
        tf_CV2Mean,
        tf_LvR
)

df <- select(
        ward_others, 
        MFRBlockHz, 
        tf_MedIsi,
        tf_CV2Mean,
        tf_LvR
)

# PCA results over df1
df.pca <- prcomp(df, center = TRUE, scale = TRUE)
summary(df.pca)

# Plot PCA
ggbiplot(df.pca)

# Plot PCA >> Elbow method >> 3 PCs explain 72%, 5 PCs explain 87% of variance
plot(df.pca, type='l')
plot(df.pca) 
fviz_eig(df.pca)
fviz_pca_ind(df.pca, addEllipses = FALSE, col.ind = "gray")

fviz_pca_var(df.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

# Let's choose first 3 pcs
pcs <- data.frame(df.pca$x[,1:3])
plot3d(pcs$PC1, pcs$PC2, pcs$PC3)

# Determine number of clusters >> 2? 3?
wss <- (nrow(df)-1)*sum(apply(df,2,var))
for (i in 2:10) wss[i] <- sum(kmeans(df, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# Ward Hierarchical Clustering
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=2) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 2, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# Visualize dendrograms
fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

# Export preliminary results
df_ward <- mutate(ward_others, NewCluster = groups)
df_ward <- subset(df_ward, select=-c(Cluster))

df_ward$NewCluster[df_ward$NewCluster == 1] <- 4
df_ward$NewCluster[df_ward$NewCluster == 2] <- 3
df_ward$NewCluster[df_ward$NewCluster == 4] <- 2

df_ward$NewCluster
ward_PkC$Cluster

names(df_ward)[names(df_ward) == 'NewCluster'] <- 'Cluster'

whole_clusters <- rbind(ward_PkC, df_ward)
whole_clusters$Cluster

write.csv(whole_clusters, row.names = F, 'WardClusteringOthers.csv')
