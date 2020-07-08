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

data = read.csv("WardClustering_ACGs_and_OtherFeatures_5Classes.csv")  # latest file!

# Define factor columns
cols <- c("SampleP", "Sample", "Unit", "RespPk", "RespGo", "RespGr", "RespMF", "RespMLI", "Cluster_x",
          "Cluster_y", "SmallCluster", "SmallClusterName", "ClusterACG")
data[cols] <- lapply(data[cols], factor)

# Define numeric columns for PCA
nums <- unlist(lapply(data, is.numeric))  
df <- data[ , nums]
df <- df[,sapply(df, function(v) var(v, na.rm=TRUE)!=0)]

# PCA (we need to ignore missing values from responsive units)
df.pca <- prcomp(na.omit(df), center = TRUE, scale = TRUE)
summary(df.pca)  
ggbiplot(df.pca)

# See only electrical
df2 <- select(df,
             MFRBlockHz,
             tf_MedIsi,
             tf_Entropy,
             tf_CV2Mean,
             tf_LvR)

df.pca <- prcomp(na.omit(df2), center = TRUE, scale = TRUE)
summary(df.pca)  
ggbiplot(df.pca)

# Extract PCs
pcs <- data.frame(df.pca$x[,1:5])

# Ideal number of clusters
wss <- (nrow(df)-1)*sum(apply(df2,2,var))
for (i in 2:9) wss[i] <- sum(kmeans(df2, centers=i)$withinss)
plot(1:9, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# Ward Hierarchical Clustering
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=2) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 2, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)


# Visualize dendrograms
fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

df_ward <- mutate(df2,
                  Sample = data$Sample,
                  Unit=data$Unit,
                  Cluster = groups)

# Export Pk cells
df_ward[(df_ward$Cluster==2),]
write.csv(df_ward[(df_ward$Cluster==2),], row.names = F, 'WardClustering_OnlyPk.csv')

# Let's try to isolate CS
res <- df_ward[(df_ward$Cluster==1),]

df3 <- select(res,
              MFRBlockHz,
              tf_MedIsi,
              tf_Entropy,
              tf_CV2Mean,
              tf_LvR)

df.pca <- prcomp(na.omit(df3), center = TRUE, scale = TRUE)
summary(df.pca)  
ggbiplot(df.pca)

# Extract PCs
pcs <- data.frame(df.pca$x[,1:5])

d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=2) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 2, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

df_ward <- mutate(df3,
                  Sample = res$Sample,
                  Unit=res$Unit,
                  Cluster = groups)




# count(df_ward$Cluster==2)
write.csv(df_ward[(df_ward$Cluster==2),], row.names = F, 'WardClustering_OnlyCS.csv')

# We got to isolate CS!!!

gm <- df_ward[(df_ward$Cluster==1),]

df4 <- select(gm,
              MFRBlockHz,
              tf_MedIsi,
              tf_Entropy,
              tf_CV2Mean,
              tf_LvR)

df.pca <- prcomp(na.omit(df4), center = TRUE, scale = TRUE)
summary(df.pca)  
ggbiplot(df.pca)

# Extract PCs
pcs <- data.frame(df.pca$x[,1:5])

d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=4) 

df_ward <- mutate(df4,
                  Sample = gm$Sample,
                  Unit=gm$Unit,
                  Cluster = groups)

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 7, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

write.csv(df_ward, row.names = F, 'WardClustering_GrMl.csv')

count(df_ward$Cluster==1)
count(df_ward$Cluster==2)
count(df_ward$Cluster==3)
count(df_ward$Cluster==4)
