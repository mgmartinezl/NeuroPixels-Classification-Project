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

data = read.csv("UltimateData.csv")  # latest file!

# Define factor columns
cols <- c("SampleP", "Sample", "Unit", "RespPk", "RespGo", "RespGr", "RespMF", "RespMLI", "Cluster_x",
          "Cluster_y", "SmallCluster", "SmallClusterName", "ClusterACG", "Cluster")

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
              ClusterNo,
              MFRBlockHz,
              tf_MedIsi,
              tf_Entropy,
              tf_CV2Mean,
              tf_LvR)

df.pca <- prcomp(na.omit(df2), center = TRUE, scale = TRUE)
pcs <- data.frame(df.pca$x[,1:5])

fviz_nbclust(pcs, hcut, method = c("silhouette", "wss", "gap_stat"))
