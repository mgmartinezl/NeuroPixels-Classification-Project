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
GoMF <- data[data$ClusterNo == 5, ]

# Define factor columns
cols <- c("SampleP", "Sample", "Unit", "RespPk", "RespGo", "RespGr", "RespMF", "RespMLI", "Cluster_x",
          "Cluster_y", "SmallCluster", "SmallClusterName", "ClusterACG", "Cluster")

data[cols] <- lapply(data[cols], factor)

# Define numeric columns for PCA
nums <- unlist(lapply(data, is.numeric))  
df <- data[ , nums]
df <- df[,sapply(df, function(v) var(v, na.rm=TRUE)!=0)]

GoMF <- df[df$ClusterNo == 5, ]

# See only electrical
df2 <- select(GoMF,
              MFRBlockHz,
              tf_MedIsi,
              tf_Entropy,
              tf_CV2Mean,
              tf_LvR)

# PCA (we need to ignore missing values from responsive units)
df.pca <- prcomp(na.omit(df2), center = TRUE, scale = TRUE)
summary(df.pca)  
# ggbiplot(df.pca)

pcs <- data.frame(df.pca$x[,1:5])

# We clearly have 2 clusters here. Hopefully Golgi and MF separated
fviz_nbclust(pcs, hcut, method = c("silhouette", "wss", "gap_stat"))

# Ward
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=2) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 2, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

df_ward <- mutate(df2,
                  Sample = GoMF$Sample,
                  Unit=GoMF$Unit,
                  RespPk = GoMF$RespPk,
                  RespGo = GoMF$RespGo,
                  RespMLI = GoMF$RespMLI,
                  RespMF = GoMF$RespMF,
                  RespGr = GoMF$RespGr,
                  ClusterGoMF = groups)

write.csv(df_ward, row.names = F, 'WardClustering_GoMF.csv')
