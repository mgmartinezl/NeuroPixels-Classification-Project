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

data = read.csv("WardClustering_ACGs_and_OtherFeatures_5Classes_Comparison.csv")  # latest file!

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

c1 <- rainbow(10)
c2 <- rainbow(10, alpha=0.2)
c3 <- rainbow(10, v=0.7)

# Ward hierarchical
hc <- as.dendrogram(eclust(pcs, "hclust", k = 3, hc_metric = "euclidean", 
                           hc_method = "ward.D", graph = FALSE))

#dend <- color_labels(hc, 2, col = c1)
#dend <- color_branches(dend, 2, col = c1)
#plot(dend) 

fviz_dend(hc, 
          k = 3,                 
          #rect = TRUE,
          #rect_border = c("#2E9FDF", "#00AFBB", "#E7B800"),
          #cex = 0.5,                 
          k_colors = c("#2E9FDF", "#00AFBB", "#E7B800"),
          color_labels_by_k = TRUE, 
          # ggtheme = theme_minimal(),     
          show_labels = FALSE,
          # rect_fill=TRUE
)

fviz_dend(hc, 
          k = 3,                 
          k_colors = c("#FF62BC", "#C77CFF", "#00A9FF"),
          color_labels_by_k = TRUE,     
          show_labels = FALSE
)

#hc.v <- assign_values_to_leaves_edgePar(hc, value = data$Cluster, edgePar = "col")
#plot(hc.v)



#########################################################################################################
# Subclusters

data = read.csv("WardClustering_ACGs_and_OtherFeatures_5Classes_Comparison.csv")  # latest file!

vc <- c('Gr', 'GoMF', 'MLI', 'Unknown')
data = data[data$Cluster %in% vc, ]

# Define factor columns
cols <- c("SampleP", "Sample", "Unit", "RespPk", "RespGo", "RespGr", "RespMF", "RespMLI", "Cluster_x",
          "Cluster_y", "SmallCluster", "SmallClusterName", "ClusterACG", "Cluster")
data[cols] <- lapply(data[cols], factor)


# Define numeric columns for PCA
nums <- unlist(lapply(data, is.numeric))  
df <- data[ , nums]
df <- df[,sapply(df, function(v) var(v, na.rm=TRUE)!=0)]

# See only electrical
df2 <- select(data,
              Cluster,
              MFRBlockHz,
              tf_MedIsi,
              tf_Entropy,
              tf_CV2Mean,
              tf_LvR)

# PCA (we need to ignore missing values from responsive units)
df.pca <- prcomp(na.omit(df2), center = TRUE, scale = TRUE)
summary(df.pca)  
ggbiplot(df.pca)

# Extract PCs
pcs <- data.frame(df.pca$x[,1:5])

# Ward hierarchical
hc <- as.dendrogram(eclust(pcs, "hclust", k = 4, hc_metric = "euclidean", 
                           hc_method = "ward.D", graph = FALSE))

fviz_dend(hc, 
          k = 4,                 
          rect = TRUE,
          #rect_border = c("#F8766D", "#C77CFF", "#00A9FF", "#F6F6F6"),
          #cex = 0.5,                 
          #k_colors = c("#F8766D", "#C77CFF", "#00A9FF", "#F6F6F6"),
          color_labels_by_k = TRUE, 
          # ggtheme = theme_minimal(),     
          show_labels = FALSE,
          rect_fill=TRUE
)

fviz_dend(hc, 
          k = 4,
          #k_colors = c("#2E9FDF", "#E7B800", "#00AFBB", "#FC4E07"),
          k_colors = c("#00A9FF", "#E7B800", "#00AFBB", "#FC4E07"),
          color_labels_by_k = TRUE, 
          show_labels = FALSE
)

df_ward <- mutate(df2, Cluster = groups)

count(df_ward$Cluster==1)
count(df_ward$Cluster==2)
count(df_ward$Cluster==3)
count(df_ward$Cluster==4)
