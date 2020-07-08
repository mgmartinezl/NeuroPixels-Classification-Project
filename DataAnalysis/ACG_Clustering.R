library(dplyr)
library(ggbiplot)
library(rgl)
library(RColorBrewer)
library(scales)
library(pvclust)
library(factoextra)
library(fpc)
library(NbClust)

data = read.csv("Neuropixels_ACGs.csv")
cols <- c("Sample", "Unit")
data[cols] <- lapply(data[cols], factor)

nums <- unlist(lapply(data, is.numeric))  
df <- data[ , nums]
df <- df[,sapply(df, function(v) var(v, na.rm=TRUE)!=0)]

# PCA results
df.pca <- prcomp(df, center = TRUE, scale = TRUE)
summary(df.pca)  # 3/4 PCs will do the job

ggbiplot(df.pca)
plot3d(pcs$PC1, pcs$PC2, pcs$PC3)

pcs <- data.frame(df.pca$x[,1:8])

wss <- (nrow(df)-1)*sum(apply(df,2,var))
for (i in 2:9) wss[i] <- sum(kmeans(df, centers=i)$withinss)
plot(1:9, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# Ward Hierarchical Clustering
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=5) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 2, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# Visualize dendrograms
fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

df_ward <- mutate(df, 
                  RefPeriod = data$ACG_80,
                  Sample = data$Sample,
                  Unit=data$Unit,
                  ClusterACG = groups)

write.csv(df_ward, row.names = F, 'WardClustering_ACGs_5Classes.csv')


