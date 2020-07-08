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

# With the input of last clustering, we got rid of some garbage and then we are going to cluster again
# but only with half of the data, so we can do classification with the other part

# Randomly split the data
data = read.csv("UltimateData.csv")

set.seed(777)

c1 = data[data$ClusterNo == 1, ]
c2 = data[data$ClusterNo == 2, ]
c3 = data[data$ClusterNo == 3, ]
c4 = data[data$ClusterNo == 4, ]
c5 = data[data$ClusterNo == 5, ]

ss_c1 = floor(0.5*nrow(c1))
ss_c2 = floor(0.5*nrow(c2))
ss_c3 = floor(0.5*nrow(c3))
ss_c4 = floor(0.5*nrow(c4))
ss_c5 = floor(0.5*nrow(c5))

picked_c1 = sample(seq_len(nrow(c1)),size = ss_c1)
picked_c2 = sample(seq_len(nrow(c2)),size = ss_c2)
picked_c3 = sample(seq_len(nrow(c3)),size = ss_c3)
picked_c4 = sample(seq_len(nrow(c4)),size = ss_c4)
picked_c5 = sample(seq_len(nrow(c5)),size = ss_c5)

development_c1 = c1[picked_c1,]
holdout_c1 = c1[-picked_c1,]

development_c2 = c2[picked_c2,]
holdout_c2 = c2[-picked_c2,]

development_c3 = c3[picked_c3,]
holdout_c3 = c3[-picked_c3,]

development_c4 = c4[picked_c4,]
holdout_c4 = c4[-picked_c4,]

development_c5 = c5[picked_c5,]
holdout_c5 = c5[-picked_c5,]

dev <- rbind(development_c1, development_c2, development_c3, development_c4, development_c5)
holdout <- rbind(holdout_c1, holdout_c2, holdout_c3, holdout_c4, holdout_c5)

write.csv(dev, row.names = F, 'ClusteringData.csv')
write.csv(holdout, row.names = F, 'ClassificationData.csv')


# Part II. Hierarchical clustering on dev set

dev <- read.csv("ClusteringData.csv")

df <- select(dev,
             MFRBlockHz,
             tf_MedIsi,
             tf_Entropy,
             tf_CV2Mean,
             tf_LvR)

# PCA (we need to ignore missing values from responsive units)
df.pca <- prcomp(na.omit(df), center = TRUE, scale = TRUE)
summary(df.pca)  
# ggbiplot(df.pca)

pcs <- data.frame(df.pca$x[,1:5])

# We clearly have 2 clusters here. Hopefully Golgi and MF separated
fviz_nbclust(pcs, hcut, method = c("silhouette", "wss", "gap_stat"))

# Ward
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=3) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 3, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

fviz_dend(as.dendrogram(hc.res), 
          k = 3,
          k_colors = c("#00A9FF", "#E7B800", "#00AFBB", "#FC4E07", "#C77CFF"),
          color_labels_by_k = TRUE, 
          show_labels = FALSE
)

# Now let's split third cluster
df_ward <- cbind(dev, NewCluster = groups)

# sort(table(df_ward$NewCluster))
# names(sort(table(df_ward$NewCluster)))

grMl <- df_ward[df_ward$NewCluster == 2, ]
grMl <- subset(grMl, select=-c(NewCluster))

df2 <- select(grMl,
              MFRBlockHz,
              tf_MedIsi,
              tf_Entropy,
              tf_CV2Mean,
              tf_LvR)

# PCA (we need to ignore missing values from responsive units)
df2.pca <- prcomp(na.omit(df2), center = TRUE, scale = TRUE)
summary(df2.pca)  
# ggbiplot(df.pca)

pcs <- data.frame(df2.pca$x[,1:5])

# We clearly have 2 clusters here. Hopefully Golgi and MF separated
fviz_nbclust(pcs, hcut, method = c("silhouette", "wss", "gap_stat"))

# Ward
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=4) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 4, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

fviz_dend(as.dendrogram(hc.res), 
          k = 4,
          k_colors = c("#00A9FF", "#E7B800", "#00AFBB", "#FC4E07", "#C77CFF"),
          color_labels_by_k = TRUE, 
          show_labels = FALSE
)


df_ward2 <- cbind(grMl, NewCluster = groups)
PkCS <- df_ward[df_ward$NewCluster %in% c(1,3), ]

# sort(table(df_ward2$NewCluster))
df_ward2$NewCluster[df_ward2$NewCluster == 1] <- 8
df_ward2$NewCluster[df_ward2$NewCluster == 2] <- 9
df_ward2$NewCluster[df_ward2$NewCluster == 3] <- 5
df_ward2$NewCluster[df_ward2$NewCluster == 4] <- 6
df_ward2$NewCluster[df_ward2$NewCluster == 8] <- 2
df_ward2$NewCluster[df_ward2$NewCluster == 9] <- 4

# names(sort(table(df_ward2$NewCluster)))

# Put all together
total <- rbind(PkCS, df_ward2)

write.csv(total, row.names = F, 'WardClusteringAll.csv')
