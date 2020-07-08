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

df <- select(data,
             MFRBlockHz, 
             tf_MIFRBlockHz,
             tf_MedIsi,
             tf_ModeIsi,
             tf_Perc5Isi,
             tf_Entropy,
             tf_CV2Mean,
             tf_CV2Median,
             tf_CV,
             tf_Ir,
             tf_Lv,
             tf_LvR,
             tf_LcV,
             tf_Si,
             tf_skw)

df <- select(data,
             MFRBlockHz, 
             tf_MedIsi,
             tf_Entropy,
             tf_CV2Mean,
             tf_LvR)

# PCA results
df.pca <- prcomp(df, center = TRUE, scale = TRUE)
summary(df.pca)  # 3/4 PCs will do the job

ggbiplot(df.pca) # there is a clear cluster here

plot(df.pca, type='l')
plot(df.pca) 
fviz_eig(df.pca) # 3 PCs are fair to use by elbow rule, 4 by 85% of variance

# fviz_pca_ind(df.pca,
#             col.ind = "cos2", # Color by the quality of representation
#             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
#             repel = TRUE     # Avoid text overlapping
#             )

# Entropy, MFR are super key! Same finding as vanDijck
fviz_pca_var(df.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
             )

# Try: Entropy, MFR, CV2Mean, LvR, MedIsi

# PCs
pcs <- data.frame(df.pca$x[,1:3])

# 3D analysis
plot3d(pcs$PC1, pcs$PC2, pcs$PC3)

# Determine number of clusters >> 2 clearly
wss <- (nrow(df)-1)*sum(apply(df,2,var))
for (i in 2:9) wss[i] <- sum(kmeans(df, centers=i)$withinss)
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

# Export preliminary results
df_ward <- mutate(df, ResponsivePk = data$ResponsiveUnit_Pk, Sample = data$Sample, Unit = data$Unit, Cluster = groups)
write.csv(df_ward, row.names = F, 'WardClustering.csv')
