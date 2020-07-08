library(dplyr)
library(ggbiplot)
library(rgl)
library(RColorBrewer)
library(scales)
library(pvclust)
library(factoextra)
library(fpc)
library(NbClust)

data = read.csv("Neuropixels_FeaturesWithNonQualResponsiveUnits.csv")

df <- select(data,
             MFRBlockHz, 
             tf_MedIsi,
             tf_Entropy,
             tf_CV2Mean,
             tf_LvR)


# PCA results
df.pca <- prcomp(df, center = TRUE, scale = TRUE)
summary(df.pca)  # 3/4 PCs will do the job

ggbiplot(df.pca)

plot(df.pca, type='l')
plot(df.pca) 
fviz_eig(df.pca)

# PCs
pcs <- data.frame(df.pca$x[,1:5])

wss <- (nrow(df)-1)*sum(apply(df,2,var))
for (i in 2:9) wss[i] <- sum(kmeans(df, centers=i)$withinss)
plot(1:9, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")

# Ward Hierarchical Clustering
d <- dist(pcs, method = "euclidean")
fit <- hclust(d, method="ward.D")
groups <- cutree(fit, k=3) 

# Ward hierarchical
hc.res <- eclust(pcs, "hclust", k = 3, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# Visualize dendrograms
fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

df_ward <- mutate(df, 
                  Sample=data$Sample, 
                  Unit=data$Unit, 
                  RespPk=data$ResponsiveUnit_Pk,
                  RespGo=data$ResponsiveUnit_Go, 
                  RespGr=data$ResponsiveUnit_GrC,
                  RespMF=data$ResponsiveUnit_MF,
                  RespMLI=data$ResponsiveUnit_MLI,
                  wf_RiseTime=data$wf_RiseTime, 
                  wf_PosDecayTime=data$wf_PosDecayTime,
                  wf_FallTime=data$wf_FallTime,
                  wf_NegDecayTime=data$wf_NegDecayTime,
                  wf_MaxAmpNorm=data$wf_MaxAmpNorm,
                  wf_Duration=data$wf_Duration,
                  wf_PosHwDuration=data$wf_PosHwDuration,
                  wf_NegHwDuration=data$wf_NegHwDuration,
                  wf_Onset=data$wf_Onset,
                  wf_End=data$wf_End,
                  wf_Crossing=data$wf_Crossing,
                  wf_Pk10=data$wf_Pk10,
                  wf_Pk90=data$wf_Pk90,
                  wf_Pk50=data$wf_Pk50,
                  wf_PkTrRatio=data$wf_PkTrRatio,
                  wf_DepolarizationSlope=data$wf_DepolarizationSlope,
                  wf_RepolarizationSlope=data$wf_RepolarizationSlope,
                  wf_RecoverySlope=data$wf_RecoverySlope,
                  Cluster = groups)

write.csv(df_ward, row.names = F, 'WardClustering_WithNonQualityResponsiveUnits_3Classes_5Responsive_AllFeatures.csv')
