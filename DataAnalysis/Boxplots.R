library(dplyr)
library(ggbiplot)
library(rgl)
library(RColorBrewer)
library(scales)
library(pvclust)
library(factoextra)
library(fpc)
library(NbClust)

data = read.csv("WardClusteringOthers.csv")

df <- select(
      data, 
      MFRBlockHz, 
      tf_MedIsi,
      tf_Entropy,
      tf_CV2Mean,
      tf_LvR,
      Cluster)

df <- as.data.frame.matrix(df) 

df$Cluster <- as.factor(df$Cluster)

library(ggplot2)

p <- ggplot(df, aes(x=Cluster, y=MFRBlockHz, color=Cluster)) + geom_violin(trim=FALSE)
p + stat_summary(fun=mean, geom="point", shape=23, size=2)
p + geom_boxplot(width=0.1)