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

data = read.csv("WardClustering_ACGs_and_OtherFeatures_5Classes.csv")
cols <- c("Sample", "Unit", "ClusterACG")
data[cols] <- lapply(data[cols], factor)

ggplot(data, aes(x=tf_CV2Mean, y=tf_Entropy, color=SmallClusterName)) + 
  geom_point(alpha=1,aes(shape = SmallClusterName, color = SmallClusterName), size = 2) + 
  xlab("Mean Cv2") + ylab("Entropy") + theme(legend.position="top")  + theme_minimal()

ggplot(data, aes(x=tf_CV2Mean, y=tf_Entropy, color=ClusterACG)) + 
  geom_point(alpha=1,aes(shape = ClusterACG, color = ClusterACG), size = 2) + 
  xlab("Mean Cv2") + ylab("Entropy") + theme(legend.position="top")  + theme_minimal()

ggplot(data, aes(x=tf_CV2Mean, y=MFRBlockHz, color=SmallClusterName)) + 
  geom_point(alpha=1,aes(shape = SmallClusterName, color = SmallClusterName), size = 2) + 
  xlab("Mean Cv2") + ylab("MFR") + theme(legend.position="top")  + theme_minimal()

ggplot(data, aes(x=tf_CV2Mean, y=MFRBlockHz, color=ClusterACG)) + 
  geom_point(alpha=1,aes(shape = ClusterACG, color = ClusterACG), size = 2) + 
  xlab("Mean Cv2") + ylab("MFR") + theme(legend.position="top")  + theme_minimal()

