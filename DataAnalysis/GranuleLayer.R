library(dplyr)
library(ggbiplot)
library(rgl)
library(RColorBrewer)
library(scales)
library(pvclust)
library(factoextra)
library(fpc)
library(NbClust)
library(Rtsne)

# Let's explore big cluster #3

data = read.csv("WardClustering_3Clusters.csv")

mix = data %>% filter(Cluster == 2)

df <- select(
  mix, 
  MFRBlockHz, 
  tf_MedIsi,
  tf_Entropy,
  tf_CV2Mean,
  tf_LvR
)

df_scaled <- scale(df)
df_scaled <- unique(df_scaled) # Required by tSNE

tsne <- Rtsne(df_scaled, dims=2, perplexity=40, verbose=TRUE, max_iter=500)

plot(tsne$Y, col='gray', ylab='tSNE y2', xlab='tSNE y1')  # https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne

# tSNE is only good for visualization, no daya analysis and processing

# Ward hierarchical
hc.res <- eclust(tsne, "hclust", k = 2, hc_metric = "euclidean", 
                 hc_method = "ward.D", graph = FALSE)

# Visualize dendrograms
fviz_dend(hc.res, show_labels = FALSE, palette = "jco", as.ggplot = TRUE)

