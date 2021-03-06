library(ggplot2)
library(reshape2)
library(gridExtra)
library(ggpubr)

setwd("C:/Users/kocyb_000/Documents/Uni/TimeSeriesClassification/Python")
#setwd("E:/Documents/Uni/TimeSeriesClassification/Python")

filename = "Meat"
df_euclid = read.csv(paste0(filename,"_results_euclidean.csv"))
df_SAX = read.csv(paste0(filename,"_results_SAX.csv"))
#df_SAX_large = read.csv("SAX_approx_large_raw_results.csv")
#head(df_SAX_large)
#df_SAX = rbind(df_SAX,df_SAX_large)
#remove(df_SAX_large)
SAX_time = df_SAX[,c(1,2,3)]
SAX_acc = df_SAX[,c(1,2,4)]
SAX_time$pruned_shapelet_num = factor(SAX_time$pruned_shapelet_num)
SAX_acc$raw_shapelet_num = factor(SAX_acc$raw_shapelet_num)

time_SAX = ggplot(SAX_time, aes(x=raw_shapelet_num, y=time, color=pruned_shapelet_num)) +
  geom_line() +
  geom_point() +
  scale_color_discrete(name="Pruned Shapelets") +
  scale_y_continuous(limits=c(0,40)) +
  theme_bw() +
  labs(title="Shapelet Discovery and Classification Running Time (Gun_Point)",x="Raw Generated Shapelets", y="Time (s)", legend.title="Pruned Shapelets")

time_e = ggplot(df_euclid[c(1,3)], aes(x=raw_shapelet_num, y=time)) +
  geom_line() +
  geom_point() +
  theme_bw() +
  scale_y_continuous(limits=c(0,40)) +
  labs(x="Raw Generated Shapelets", y="Time (s)", legend.title="Pruned Shapelets")
grid.arrange(time_SAX, time_e)
ggarrange(time_SAX, time_e)

acc_e = ggplot(df_euclid[c(1,4)], aes(x=raw_shapelet_num, y=accuracy)) +
  geom_line() +
  geom_point() +
  theme_bw() +
  scale_y_continuous(limits=c(0.5,1)) +
  labs(title="Ultra-fast shapelets",x="Pruned Shapelets", y="Accuracy", legend.title="Pruned Shapelets")

acc_SAX = ggplot(SAX_acc, aes(x=pruned_shapelet_num, y=accuracy, color=raw_shapelet_num)) +
  geom_line() +
  geom_point() +
  scale_color_discrete(name="Raw Shapelets") +
  theme_bw() + 
  scale_y_continuous(limits=c(0.5,1)) +
  labs(title="SAX Pruning" ,x="Pruned Shapelets", y="Accuracy", legend.title="Pruned Shapelets")

grid.arrange(acc_SAX, acc_e)

### Plot accuracy vs time
df_euclid["Series"] = "Euclidean"
df_SAX["Series"] = "RSAX"
timevsacc = rbind(df_euclid[c(3,4,5)], df_SAX[c(3,4,5)])

acc_time_e = ggplot(timevsacc, aes(x=time, y=accuracy, color=Series)) +
  geom_line() +
  geom_point() +
  theme_bw() +
  scale_y_continuous(limits=c(0.0,1)) +
  labs(title="Ultra-fast shapelets",x="Time (s)", y="Accuracy", legend.title="Pruned Shapelets")
acc_time_e
