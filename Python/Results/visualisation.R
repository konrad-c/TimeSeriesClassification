library(ggplot2)
library(reshape2)
library(gridExtra)

setwd("C:/Users/kocyb_000/Documents/Uni/TimeSeriesClassification/Python/Results")
#setwd("E:/Documents/Uni/TimeSeriesClassification/Python/Results")

# Cricket
data <- read.csv("Cricket/AccuracyLengthEuclidean.csv")
data <- read.csv("Cricket/AccuracyLengthDTW_MaxWindow.csv")

# Gestures
data <- read.csv("Gestures/AccuracyLengthEuclidean.csv")
data <- read.csv("Gestures/AccuracyLengthDTW_MaxWindow.csv")

data$Group = factor(data$TimeSeriesLength)
plotXYZ <- function(df, name, metric){
  results <- lapply(c(1,2,3), function(x){
    axis <- c("AccuracyX","AccuracyY","AccuracyZ")
    data_name <- c(paste0(name,"X -"),paste0(name,"Y -"),paste0(name,"Z -"))
    title_val <- paste0(data_name[x], " Classification Accuracy (",metric,") vs. Time Series Length")
    return(
      ggplot(df, aes_string(x="TimeSeriesLength", y=axis[x], group="Group")) +
        geom_boxplot(width=15) +
        #geom_point() +
        #geom_smooth(method="loess") +
        #scale_y_continuous(limits=c(0.5,0.67)) +
        theme_bw() +
        labs(
          title=title_val,
          x="Time Series Length",
          y="Classification Accuracy"
        )#, legend.title="")
    )
  })
  return(results)
}

plots = plotXYZ(data, "Cricket","DTW")
plots = plotXYZ(data, "Gestures","DTW")
plots[[1]]
plots[[2]]
plots[[3]]
grid.arrange(plots[[1]],plots[[2]],plots[[3]])

# Wafer
wafer <- read.csv("Wafer/AccuracyLengthEuclidean.csv")
wafer <- wafer[as.numeric(wafer$TimeSeriesLength) > 3,]

wafer <- read.csv("Wafer/AccuracyLengthDTW.csv")

wafer$Group = factor(wafer$TimeSeriesLength)

# WAFER Boxplot
ggplot(wafer, aes(x=TimeSeriesLength, y=Accuracy, group=Group)) +
  geom_boxplot(width=10) +
  theme_bw() +
  #scale_y_continuous(limits=c(0.8,1)) +
  labs(
    title="Wafer Classification Accuracy (DTW) vs. Time Series Length",
    x="Time Series Length",
    y="Classification Accuracy"
  )

# WAFER Scatter plot
wafer <- read.csv("Wafer/AccuracyLengthEuclidean.csv")
wafer <- read.csv("Wafer/AccuracyLengthDTW.csv")
ggplot(wafer, aes(x=TimeSeriesLength, y=Accuracy)) +
  geom_point() +
  geom_smooth(method = "loess") +
  theme_bw() +
  labs(
    title="Wafer Classification Accuracy (Euclidean) vs. Time Series Length",
    x="Time Series Length",
    y="Classification Accuracy"
  )

#### LENGTHS ####
cdata <- read.csv("Cricket/Lengths.csv")
cdata$Dataset <- "Cricket"
gdata <- read.csv("Gestures/Lengths.csv")
gdata$Dataset <- "Gestures"
wdata <- read.csv("Wafer/Lengths.csv")
wdata$Dataset <- "Wafer"

ldata <- rbind(cdata,gdata,wdata)

cgraph <- ggplot(cdata, aes(x=Dataset,y=Length)) +
  geom_boxplot() +
  theme_bw()+
  labs(x="", y="") +
  coord_flip()
ggraph <- ggplot(gdata, aes(x=Dataset,y=Length)) +
  geom_boxplot() +
  theme_bw()+
  labs(x="", y="") +
  coord_flip()
wgraph <- ggplot(wdata, aes(x=Dataset,y=Length)) +
  geom_boxplot() +
  theme_bw()+
  labs(x="",y="Time Series Length") +
  coord_flip()
grid.arrange(cgraph, ggraph, wgraph, ncol=1, nrow=3)

sqrt(var(data$AccuracyX))
[1] 0.06147466
> sqrt(var(data$AccuracyY))
[1] 0.06416432
> sqrt(var(data$AccuracyZ))
[1] 0.05923888

(0.5975-0.5293)/0.05923888
