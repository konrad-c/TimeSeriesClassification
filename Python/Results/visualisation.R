library(ggplot2)
library(reshape2)
library(gridExtra)

setwd("C:/Users/kocyb_000/Documents/Uni/TimeSeriesClassification/Python/Results")

# Euclidean Length 
cricket <- read.csv("Cricket/AccuracyLengthEuclidean.csv")
cricket <- read.csv("Cricket/AccuracyLengthDTW_MaxWindow.csv")
cricket$TimeSeriesLength <- factor(cricket$TimeSeriesLength)

plotXYZ <- function(df, metric){
  results <- lapply(c(1,2,3), function(x){
    axis <- c("AccuracyX","AccuracyY","AccuracyZ")
    data_name <- c("CricketX -","CricketY -","CricketZ -")
    title_val <- paste0(data_name[x], " Classification Accuracy (",metric,") vs. Time Series Length")
    return(
      ggplot(df, aes_string(x="TimeSeriesLength", y=axis[x])) +
        geom_boxplot() +
        #geom_point() +
        #geom_smooth(method="loess") +
        #scale_y_continuous(limits=c(0,1)) +
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

plots = plotXYZ(cricket,"Euclidean")
plots[[1]]
plots[[2]]
plots[[3]]
grid.arrange(plots[[1]],plots[[2]],plots[[3]])

# Wafer
wafer <- read.csv("Wafer/AccuracyLengthEuclidean.csv")
wafer <- read.csv("Wafer/AccuracyLengthDTW.csv")

# WAFER Boxplot
wafer$TimeSeriesLength = factor(wafer$TimeSeriesLength)
ggplot(wafer, aes(x=TimeSeriesLength, y=Accuracy)) +
  geom_boxplot() +
  theme_bw() +
  #scale_y_continuous(limits=c(0.8,1)) +
  labs(
    title="Wafer Classification Accuracy (Euclidean) vs. Time Series Length",
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







