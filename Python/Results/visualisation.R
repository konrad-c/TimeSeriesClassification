library(ggplot2)
library(reshape2)
library(gridExtra)

setwd("C:/Users/kocyb_000/Documents/Uni/TimeSeriesClassification/Python/Results")

# Euclidean Length 
eldf <- read.csv("Cricket/AccuracyLengthEuclidean.csv")
eldf$TimeSeriesLength <- factor(eldf$TimeSeriesLength)

plotXYZ <- function(df, metric){
  results <- lapply(c(1,2,3), function(x){
    axis <- c("AccuracyX","AccuracyY","AccuracyZ")
    data_name <- c("CricketX -","CricketY -","CricketZ -")
    title_val <- paste0(data_name[x], " Classification Accuracy (",metric,") vs. Time Series Length")
    return(
      ggplot(df, aes_string(x="TimeSeriesLength", y=axis[x])) +
        geom_boxplot() +
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

plots = plotXYZ(eldf,"Euclidean")
plots[[2]]
grid.arrange(plots[[1]],plots[[2]],plots[[3]])

# Wafer
wafer <- read.csv("Wafer/AccuracyLengthEuclidean.csv")
wafer$TimeSeriesLength = factor(wafer$TimeSeriesLength)

ggplot(wafer, aes(x=TimeSeriesLength, y=Accuracy)) +
  geom_boxplot() +
  theme_bw() +
  #scale_y_continuous(limits=c(0,1)) +
  labs(
    title="Wafer Classification Accuracy (Euclidean) vs. Time Series Length",
    x="Time Series Length",
    y="Classification Accuracy"
  )






