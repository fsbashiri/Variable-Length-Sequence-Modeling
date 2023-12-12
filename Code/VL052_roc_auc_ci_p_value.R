.libPaths(c('/Path/to/shared_R/library/', .libPaths()))

library("pROC")  # ROC-AUC and 95% CI
library("PRROC")  # PR-AUC
library("optparse")

# specify desired options in a list
option_list <- list(
  make_option(c("-i", "--input"), type="character", 
              default="/Output/log_scores.csv",
              help="Path to .csv file that contains true labels and predicted scores [default %default]"),
  make_option(c("-p", "--pred"), type="character",
              default=" ", 
              help="The predictor object or column to be tested paired with other predictions.
              If None, no p-value is returned [default %default]")
)


# get command line options, if help option encountered print help and exit, 
# otherwise if options not found on command line then set defaults,
opt <- parse_args(OptionParser(option_list=option_list))

# read data from 
data <- read.csv(opt$input) # "/Path/to/Output/log_YYYYmmdd-HHMM/log_scores.csv")
cnames <- colnames(data)[-1]  # first column is y_true, the rest are different predictors

# compute AUC and 95% CI (DeLong)
for (c in cnames)
{
  # print(c)
  cat("predictor: ", c, "\n")
  print(auc(data$y_true, data[[c]]))
  print(ci.auc(data$y_true, data[[c]]))
  print(pr.curve(data[[c]][data$y_true==1], data[[c]][data$y_true==0]))
}

# get the p-value by comparing opt$pred prediction with other predictions
if (opt$pred != " ")
{
  print("\n\nComparisons for p-value:")
  for (c in cnames[!cnames %in% opt$pred])
  {
    cat("====\n* predictors: ", opt$pred, "vs.", c, "\n")
    # create roc curves
    roc1 <- roc(data$y_true, data[[opt$pred]])
    roc2 <- roc(data$y_true, data[[c]])
    # apply roc.test
    print(roc.test(roc1, roc2, method=c("delong")))
  }
}


