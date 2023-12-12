.libPaths(c('/Path/to/R_rms/library/', .libPaths()))  # path to rms package

library("rms")

# Output address and filename
Log_dir <- '/Path/to/Output/'
Output_fname <- paste('AllModels_CalibValues_', Sys.Date(), '.csv', sep="")
Output_file <- paste(dirname(Log_dir), 'Results', Output_fname, sep="/")


# Ver02 of DL predictions - used for AMIA abstract and paper
calib_list = list(
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Adult_Det. LSTM/GRU min-max", "xx.png", "calib_values.csv", 0.82),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Adult_Det. LSTM/GRU mean-std", "xx.png", "calib_values.csv", 0.81),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Adult_Det. LSTM/GRU ple-dt", "xx.png", "calib_values.csv", 0.81),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Adult_Det. TDW-CNN min-max", "xx.png", "calib_values.csv", 0.81),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Adult_Det. TDW-CNN mean-std", "xx.png", "calib_values.csv", 0.81),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Adult_Det. TDW-CNN ple-dt", "xx.png", "calib_values.csv", 0.81),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Adult_Det. TCN min-max", "xx.png", "calib_values.csv", 0.81),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Adult_Det. TCN mean-std", "xx.png", "calib_values.csv", 0.80),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Adult_Det. TCN ple-dt", "xx.png", "calib_values.csv", 0.82),
  # AKI
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "AKI LSTM/GRU min-max", "xx.png", "calib_values.csv", 0.92),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "AKI LSTM/GRU mean-std", "xx.png", "calib_values.csv", 0.92),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "AKI LSTM/GRU ple-dt", "xx.png", "calib_values.csv", 0.92),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "AKI TDW-CNN min-max", "xx.png", "calib_values.csv", 0.92),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "AKI TDW-CNN mean-std", "xx.png", "calib_values.csv", 0.91),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "AKI TDW-CNN ple-dt", "xx.png", "calib_values.csv", 0.88),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "AKI TCN min-max", "xx.png", "calib_values.csv", 0.91),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "AKI TCN mean-std", "xx.png", "calib_values.csv", 0.89),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "AKI TCN ple-dt", "xx.png", "calib_values.csv", 0.92),
  # Sepsis
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Sepsis LSTM/GRU min-max", "xx.png", "calib_values.csv", 0.86),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Sepsis LSTM/GRU mean-std", "xx.png", "calib_values.csv", 0.87),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Sepsis LSTM/GRU ple-dt", "xx.png", "calib_values.csv", 0.87),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Sepsis TDW-CNN min-max", "xx.png", "calib_values.csv", 0.86),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Sepsis TDW-CNN mean-std", "xx.png", "calib_values.csv", 0.86),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Sepsis TDW-CNN ple-dt", "xx.png", "calib_values.csv", 0.87),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Sepsis TCN min-max", "xx.png", "calib_values.csv", 0.85),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Sepsis TCN mean-std", "xx.png", "calib_values.csv", 0.86),
  c("log_YYYYmmdd-HHMM", "log_scores.csv", "Sepsis TCN ple-dt", "xx.png", "calib_values.csv", 0.86)
)


# empty dataframe to store all calibration measurements
df_total <- data.frame(matrix(ncol = 5, nrow = 0))

# loop over items (predictors) in the list
for (item in calib_list){
  # read scores
  data <- read.csv(paste(Log_dir, item[[1]], item[[2]], sep = "/"))
  # create calibration values
  res <- val.prob(data$y_pred, as.numeric(data$y_true), pl = TRUE, statloc = FALSE) 
  # collect results
  df <- data.frame("Intercept" = res["Intercept"],
                   "Slope" = res["Slope"],
                   "U" = res["U"],  # Unreliability index
                   "U:p" = res["U:p"],  # Unreliability p-value
                   "Brier" = res["Brier"],
                   row.names = item[[3]])
  # save results for a single predictor in .csv file
  # write.csv(df, paste(Log_dir, item[[1]], item[[5]], sep = "/"))
  # append rows to df_total
  df_total <- rbind(df_total, df)
}
# save calib measurements for all models into one .csv file
write.csv(df_total, Output_file)



