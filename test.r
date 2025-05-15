
library(signal)  

compute_max_activity <- function(emg_data) {
  return(max(emg_data, na.rm = TRUE))
}

compute_mean_activity <- function(emg_data) {
  return(mean(emg_data, na.rm = TRUE))
}
compute_median_frequency <- function(emg_data, fs = 1000) {
  # Applying to get the frequency spectrum
  emg_spectrum <- abs(fft(emg_data))^2
  freq <- seq(0, fs/2, length.out = length(emg_spectrum)/2)
  
  # Get the power spectrum up to Nyquist frequency
  emg_spectrum <- emg_spectrum[1:(length(emg_spectrum)/2)]
  
  # Find the median frequency
  cumulative_spectrum <- cumsum(emg_spectrum) / sum(emg_spectrum)
  median_freq <- freq[which.min(abs(cumulative_spectrum - 0.5))]
  
  return(median_freq)
}

emg_data <- read.csv("Test1_heben1_1.csv", skip = 7)

cat("emg data:\n")
#emg_data[[1]]
cat("end emg data\n")

signal <- emg_data[, 1]

max_activity <- compute_max_activity(signal)
mean_activity <- compute_mean_activity(signal)
median_freq <- compute_median_frequency(signal)

cat("Dataset Results:\n")
cat("Max Activity: ", max_activity, "\n")
cat("Mean Activity: ", mean_activity, "\n")
cat("Median Frequency: ", median_freq, " Hz\n\n")


# Statistical comparison (e.g., using t-test)
# Assuming that the data are normally distributed, use a t-test for comparison
#t_max_activity <- t.test(max_activity_1, max_activity_2)
#t_mean_activity <- t.test(mean_activity_1, mean_activity_2)
#t_median_freq <- t.test(median_freq_1, median_freq_2)

# Output statistical comparison results
#cat("\nStatistical Comparison (t-test results):\n")
#cat("Max Activity p-value: ", t_max_activity$p.value, "\n")
#cat("Mean Activity p-value: ", t_mean_activity$p.value, "\n")
#cat("Median Frequency p-value: ", t_median_freq$p.value, "\n")
