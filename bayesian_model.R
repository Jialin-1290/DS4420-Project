# Load packages for stock data, data work, and moving averages
library(quantmod)
library(dplyr)
library(zoo)
library(MASS)

set.seed(42)

# Set the stock tickers
tickers <- c("NVDA", "AAPL", "MSFT", "AVGO", "MU", "ORCL", "AMD", "TSM")

# Download stock close prices from Yahoo Finance
get_stock_data <- function(tickers, start_date = "2015-01-01", end_date = "2026-01-01") {
  all_data <- list()
  
  # Download one stock at a time
  for (tic in tickers) {
    x <- getSymbols(tic, src = "yahoo", from = start_date, to = end_date, auto.assign = FALSE)
    
    # Keep only date and close price
    df <- data.frame(
      Date = index(x),
      Close = as.numeric(Cl(x))
    )
    
    # Rename the close column to the ticker name
    colnames(df)[2] <- tic
    all_data[[tic]] <- df
  }
  
  # Merge all stocks into one table
  merged <- Reduce(function(x, y) merge(x, y, by = "Date", all = TRUE), all_data)
  merged <- merged %>% arrange(Date)
  
  return(merged)
}

# Get the full stock data
data <- get_stock_data(tickers)

# Make a copy for log prices
log_data <- data

# Add log price columns for each stock
for (tic in tickers) {
  log_data[[paste0("log_", tic)]] <- log(log_data[[tic]])
}

# Fit Bayesian linear regression with Gibbs sampling
bayes_linear_regression <- function(X, y, tau2 = 100, alpha0 = 2, beta0 = 1,
                                    n_iter = 4000, burn_in = 1000) {
  # Number of rows and columns
  n <- nrow(X)
  p <- ncol(X)
  
  # Prior mean and prior variance
  b0 <- matrix(0, nrow = p, ncol = 1)
  V0 <- diag(tau2, p)
  V0_inv <- solve(V0)
  
  # Store draws
  beta_draws <- matrix(NA, nrow = n_iter, ncol = p)
  sigma2_draws <- numeric(n_iter)
  
  # Set starting values
  beta_curr <- matrix(0, nrow = p, ncol = 1)
  sigma2_curr <- 1
  
  # Compute X'X and X'y
  XtX <- t(X) %*% X
  Xty <- t(X) %*% y
  
  # Run Gibbs sampling
  for (iter in 1:n_iter) {
    # Get posterior variance and mean for beta
    Vn <- solve(V0_inv + XtX / sigma2_curr)
    bn <- Vn %*% (V0_inv %*% b0 + Xty / sigma2_curr)
    
    # Draw beta from the conditional posterior
    beta_curr <- MASS::mvrnorm(1, mu = as.numeric(bn), Sigma = Vn)
    beta_curr <- matrix(beta_curr, ncol = 1)
    
    # Get posterior shape and rate for sigma2
    alpha_n <- alpha0 + n / 2
    beta_n <- beta0 + as.numeric(t(y - X %*% beta_curr) %*% (y - X %*% beta_curr)) / 2
    
    # Draw sigma2 from the conditional posterior
    sigma2_curr <- 1 / rgamma(1, shape = alpha_n, rate = beta_n)
    
    # Save the draws
    beta_draws[iter, ] <- as.numeric(beta_curr)
    sigma2_draws[iter] <- sigma2_curr
  }
  
  # Remove burn-in draws
  keep <- (burn_in + 1):n_iter
  
  # Return the main Bayesian results
  list(
    beta_draws = beta_draws[keep, , drop = FALSE],
    sigma2_draws = sigma2_draws[keep],
    posterior_mean = matrix(colMeans(beta_draws[keep, , drop = FALSE]), ncol = 1)
  )
}

# Draw one stock plot
make_plot <- function(plot_df, ticker_name) {
  # Set the plot style
  par(
    bg = "white",
    mar = c(5, 5, 4, 2) + 0.1,
    cex.main = 1.6,
    cex.lab = 1.3,
    cex.axis = 1.1
  )
  
  # Plot the actual price
  plot(
    x = plot_df$Date,
    y = plot_df$Actual,
    type = "l",
    col = "orange",
    lty = 1,
    lwd = 2.5,
    xlab = "Date",
    ylab = "Price",
    main = paste("Bayesian Forecast for", ticker_name, "Stock Price")
  )
  
  # Add the forecast line
  lines(
    x = plot_df$Date,
    y = plot_df$Forecast,
    col = "blue",
    lty = 1,
    lwd = 1.8
  )
  
  # Add the legend
  legend(
    "topleft",
    legend = c("Actual", "Forecast"),
    col = c("orange", "blue"),
    lty = c(1, 1),
    lwd = c(2.5, 1.8),
    bty = "o",
    cex = 1.1
  )
  
  box()
}

# Make empty lists to save results
results_list <- list()
coef_list <- list()
plot_list <- list()

# Run the model for each stock
for (target_ticker in tickers) {
  # Set the target stock log column
  target_log_col <- paste0("log_", target_ticker)
  
  # Get the other stocks
  other_tickers <- tickers[tickers != target_ticker]
  other_log_cols <- paste0("log_", other_tickers)
  
  # Build the model data
  model_data <- log_data %>%
    mutate(
      # Moving averages of the target stock on the log scale
      ma20_raw = zoo::rollmean(.data[[target_log_col]], k = 20, fill = NA, align = "right"),
      ma50_raw = zoo::rollmean(.data[[target_log_col]], k = 50, fill = NA, align = "right"),
      
      # Use one-day lag for moving averages
      ma20 = dplyr::lag(ma20_raw, 1),
      ma50 = dplyr::lag(ma50_raw, 1),
      
      # Use one-day lag for target stock
      x_target_lag1 = dplyr::lag(.data[[target_log_col]], 1),
      
      # Current target value on the log scale
      y_t = .data[[target_log_col]]
    )
  
  # Add one-day lag for the other stocks
  for (col_name in other_log_cols) {
    new_name <- paste0("x_", col_name)
    model_data[[new_name]] <- dplyr::lag(model_data[[col_name]], 1)
  }
  
  # Set the predictor columns
  predictor_cols <- c(
    "x_target_lag1",
    "ma20",
    "ma50",
    paste0("x_", other_log_cols)
  )
  
  # Keep only needed columns and drop missing rows
  model_data <- model_data %>%
    dplyr::select(Date, y_t, all_of(predictor_cols)) %>%
    na.omit()
  
  # Split data into train and test
  n <- nrow(model_data)
  train_size <- floor(0.8 * n)
  
  train_data <- model_data[1:train_size, ]
  test_data <- model_data[(train_size + 1):n, ]
  
  # Build X and y for train and test
  x_train <- as.matrix(train_data %>% dplyr::select(-Date, -y_t))
  y_train <- as.matrix(train_data$y_t)
  
  x_test <- as.matrix(test_data %>% dplyr::select(-Date, -y_t))
  y_test <- as.matrix(test_data$y_t)
  
  # Add intercept column
  x_train <- cbind(1, x_train)
  x_test <- cbind(1, x_test)
  
  # Fit the Bayesian model
  bayes_fit <- bayes_linear_regression(
    x_train,
    y_train,
    tau2 = 100,
    alpha0 = 2,
    beta0 = 1,
    n_iter = 4000,
    burn_in = 1000
  )
  
  # Get posterior mean
  beta_post <- bayes_fit$posterior_mean
  
  # Get posterior draws
  beta_draws <- bayes_fit$beta_draws
  sigma2_draws <- bayes_fit$sigma2_draws
  
  # Save simple convergence check plots for AAPL
  if (target_ticker == "AAPL") {
    # Save trace plot for sigma2
    png("bayesian_AAPL_sigma2_trace.png", width = 3000, height = 1500, res = 300)
    plot(
      sigma2_draws,
      type = "l",
      col = "blue",
      lwd = 1.5,
      xlab = "Iteration",
      ylab = expression(sigma^2),
      main = expression(paste("Trace Plot for ", sigma^2, " (AAPL)"))
    )
    box()
    dev.off()
    
    # Save ACF plot for sigma2
    png("bayesian_AAPL_sigma2_acf.png", width = 3000, height = 1500, res = 300)
    acf(
      sigma2_draws,
      main = expression(paste("ACF for ", sigma^2, " (AAPL)"))
    )
    dev.off()
    
    # Save trace plot for one beta draw
    png("bayesian_AAPL_beta2_trace.png", width = 3000, height = 1500, res = 300)
    plot(
      beta_draws[, 2],
      type = "l",
      col = "blue",
      lwd = 1.5,
      xlab = "Iteration",
      ylab = expression(beta[2]),
      main = expression(paste("Trace Plot for ", beta[2], " (AAPL)"))
    )
    box()
    dev.off()
    
    # Save ACF plot for one beta draw
    png("bayesian_AAPL_beta2_acf.png", width = 3000, height = 1500, res = 300)
    acf(
      beta_draws[, 2],
      main = expression(paste("ACF for ", beta[2], " (AAPL)"))
    )
    dev.off()
  }
  
  # Make empty matrix for prediction draws
  pred_draws <- matrix(NA, nrow = nrow(x_test), ncol = nrow(beta_draws))
  
  # Draw predictions from the posterior predictive distribution
  for (s in 1:nrow(beta_draws)) {
    beta_s <- matrix(beta_draws[s, ], ncol = 1)
    mu_s <- x_test %*% beta_s
    y_s <- rnorm(nrow(x_test), mean = as.numeric(mu_s), sd = sqrt(sigma2_draws[s]))
    pred_draws[, s] <- y_s
  }
  
  # Get posterior predictive mean on the log scale
  y_pred <- matrix(rowMeans(pred_draws), ncol = 1)
  
  # Change log prices back to original prices
  actual_price <- exp(y_test)
  forecast_price <- exp(y_pred)
  
  # Compute error values on the original price scale
  rmse <- sqrt(mean((actual_price - forecast_price)^2))
  mae <- mean(abs(actual_price - forecast_price))
  
  # Save summary results
  results_list[[target_ticker]] <- data.frame(
    Stock = target_ticker,
    `Posterior mean of Sigma²` = mean(sigma2_draws),
    RMSE = as.numeric(rmse),
    MAE = as.numeric(mae),
    check.names = FALSE
  )
  
  # Save posterior mean for each variable
  variable_names <- c("Intercept", colnames(train_data %>% dplyr::select(-Date, -y_t)))
  
  coef_df <- data.frame(
    Variable = variable_names,
    PosteriorMean = as.numeric(beta_post)
  )
  coef_list[[target_ticker]] <- coef_df
  
  # Build plot data on the original price scale
  plot_df <- data.frame(
    Stock = target_ticker,
    Date = as.Date(test_data$Date),
    Actual = as.numeric(actual_price),
    Forecast = as.numeric(forecast_price)
  )
  
  # Save plot data to list for final CSV
  plot_list[[target_ticker]] <- plot_df
  
  # Save plot as PNG
  file_name_png <- paste0("bayesian_", target_ticker, "_plot.png")
  
  png(file_name_png, width = 3000, height = 1500, res = 300)
  make_plot(plot_df, target_ticker)
  dev.off()
  
  # Print progress
  cat("Done:", target_ticker, "\n")
}

# Combine all stock plot data into one CSV file
bayesian_plot_df <- do.call(rbind, plot_list)
write.csv(bayesian_plot_df, "bayesian_all.csv", row.names = FALSE)

# Combine all stock results into one table
results_table <- do.call(rbind, results_list)

# Sort by RMSE and round numbers
results_table <- results_table %>%
  arrange(RMSE) %>%
  mutate(
    `Posterior mean of Sigma²` = round(`Posterior mean of Sigma²`, 6),
    RMSE = round(RMSE, 6),
    MAE = round(MAE, 6)
  )

# Remove row names
rownames(results_table) <- NULL

# Print the final summary table
cat("\nBayesian Linear Regression Summary:\n")
print(results_table)
