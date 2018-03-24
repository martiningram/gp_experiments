library(pdist)

# Use Nile dataset
data(Nile)

# Get x and y
x <- start(Nile)[[1]]:end(Nile)[[1]]
y <- as.vector(Nile)
y <- y - mean(y)
y <- y / sd(y)

# OK -- let's write some GP equations
full_rbf_kernel <- function(v1, v2, l, tau) {
  # This is the non-squared distance matrix.
  distances <- as.matrix(pdist(v1, v2))
  kernel_matrix <- exp(-distances^2 / (2 * l^2))
  return(tau^2 * kernel_matrix)
}

predict_points <- function(x_train, x_new, sigma_noise, y, kernel_fun) {
  # Standardise y first
  mean_y <- mean(y)
  y <- y - mean_y
  
  # Compute the main inverse
  training_part <- kernel_fun(x_train, x_train)
  diag(training_part) <- diag(training_part) + sigma_noise^2
  
  inverse <- chol2inv(chol(training_part))
  
  new_with_train <- kernel_fun(x_new, x_train)
  times_inv <- new_with_train %*% inverse
  
  predicted_mean <- times_inv %*% y
  predicted_cov <- kernel_fun(x_new, x_new) - (times_inv %*% t(new_with_train))
  
  return(list('mean' = predicted_mean + mean_y,
              'cov' = predicted_cov))
}

fit_marginal_likelihood_rbf <- function(x_train, y_train) {
  
  y_train <- y_train - mean(y_train)
  
  to_optimize <- function(sigma, l, tau) {
    kernel <- full_rbf_kernel(x_train, x_train, tau = tau, l = l)
    diag(kernel) <- diag(kernel) + sigma^2
    
    first_part <- log(det(kernel))
    second_part <- t(y_train) %*% chol2inv(chol(kernel)) %*% y_train
    
    # We want to minimize, so it's the sum of these
    return(first_part + second_part)
  }
  
  vector_wrapper <- function(x) {
    return(to_optimize(sigma = x[1], l = x[2], tau = x[3]))
  }
  
  start_par <- c(10, 10, 10)
  
  fit_result <- optim(start_par, vector_wrapper)
  
  params <- list('sigma' = fit_result$par[1],
                 'l' = fit_result$par[2],
                 'tau' = fit_result$par[3])
  
  return(params)
}

# Find the best parameters
param_results <- fit_marginal_likelihood_rbf(as.matrix(x), y)

kernel_fun <- function(x1, x2) full_rbf_kernel(x1, x2, 
                                               l = param_results[['l']], 
                                               tau = param_results[['tau']])
test_points <- seq(min(x) - 10, max(x) + 10, length.out = 200)

results <- predict_points(as.matrix(x), as.matrix(test_points), 
                          param_results[['sigma']], 
                          y, kernel_fun)
marginal_vars <- diag(results$cov) + param_results[['sigma']]^2

# Try instead to draw samples from this
library(MASS)

samples <- mvrnorm(n = 1000, mu = results$mean, Sigma = results$cov)
quants <- apply(samples, 2, function (x) quantile(x, probs = c(0.025, 0.5, 0.975)))

library(ggplot2)
to_plot <- data.frame(year = test_points,
                      means = results$mean,
                      vars = marginal_vars,
                      dataset = 'test')

train_df <- data.frame(year = x,
                       value = y,
                       dataset = 'train')

p <- ggplot(data = to_plot, aes(x = year, y = means)) +
  geom_point(colour = 'blue') +
  geom_ribbon(aes(ymin = means - 2 * sqrt(vars), 
                  ymax = means + 2 * sqrt(vars)),
              alpha = 0.5, fill = 'blue') +
  geom_line(colour = 'blue') +
  theme_classic() +
  geom_point(data = train_df, aes(x = year, y = value))

print(p)
