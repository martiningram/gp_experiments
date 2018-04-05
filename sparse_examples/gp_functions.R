library(pdist)
library(ggplot2)

full_rbf_kernel <- function(v1, v2, l, tau) {
  # This is the non-squared distance matrix.
  if (identical(v1, v2)) {
    distances <- dist(v1)
  }
  else {
    distances <- pdist(v1, v2)
  }

  distances <- as.matrix(distances)
  kernel_matrix <- exp(-distances^2 / (2 * l^2))
  return(tau^2 * kernel_matrix)
}

diag_rbf_kernel <- function(v1, v2, l, tau) {
  if (identical(v1, v2)) {
    return (tau^2 * rep(1, dim(v1)[1]))
  }
  else {
    # Make them conformable
    max_dim <- min(nrow(v1), nrow(v2))
    v1 <- v1[1:max_dim, ]
    v2 <- v2[1:max_dim, ]
    
    # Calculate the row-wise square distances
    row_diffs <- exp(-((v1 - v2) %*% (v1 - v2)) / (2 * l^2))
    return (tau^2 * row_diffs)
  }
}

cholesky_inverse <- function(matrix_to_invert) {

  return(chol2inv(chol(matrix_to_invert)))

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

fit_marginal_likelihood_rbf <- function(x_train, y_train, start_sigma = 10,
                                        start_l = 10, start_tau = 10) {

  y_train <- y_train - mean(y_train)

  to_optimize <- function(sigma, l, tau) {
    kernel <- full_rbf_kernel(x_train, x_train, tau = tau, l = l)
    diag(kernel) <- diag(kernel) + sigma^2

    signed_det <- determinant(kernel, logarithm = TRUE)
    first_part <- signed_det[['sign']] * signed_det[['modulus']]
    second_part <- t(y_train) %*% chol2inv(chol(kernel)) %*% y_train

    # We want to minimize, so it's the sum of these
    return(first_part + second_part)
  }

  vector_wrapper <- function(x) {
    return(to_optimize(sigma = x[1], l = x[2], tau = x[3]))
  }

  start_par <- c(start_sigma, start_l, start_tau)

  fit_result <- optim(start_par, vector_wrapper)

  params <- list('sigma' = fit_result$par[1],
                 'l' = fit_result$par[2],
                 'tau' = fit_result$par[3])

  return(params)
}

optimise_and_fit_rbf_gp <- function(x_train, y_train, x_new, start_sigma = 10,
                                    start_l = 10, start_tau = 10) {

  # Fit the kernel hyperparameters
  param_results <- fit_marginal_likelihood_rbf(x_train, y_train, start_sigma =
                                               start_sigma, start_tau =
                                               start_tau, start_l = start_l)

  # Curry the kernel
  kernel_fun <- function(x1, x2) full_rbf_kernel(x1, x2, 
                                                 l = param_results[['l']],
                                                 tau = param_results[['tau']])

  predictions <- predict_points(x_train, x_new, param_results[['sigma']],
                                y_train, kernel_fun)

  return(list('predictions' = predictions,
              'hyperparameters' = param_results))

}

plot_gp <- function(x, mean, covariance, sigma, x_train = NULL, y_train = NULL) {
  
  vars <- diag(covariance)
  
  plot_frame <- data.frame(x = x,
                           means = mean,
                           vars = diag(covariance),
                           vars_with_noise = diag(covariance) + sigma^2)
  
  p <- ggplot(data = plot_frame, aes(x = x, y = means)) +
    geom_point(aes(colour = 'Mean')) +
    geom_ribbon(aes(ymin = means - 2 * sqrt(vars),
                    ymax = means + 2 * sqrt(vars),
                    fill = 'Process noise'),
                alpha = 0.2) +
    geom_ribbon(aes(ymin = means - 2 * sqrt(vars_with_noise),
                    ymax = means + 2 * sqrt(vars_with_noise),
                    fill = 'Measurement noise'),
                alpha = 0.2) +
    geom_line(aes(colour = 'Mean')) +
    theme_classic()
  
  if (!(is.null(x_train)) & !(is.null(y_train))) {
    train_df <- data.frame(x = x_train,
                           y = y_train)
    
    p <- p +     
      geom_point(data = train_df, aes(x = x, y = y, 
                                      colour = 'Observations'), 
                            inherit.aes=FALSE)
  }
  
  return(p)
  
}
