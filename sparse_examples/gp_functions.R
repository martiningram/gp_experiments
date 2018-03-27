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
