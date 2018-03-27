source('./gp_functions.R')


calculate_q <- function(kernel_fun, inducing, a, b) {

  k_au <- kernel_fun(a, inducing)
  k_ub <- kernel_fun(inducing, b)
  k_uu <- kernel_fun(inducing, inducing)
  k_uu_inv <- cholesky_inverse(k_uu)

  return(k_au %*% k_uu_inv %*% k_ub)

}

calculate_dic <- function(sigma_noise, inducing, x_train, y_train, x_new,
                          kernel_fun) {

  k_uf <- kernel_fun(inducing, x_train)
  k_fu <- t(k_uf) # At least I think that's right...?
  k_uu <- kernel_fun(inducing, inducing)

  big_sigma <- cholesky_inverse(sigma_noise^(-2) * k_uf %*% k_fu + k_uu)

  k_star_u <- kernel_fun(x_new, inducing)

  k_star_u_times_sigma <- k_star_u %*% big_sigma 

  dic_mean <- sigma_noise^(-2) * k_star_u_times_sigma %*% k_uf %*% y_train
  dic_cov <- k_star_u %*% big_sigma %*% t(k_star_u)

  return(list('mean' = dic_mean, 'cov' = dic_cov))

}
