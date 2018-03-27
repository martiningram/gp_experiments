library(ggplot2)
source('./gp_functions.R')

# Use Nile dataset
data(Nile)

# Get x and y
x <- start(Nile)[[1]]:end(Nile)[[1]]
y <- as.vector(Nile)
y <- y - mean(y)
y <- y / sd(y)

# OK -- let's write some GP equations
# Find the best parameters
test_points <- seq(min(x) - 10, max(x) + 10, length.out = 200)

fit <- optimise_and_fit_rbf_gp(as.matrix(x), y, as.matrix(test_points))

param_results <- fit[['hyperparameters']]
results <- fit[['predictions']]

marginal_vars <- diag(results$cov)# + param_results[['sigma']]^2

# Try instead to draw samples from this
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
