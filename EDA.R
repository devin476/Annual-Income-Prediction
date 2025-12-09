library(tidyverse)
library(broom)

adult <- read.csv("adult.csv")

summary(adult)

# Make outcome a factor with "<=50K" as reference level
adult <- adult %>% mutate(income = str_trim(income), income = factor(income, levels = c("<=50K", ">50K")))

set.seed(123)
n <- nrow(adult)
train_idx <- sample(1:n, size = floor(0.7 * n))
adult_train <- adult[train_idx, ]
adult_test <- adult[-train_idx, ]

#output table with count and proportions of each income class
adult_train %>% count(income) %>% mutate(prop = n/sum(n))

adult_train %>% ggplot(aes(x = income, y = age)) +
  geom_boxplot() + labs(title = "Age vs Income")

adult_train %>% ggplot(aes(x = income, y = hours.per.week)) +
  geom_boxplot() + labs(title = "Hours per week vs Income")

unique(adult$income)

# Model 1: simple baseline with a few key predictors
m1 <- glm(income ~ age + education.num + hours.per.week,
          data = adult_train,
          family = binomial)

# Model 2: add marital status and sex
m2 <- glm(income ~ age + education.num + hours.per.week +
            marital.status + sex,
          data = adult_train,
          family = binomial)

# Model 3: richer model with some additional strong predictors
m3 <- glm(income ~ age + education.num + hours.per.week +
            marital.status + sex +
            capital.gain + capital.loss,
          data = adult_train,
          family = binomial)

# Compare AICs
AIC(m1, m2, m3)
#m3 perfroms the best

#odds ratio
or_tbl <- tidy(m3, conf.int = TRUE, conf.level = 0.95, exponentiate = TRUE)

or_tbl %>%  arrange(term) %>% print(n = Inf)
