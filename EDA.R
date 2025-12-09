library(tidyverse)
library(broom)
##Data Prep
adult <- read.csv("adult.csv")

summary(adult)

# Make outcome a factor with "<=50K" as reference level
adult <- adult %>% mutate(income = str_trim(income), income = factor(income, levels = c("<=50K", ">50K")))

set.seed(123)
n <- nrow(adult)
train_idx <- sample(1:n, size = floor(0.7 * n))
adult_train <- adult[train_idx, ]
adult_test <- adult[-train_idx, ]

##EDA
#output table with count and proportions of each income class
adult_train %>% count(income) %>% mutate(prop = n/sum(n))

adult_train %>% ggplot(aes(x = income, y = age)) +
  geom_boxplot() + labs(title = "Age vs Income")

adult_train %>% ggplot(aes(x = income, y = hours.per.week)) +
  geom_boxplot() + labs(title = "Hours per week vs Income")

unique(adult$income)

#summaries w/ skew
adult_train %>%
  select(age, fnlwgt, education.num, capital.gain, capital.loss, hours.per.week) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  group_by(variable) %>%
  summarise(mean = mean(value), median = median(value), sd = sd(value), IQR = IQR(value),
            skewness = (mean - median) / sd, .groups = "drop")

#distribution plots
adult_train %>%
  select(age, education.num, hours.per.week, capital.gain) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(x = value)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "blue", alpha = 0.7) +
  geom_density(color = "red", linewidth = 1) +
  facet_wrap(~name, scales = "free", ncol = 2) +
  labs(title = "Distribution of Numeric Predictors")

#age vs income
adult_train %>%
  ggplot(aes(x = income, y = age, fill = income)) +
  geom_boxplot(alpha = 0.7) +
  geom_violin(alpha = 0.3) +
  labs(title = "Age Distribution x Income Class", y = "Age", x = "Income")

#ttest for age difference
t.test(age ~ income, data = adult_train)

#education plot
adult_train %>%
  ggplot(aes(x = income, y = education.num, fill = income)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Education x Income Class", y = "Years", x = "Income Class")

#correlation matrix
adult_train %>%
  select(age, education.num, hours.per.week, capital.gain, capital.loss) %>%
  cor(use = "complete.obs")

#age, education, income
adult_train %>%
  ggplot(aes(x = age, y = education.num, color = income)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess") +
  labs(title = "Age vs Education x Income", x = "Age", y = "Education", color = "Income")

##Models
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
