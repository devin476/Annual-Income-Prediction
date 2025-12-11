library(tidyverse)
library(broom)
library(MASS)
library(pROC)
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


#objective 2 

#This function is used to compare all competing classifiers using the same validation set
#Helper function to compute metrics of the predicted probabilities
get_metrics <- function(truth, prob, threshold = 0.5, positive = ">50K") {
  # truth: factor vector
  # prob: numeric probabilities of positive class
  pred_class <- ifelse(prob >= threshold, positive, setdiff(levels(truth), positive)[1])
  pred_class <- factor(pred_class, levels = levels(truth))
  
  #create confusion matrix
  tab <- table(Predicted = pred_class, Actual = truth)
  
  # confusion matrix cells
  TP <- tab[positive, positive]
  TN <- tab[setdiff(levels(truth), positive), 
            setdiff(levels(truth), positive)]
  FP <- tab[positive, setdiff(levels(truth), positive)]
  FN <- tab[setdiff(levels(truth), positive), positive]
  
  N  <- sum(tab)
  
  #core performance metrics
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  ppv         <- TP / (TP + FP)
  npv         <- TN / (TN + FN)
  prevalence  <- (TP + FN) / N
  accuracy    <- (TP + TN) / N
  
  #AUROC uses full probability information
  roc_obj <- roc(response = truth,
                 predictor = prob,
                 levels = rev(levels(truth)))  # make sure positive is correct
  auc_val <- as.numeric(auc(roc_obj))
  
  #return a tibble so that the results from different models can be binded later
  tibble(
    threshold  = threshold,
    accuracy   = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    PPV        = ppv,
    NPV        = npv,
    prevalence = prevalence,
    AUROC      = auc_val
  )
}

#Simple logistic baseline model metrics
logistic_simple <- glm(income ~ age + education.num + hours.per.week, data = adult_train, family = binomial)

summary(logistic_simple)

#validation set probabilities
valid_prob_logistic_simple <- predict(logistic_simple, newdata = adult_test, type = "response")

#compute classification metrics for the baseline simple logistic model on test set
metrics_logit_simple <- get_metrics(adult_test$income, valid_prob_logistic_simple, threshold = 0.5)
metrics_logit_simple$Model <- "Logit: Simple"
metrics_logit_simple

#Comlex logistic model metrics
logistic_complex <- glm(income ~ age + I(age^2) + education.num + hours.per.week + marital.status + sex + capital.gain + capital.loss +
                          age:hours.per.week, data = adult_train, family = binomial)
summary(logistic_complex)

#validation set probabilities
valid_prob_logistic_complex <- predict(logistic_complex, newdata = adult_test, type= "response")

#compute classification metrics for the baseline complex logistic model on test set
metrics_logit_complex <- get_metrics(adult_test$income, valid_prob_logistic_complex, threshold = 0.5)
metrics_logit_complex$Model <- "Logit: Complex"
metrics_logit_complex

#LDA and QDA model metrics
# Make sure character predictors are factors
adult_train <- adult_train %>% mutate(across(where(is.character), as.factor))
adult_test <- adult_test %>% mutate(across(where(is.character), as.factor))

# LDA
lda_fit <- lda( income ~ age + education.num + hours.per.week +  marital.status + sex + capital.gain + capital.loss,
  data = adult_train)


lda_pred <- predict(lda_fit, newdata = adult_test)
valid_prob_lda <- lda_pred$posterior[, ">50K"]

#compute classification metrics for the LDA logistic model on test set
metrics_lda <- get_metrics(adult_test$income, valid_prob_lda, threshold = 0.5)
metrics_lda$Model <- "LDA"
metrics_lda

# QDA
qda_fit <- qda(income ~ age + education.num + hours.per.week + marital.status + sex + capital.gain + capital.loss,
  data = adult_train)


qda_pred <- predict(qda_fit, newdata = adult_test)
valid_prob_qda <- qda_pred$posterior[, ">50K"]

#compute classification metrics for the QDA logistic model on test set
metrics_qda <- get_metrics(adult_test$income, valid_prob_qda, threshold = 0.5)
metrics_qda$Model <- "QDA"
metrics_qda
