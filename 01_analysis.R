# ==============================================================
#  Phone Battery Drain Analyzer Using R
#  Script 1 : Complete analysis pipeline (Stages 1 - 7)
#  Run from the project root folder:  source("R/01_analysis.R")
# ==============================================================

# ---- 0. Packages ---------------------------------------------
pkgs <- c("ggplot2", "dplyr", "corrplot", "caret", "ranger")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}
dir.create("outputs", showWarnings = FALSE)
dir.create("models",  showWarnings = FALSE)

# ---- Stage 1 : Loading the dataset ---------------------------
battery <- read.csv("data/battery_data.csv", stringsAsFactors = FALSE)
head(battery)
dim(battery)        # rows and columns
names(battery)
str(battery)
summary(battery)

# ---- Stage 2 : Data cleaning ---------------------------------
# 2.1 Missing values
na_summary <- colSums(is.na(battery))
print(na_summary)

# 2.2 Duplicate rows
dup_count <- sum(duplicated(battery))
cat("Duplicate rows:", dup_count, "\n")
battery <- battery[!duplicated(battery), ]

# 2.3 Fill missing numeric values with the column median
for (col in c("SignalStrength", "BatteryTempC")) {
  battery[[col]][is.na(battery[[col]])] <- median(battery[[col]], na.rm = TRUE)
}
cat("Missing values after cleaning:", sum(is.na(battery)), "\n")

# 2.4 Convert text column to factor
battery$NetworkType <- factor(battery$NetworkType, levels = c("WiFi", "4G", "5G"))
cat("Rows after cleaning:", nrow(battery), "\n")

# ---- Stage 3 : Data visualization ----------------------------
# 3.1 Distribution of the target variable
p1 <- ggplot(battery, aes(x = DrainPerHour)) +
  geom_histogram(bins = 40, fill = "#1B5E3A", colour = "white") +
  geom_vline(xintercept = mean(battery$DrainPerHour), colour = "#F5A623", linewidth = 1) +
  ggtitle("Distribution of Battery Drain per Hour") +
  xlab("Drain (% per hour)") + ylab("Count")
print(p1)
ggsave("outputs/hist_drain.png", p1, width = 7, height = 4.5, dpi = 150)

# 3.2 Correlation matrix of numeric features
num_vars <- battery[, sapply(battery, is.numeric)]
cor_mat  <- cor(num_vars, use = "pairwise.complete.obs")
png("outputs/correlation_matrix.png", width = 900, height = 800, res = 120)
corrplot(cor_mat, method = "color", tl.cex = 0.7, tl.col = "black")
dev.off()
corrplot(cor_mat, method = "color", tl.cex = 0.7, tl.col = "black")

# 3.3 Drain by network type
p2 <- ggplot(battery, aes(x = NetworkType, y = DrainPerHour, fill = NetworkType)) +
  geom_boxplot() +
  scale_fill_manual(values = c("#7BC47F", "#1B5E3A", "#F5A623")) +
  ggtitle("Battery Drain by Network Type") + ylab("Drain (% per hour)") +
  theme(legend.position = "none")
print(p2)
ggsave("outputs/drain_by_network.png", p2, width = 7, height = 4.5, dpi = 150)

# 3.4 Brightness vs drain
p3 <- ggplot(battery, aes(x = ScreenBrightness, y = DrainPerHour)) +
  geom_point(alpha = 0.25, colour = "#1B5E3A") +
  geom_smooth(method = "lm", colour = "#F5A623", se = FALSE) +
  ggtitle("Screen Brightness vs Battery Drain")
print(p3)
ggsave("outputs/brightness_vs_drain.png", p3, width = 7, height = 4.5, dpi = 150)

# 3.5 Outlier check using the IQR rule
q   <- quantile(battery$DrainPerHour, c(0.25, 0.75))
iqr <- q[2] - q[1]
outliers <- sum(battery$DrainPerHour < q[1] - 1.5 * iqr |
                battery$DrainPerHour > q[2] + 1.5 * iqr)
cat("Outliers in DrainPerHour:", outliers, "(kept - they are genuine heavy-use cases)\n")

# ---- Stage 4 : Data splitting (80 % train / 20 % test) -------
set.seed(42)
trainIndex <- createDataPartition(battery$DrainPerHour, p = 0.8, list = FALSE)
train <- battery[trainIndex, ]
test  <- battery[-trainIndex, ]
cat("Train rows:", nrow(train), "| Test rows:", nrow(test), "\n")

# helper functions for evaluation
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
mae  <- function(actual, pred) mean(abs(actual - pred))
r2   <- function(actual, pred) 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)

# ---- Stage 5a : Linear Regression ----------------------------
lm_model <- lm(DrainPerHour ~ ., data = train)
summary(lm_model)
lm_pred  <- predict(lm_model, newdata = test)
cat("Linear Regression -> RMSE:", rmse(test$DrainPerHour, lm_pred),
    " MAE:", mae(test$DrainPerHour, lm_pred),
    " R2:", r2(test$DrainPerHour, lm_pred), "\n")

pa <- ggplot(data.frame(Actual = test$DrainPerHour, Predicted = lm_pred),
             aes(Actual, Predicted)) +
  geom_point(alpha = 0.4, colour = "#1B5E3A") +
  geom_abline(slope = 1, intercept = 0, colour = "#F5A623", linewidth = 1) +
  ggtitle("Linear Regression: Predicted vs Actual")
print(pa)
ggsave("outputs/lm_actual_vs_pred.png", pa, width = 6, height = 5, dpi = 150)

# ---- Stage 5b : Random Forest Regression (ranger) ------------
rf_model <- ranger(DrainPerHour ~ ., data = train, num.trees = 200,
                   importance = "impurity", seed = 42)
rf_pred  <- predict(rf_model, data = test)$predictions
cat("Random Forest -> RMSE:", rmse(test$DrainPerHour, rf_pred),
    " MAE:", mae(test$DrainPerHour, rf_pred),
    " R2:", r2(test$DrainPerHour, rf_pred), "\n")

pb <- ggplot(data.frame(Actual = test$DrainPerHour, Predicted = rf_pred),
             aes(Actual, Predicted)) +
  geom_point(alpha = 0.4, colour = "#F5A623") +
  geom_abline(slope = 1, intercept = 0, colour = "#1B5E3A", linewidth = 1) +
  ggtitle("Random Forest: Predicted vs Actual")
print(pb)
ggsave("outputs/rf_actual_vs_pred.png", pb, width = 6, height = 5, dpi = 150)

# Feature importance
imp <- sort(rf_model$variable.importance, decreasing = TRUE)
imp_df <- data.frame(Feature = names(imp), Importance = as.numeric(imp))
print(head(imp_df, 10))
pc <- ggplot(head(imp_df, 10), aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "#1B5E3A") + coord_flip() +
  ggtitle("Top 10 Features Causing Battery Drain") + xlab("")
print(pc)
ggsave("outputs/feature_importance.png", pc, width = 7, height = 4.5, dpi = 150)

# ---- Stage 5c : Logistic Regression (High drain or not) ------
# The threshold is calculated from the TRAINING set only (avoids data leakage)
thr <- quantile(train$DrainPerHour, 0.67)
train_cl <- train; test_cl <- test
train_cl$HighDrain <- as.integer(train_cl$DrainPerHour > thr)
test_cl$HighDrain  <- as.integer(test_cl$DrainPerHour  > thr)

# DrainPerHour is removed from the predictors, otherwise the label leaks
glm_model <- glm(HighDrain ~ . - DrainPerHour, data = train_cl, family = binomial)
summary(glm_model)

glm_probs <- predict(glm_model, newdata = test_cl, type = "response")
glm_class <- ifelse(glm_probs > 0.5, 1, 0)
conf <- caret::confusionMatrix(factor(glm_class, levels = c(0, 1)),
                               factor(test_cl$HighDrain, levels = c(0, 1)),
                               positive = "1")
print(conf)

# ---- Stage 6 : Model comparison ------------------------------
comparison <- data.frame(
  Model = c("Linear Regression", "Random Forest", "Logistic Regression"),
  Type  = c("Regression", "Regression", "Classification"),
  RMSE  = c(rmse(test$DrainPerHour, lm_pred), rmse(test$DrainPerHour, rf_pred), NA),
  MAE   = c(mae(test$DrainPerHour, lm_pred),  mae(test$DrainPerHour, rf_pred),  NA),
  R2_or_Accuracy = c(r2(test$DrainPerHour, lm_pred),
                     r2(test$DrainPerHour, rf_pred),
                     as.numeric(conf$overall["Accuracy"]))
)
comparison[, 3:5] <- round(comparison[, 3:5], 3)
print(comparison)
write.csv(comparison, "outputs/model_comparison.csv", row.names = FALSE)

# ---- Stage 7 : Save models for the Shiny app -----------------
saveRDS(lm_model,  "models/lm_model.rds")
saveRDS(rf_model,  "models/rf_model.rds")
saveRDS(glm_model, "models/glm_model.rds")
cat("\nAnalysis complete. Charts are in /outputs. Now run: shiny::runApp()\n")
