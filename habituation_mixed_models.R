library(readxl)
library(lmer4)
library(lmerTest)
library(emmeans)

setwd("C:/path/to/data/directory/")
df_habituation <- read_excel("data_file.xlsx")

# Re-code Sex variable (0 = Male, 1 = Female)

df_habituation$Sex <- factor(df_habituation$Sex,
                             levels = c("0","1"),
                             labels = c("Male", "Female"))

# Model 1: Base model with Sex and linear Time effect

model <- lmer(beta ~ Sex + Time_c + (1 | subject),
              data = df_habituation, REML = TRUE)
summary(model)
anova(model, ddf = "Satterthwaite")

# Model 2: Add IU and the Time × IU interaction

model <- lmer(beta ~ Sex + Time_c + IU_c +
                Time_c:IU_c +
                (1 | subject),
              data = df_habituation, REML = TRUE)
summary(model)
anova(model, ddf = "Satterthwaite")

# Model 3: Add STAI as a covariate and Time × STAI interaction

model <- lmer(beta ~ Sex + Time_c + IU_c + STAI_c +
                Time_c:IU_c + Time_c:STAI_c +
                (1 | subject),
              data = df_habituation, REML = TRUE)
summary(model)
anova(model, ddf = "Satterthwaite")

# Model 3: Add STAI as a covariate and Time × STAI interaction
# Compute mean and SD of centered IUS

mean_IU <- mean(df_habituation$IU_c, na.rm = TRUE)
sd_IU   <- sd(df_habituation$IU_c, na.rm = TRUE)

# Define probing values: mean ± 1 SD

probe_vals <- c(mean_IU - sd_IU, mean_IU, mean_IU + sd_IU)

# Estimate the simple slopes of Time at different IUS levels
# (based on the final model including covariates)

slopes <- emtrends(model, specs = ~ IU_c, var = "Time_c",
                   at = list(IU_c = probe_vals))

summary(slopes, infer = c(TRUE, TRUE))
