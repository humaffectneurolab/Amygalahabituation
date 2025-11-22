import pandas as pd
import statsmodels.api as sm

# 1. Load data
file_path = r"C:/path/to/data/file.xlsx"
data = pd.read_excel(file_path)


# 2. Specify predictors (IUS, STAI_T, sex) and outcome (slope)
# sex is coded as 0 = Male, 1 = Female

X = data[["IU", "STAI_T", "sex"]]
y = data["slope"]

# Add intercept term
X = sm.add_constant(X)

# 3. Fit linear regression model (OLS)
model = sm.OLS(y, X).fit()

# 4. Print regression summary
print(model.summary())



