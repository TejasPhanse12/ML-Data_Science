import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score

s_data = pd.read_csv("project2.csv")
print("Data imported successfully")

print("\nData Discription: \n", s_data.describe())


sns.lineplot(data=s_data, x="Hours", y="Scores")
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


print("\nCorrelation: \n", s_data.corr(method='pearson'))


print("\nplotting a distribution graph:\n")
hours = s_data['Hours']
scores = s_data['Scores']
print("Hours:\n")
plt.title('distribution graph for hours')
sns.distplot(hours)
plt.show()
print("Scores:\n")
plt.title('distribution graph for scores')
sns.distplot(scores)
plt.show()


X = s_data.iloc[:, :-1].values
y = s_data.iloc[:, 1].values

X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training complete.")

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.title("Test Data Graph")
plt.scatter(X, y)
plt.plot(X, line)
plt.show()

print("Testing data:\n", x_test)  # Testing data - In Hours

y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Printing data frame:\n", df)

actual_pred = pd.DataFrame({'Target': y_test, 'Predicted': y_pred})
print("\nActual data prediction:\n", actual_pred)

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
plt.title("Difference between Target and Predicted Value")
sns.distplot(np.array(y_test-y_pred))
plt.show()

# predict for a student who studeies for x hours
h = float(input("Enter hours of studying: "))
stud_pred = regressor.predict([[h]])
print("if Student studies for {} hours/day then scores {} % in exams.".format(h, stud_pred))

# model evaluation
print("\nModel Evaluation:\n")
print("Mean absolute error: ", metrics.mean_absolute_error(y_test, y_pred))
print("\ncoefficient of determination (R2 Scores): ", r2_score(y_test, y_pred))
