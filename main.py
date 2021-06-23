from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# Soccer is a game of fine margins. As such, even when one teams outscores the other, there may be variation in other non-scoring metrics, such as:
  # SPI (Soccer Power Index) ratings
  # xG (Expected Goal) tallies
  # Win probability %
  # Adjusted xG
  # Non-Shot xG
# As such, we will run through the FiveThirtyEight SPI dataset, examining Barclays Premier League matches between 2019-20 and 2020-21.
# If all indicators are in favor of one team, we will label them as "UNANIMOUS", meaning that the performance indicators were unanimous in favor of one team.
# If the indicators are mixed between favoring one side or the other, it will be "MIXED."
df = pd.read_csv("https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv").dropna()
df = df[(df.league=="Barclays Premier League") & (df.season >= 2019)].reset_index(drop=True)

# First, we need to generate comparisons between the performance metrics of the home and away team.
# Usually margins like these would be generated as absolute values, but since we want to know how the teams stack up relative to one another,
# we need negative values to clearly differentiate the teams, so we're just subtracing the away performance from the home performance.
df["result"] = df["score1"] - df["score2"]
df["spi"] = df["spi1"] - df["spi2"]
df["xg"] = df["xg1"] - df["xg2"]
df["prob"] = df["prob1"] - df["prob2"]
df["adj"] = df["adj_score1"] - df["adj_score2"]
df["nsxg"] = df["nsxg1"] - df["nsxg2"]

# A bunch of empty lists, which will be useful later.
results = []
spis = []
xgs = []
probs = []
adjs = []
nsxgs = []

# This function does a lot of the heavy lifting.
# If a performance metric is greater than 0 (thus, in favor of the home team), it's "HOME."
# Else, if a performance metric is less than 0 (thus, in favor of the away team), it's "AWAY."
# Else, it's a "DRAW" (unlikely with certain metrics like xG, which go to the hundredths decimal point, but not important since we are comparing all six).
def convert(array, list_to_return):
  array = df[array]
  for x in range(len(array)):
    x = array[x]
    if x > 0:
      list_to_return.append("HOME")
    elif x < 0:
      list_to_return.append("AWAY")
    else:
      list_to_return.append("DRAW")

result = convert("result", results)
spi = convert("spi", spis)
xg = convert("xg", xgs)
prob = convert("prob", probs)
adj = convert("adj", adjs)
nsxg = convert("nsxg", nsxgs)

# Inserting our newly-generated lists into the dataframe at the expense of the original calculations.
df['result'] = results
df['spi'] = spis
df['xg'] = xgs
df['prob'] = probs
df['adj'] = adjs
df['nsxg'] = xgs

# Creating a list for each row of the dataset that only features our new metrics.
# We then compare this with the master result.
# If the two are the same, it's "UNANIMOUS." If otherwise, "MIXED."
i_list = []
master_result = []

for i in range(len(df)):
    i1 = df["result"][i]
    i2 = df["spi"][i]
    i3 = df["xg"][i]
    i4 = df["prob"][i]
    i5 = df["adj"][i]
    i6 = df["nsxg"][i]
    i_list.append([i1, i2, i3, i4, i5, i6])

for i in range(len(i_list)):
    i = i_list[i]
    j = i.count(i[0])==len(i)
    if j == True:
        master_result.append("UNANIMOUS")
    else:
        master_result.append("MIXED")

df["master"] = master_result

# Splitting the relevant metrics into predictors or targets.
X_final = df[['score1','score2','spi1','spi2','xg1','xg2','prob1','prob2','adj_score1','adj_score2','nsxg1','nsxg2']]
y_final = df["master"]
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=4)

models = []
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("GBC", GradientBoostingClassifier()))
models.append(("RFC", RandomForestClassifier()))
models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(("BNB", BernoulliNB()))
models.append(("GNB", GaussianNB()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("SVC", SVC(gamma='auto')))
models.append(("DTC", DecisionTreeClassifier()))

names = []
results = []

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True)
    cv_score = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    names.append(name)
    results.append(cv_score)
    print("{}: {} {}".format(name, np.round(cv_score.mean(),3), np.round(cv_score.std(),3)))

# Random Forest performed the best, so now we'll use GridSearch to find the best parameters.
rf_params = {
    'n_estimators': [200,500],
    'max_features': ['auto', 'log2', 'sqrt'],
    'max_depth': [4,5,6,7,8],
    'criterion': ['entropy', 'gini']
}

rfc = RandomForestClassifier()

rfc_cv = GridSearchCV(estimator=rfc, param_grid=rf_params, cv=5)
rfc_cv.fit(X_train, y_train)

print("Best Params: {}".format(rfc_cv.best_params_))

# Running the model after finding the best parameters.
rfc = RandomForestClassifier(criterion="gini", max_features="auto", max_depth=8, n_estimators=200)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
accuracy = np.round(accuracy_score(y_test, predictions), 3) * 100
print("Test Accuracy on Final Model: {}%".format(accuracy))