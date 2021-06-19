import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

df = pd.read_csv('symptomsDataset.csv', index_col=None)
print(df)
columns = df.columns[:-1]

print(columns)
print(len(columns))
x = df[columns]
y = df['prognosis']
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=False)
# random_state=42 to produce the same results across a different run
print("X train")
print(x_train)

# Multinomial Naive Bayes Model
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
MNBprediction = mnb.predict(x_test)
print("Accuracy score MNB")
print(metrics.accuracy_score(y_test, MNBprediction))


# Decision Tree Model
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
DTprediction = dt.predict(x_test)
print("Accuracy score DT", metrics.accuracy_score(y_test, DTprediction))

fig, axes = plot.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
tree.plot_tree(dt,
               filled=True);
fig.savefig('d_tree.png')

# Random Forest Model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
RFpred = rf.predict(x_test)
print("Accuracy score RF", metrics.accuracy_score(y_test, RFpred))

fig, axes = plot.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               filled = True);
fig.savefig('rf_individualtree.png')

fig, axes = plot.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   filled = True,
                   ax = axes[index]);
    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_trees.png')



# import pickle
#
# # write column with pickle
# column_file = './//current_model///columns.pk'
# pickle.dump(list(columns), open(column_file, 'wb'))
#
# # write selected model with pickle
# model = rf
# filename = './//current_model///model.pk'
# pickle.dump(model, open(filename, 'wb'))
#
# # load and test selected model
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.predict(x_test)
# print("Accuracy score loaded model", metrics.accuracy_score(y_test, result))
