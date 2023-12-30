import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingClassifier
import joblib


input_excel_file = r"C:\Users\ybf\Desktop\sci\test.xlsx"
df = pd.read_excel(input_excel_file)


X = df.iloc[:, :7]  #
y = df.iloc[:, 7]   #


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)


base_models = [('RandomForest', rf_model), ('NeuralNetwork', nn_model)]


stacking_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))


kf = KFold(n_splits=5)
accuracies = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    stacking_model.fit(X_train, y_train)
    y_pred = stacking_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

average_accuracy = sum(accuracies) / len(accuracies)
print(f':average_accuracy {average_accuracy:.2f}')


model_filename = r'D:\jqxx\stacked_model.pkl'
joblib.dump(stacking_model, model_filename)

print("saved")

import pandas as pd
from sklearn.metrics import roc_curve, auc
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize



test_excel_file = r"C:\Users\ybf\Desktop\sci\test.xlsx"
test_df = pd.read_excel(test_excel_file)

model_filename = r'D:\jqxx\stacked_model.pkl'
stacking_model = joblib.load(model_filename)


X_test = test_df.iloc[:, :7]

y_test = test_df.iloc[:, 7]


positive_class_label = y_test.max()
y_test_binary = label_binarize(y_test, classes=[0, positive_class_label]).ravel()


y_score = stacking_model.predict_proba(X_test)


fpr, tpr, _ = roc_curve(y_test_binary, y_score[:, 1])
roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROCcurve (s = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('false')
plt.ylabel('true')
plt.title('curve')
plt.legend(loc="lower right")


roc_curve_filename = r'D:\jqxx\roc_curve nerve.png'
plt.savefig(roc_curve_filename)


test_df['results'] = stacking_model.predict(X_test)
result_excel_file = r'D:\jqxx\predicted_results.xlsx'
test_df.to_excel(result_excel_file, index=False)

print("saved2。")

