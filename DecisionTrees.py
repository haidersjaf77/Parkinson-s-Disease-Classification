DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
p_DT = DT.predict(x_test)
print('Training Accuracy: ', DT.score(x_train, y_train))
print('Testing Accuracy: ', DT.score(x_test, y_test))

param_grid = {
    'max_features' : ['auto', 'sqrt', 'log2'],
    'max_depth' : range(1, 10),
    'random_state' : range(30, 210, 30),
    'criterion' : ['gini', 'entropy']
}
CV_DT = GridSearchCV(estimator = DT, param_grid = param_grid, cv = 5)
CV_DT.fit(x_train, y_train)

CV_DT.best_params_

model_DT = DecisionTreeClassifier(random_state = 120, max_features = 'sqrt', max_depth = 6, criterion = 'entropy')
model_DT.fit(x_train, y_train)
pred_DT = model_DT.predict(x_test)
print(classification_report(y_test, pred_DT))

cm = confusion_matrix(y_test, pred_DT)
plt.figure(figsize = (6, 4))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.title("Confusion Matrix of Decision Trees")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_pred_proba = model_DT.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Area under curve = "+str(auc))
plt.legend(loc=4)
plt.show()

plt.figure(figsize = (20, 10))
plot_tree(model_DT, filled = True, feature_names = x.columns, class_names = ['Healthy', 'Parkinsons'])
plt.show()