RF = RandomForestClassifier()
RF.fit(x_train, y_train)
p_RF = RF.predict(x_test)
print(classification_report(y_test, p_RF))

param_grid = { 
    'n_estimators': range(100,300,25),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' :range(1,10),
    'random_state':range(100,250,50),
    'criterion' :['gini', 'entropy']
}
CV_RF = GridSearchCV(estimator= RF, param_grid=param_grid, cv= 5)
CV_RF.fit(x_train, y_train)

CV_RF.best_params_

model_RF =RandomForestClassifier(random_state=200, max_features='sqrt', n_estimators= 125, max_depth=7, criterion='entropy')
model_RF.fit(x_train, y_train)
pred_RF = model_RF.predict(x_test)
print(classification_report(y_test, pred_RF))

cm = confusion_matrix(y_test, pred_RF)
plt.figure(figsize = (6, 4))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.title("Confusion Matrix of Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_pred_proba = model_RF.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Area under curve = "+str(auc))
plt.legend(loc=4)
plt.show()

plt.figure(figsize=(80,40))
plot_tree(model_RF.estimators_[5], feature_names=x.columns, class_names=['Healthy', 'Parkinsons'], filled=True)
plt.show()