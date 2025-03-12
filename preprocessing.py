x = df.drop('status', axis = 1)
y = df['status']

sm = SMOTE(random_state = 300)
x, y = sm.fit_resample(x, y)

scaler = MinMaxScaler((-1, 1))
features = scaler.fit_transform(x)
label = y

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = 20)