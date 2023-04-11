"""
Grid Search
Grid search는 하이퍼파라미터를 최적화 해주는 기술임. scikit-learn에서 GridSearchCV class를 통해서 딥러닝에 적용 할 수 있다.
팁!: 보통 grid search는 하나의 thread만 사용하는데 'n_jobs'=-1으로 설정하면 모든 thread를 사용 할 수 있임.

Grid Search를 통해서 최적화된 값 찾기:
    batch size, epoch,
    optimizer,
    learning rate, momentum
    weight initialization
    activation
    dropout regularization
    number of neurons

"""
import keras.optimizers
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

def create_model():#basic model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

tf.random.set_seed(7)
dataset = np.loadtxt("../data/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

"""
Batch Size & Number of Epochs Tuning:
#Best: 0.686198 using {'batch_size': 10, 'epochs': 50}
"""
# model = KerasClassifier(model=create_model, verbose=0)
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [10, 50, 100]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=2, cv=3)
# grid_result = grid.fit(X, Y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# #Best: 0.686198 using {'batch_size': 10, 'epochs': 50}
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

"""
Optimizer Tuning:
#Best: 0.653646 using {'model__optimizer': 'SGD'}
"""
# def create_model(optimizer='adam'):
#     model = Sequential()
#     model.add(Dense(12, input_shape=(8,), activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     #compile model
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(model=create_model, verbose=0, batch_size=10, epochs=50)
#
# # define the grid search parameters
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
## model__
# param_grid = dict(model__optimizer=optimizer)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(X, Y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# #Best: 0.653646 using {'model__optimizer': 'SGD'}
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


"""
Learning Rate and Momentum Tuning:
    learning rate: 각각의 배치마다 얼마만큼 웨이트를 업데이트 할지 결정 
    momentum: 앞전의 업데이트가 현재 웨이트에 얼마만큼의 영향을 미칠지에 대한 결정
#Best: 0.666667 using {'optimizer__learning_rate': 0.001, 'optimizer__momentum': 0.0}
"""
# def create_model():
#     model = Sequential()
#     model.add(Dense(12, input_shape=(8,), activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     return model
#
# model = KerasClassifier(model=create_model, loss="binary_crossentropy", optimizer="SGD", epochs=50, batch_size=10, verbose=0)
#
# # define the grid search parameters
# learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# #optimizer__로 업데이트 값 설정
# param_grid = dict(optimizer__learning_rate=learn_rate, optimizer__momentum=momentum)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(X, Y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# #Best: 0.666667 using {'optimizer__learning_rate': 0.001, 'optimizer__momentum': 0.0}
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

"""
Weight Initialization Tuning: 
#Best: 0.669271 using {'model__init_mode': 'uniform'}
"""

# def create_model(init_mode='uniform'):
#     model = Sequential()
#     model.add(Dense(12, input_shape=(8,), kernel_initializer=init_mode, activation='relu'))
#     model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
#     optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)
#     model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
#     return model

# model = KerasClassifier(model=create_model, epochs=50, batch_size=10, verbose=0)
#
# # define the grid search parameters
# init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# param_grid = dict(model__init_mode=init_mode)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(X, Y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# #Best: 0.669271 using {'model__init_mode': 'uniform'}
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

"""
Activation Function Tuning: 
보통 activation은 relu, tanh를 쓰는 경우가 많지만 다른 것이 더 좋을 수가 있음으로 확인
"""

# def create_model(activation='relu'):
#     model = Sequential()
#     model.add(Dense(12, input_shape=(8,), kernel_initializer='uniform', activation=activation))
#     model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)
#     model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
#     return model

# model = KerasClassifier(model=create_model, epochs=50, batch_size=10, verbose=0)
#
# # define the grid search parameters
# activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# ## model__ 사용
# param_grid = dict(model__activation=activation)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=3, cv=3)
# grid_result = grid.fit(X, Y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# #Best: 0.674479 using {'model__activation': 'relu'}
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


"""
Dropout 
Best: 0.677083 using {'model__dropout_rate': 0.0, 'model__weight_constraint': 5.0}
"""
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm

# def create_model(dropout_rate, weight_constraint):
#     model = Sequential()
#     model.add(Dense(12, input_shape=(8,), kernel_initializer='uniform', activation='relu',kernel_constraint=MaxNorm(weight_constraint)))
#     model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     model.add(Dropout(dropout_rate))
#     optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)
#     model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(model=create_model, epochs=50, batch_size=10, verbose=0)
# # define the grid search parameters
# weight_constraint = [1.0, 2.0, 3.0, 4.0, 5.0]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# param_grid = dict(model__dropout_rate=dropout_rate, model__weight_constraint=weight_constraint)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(X, Y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# #Best: 0.677083 using {'model__dropout_rate': 0.0, 'model__weight_constraint': 5.0}
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


"""
Number of Neurons

이론상으로는 뉴론의 숫자가 많을 수록 정확도가 높아 지는데, 지금의 경우는 5가 가장 최적이 되었음..
"""
# def create_model(neurons):
#     model = Sequential()
#     model.add(Dense(neurons, input_shape=(8,), kernel_initializer='uniform', activation='relu',kernel_constraint=MaxNorm(5.0)))
#     model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     model.add(Dropout(0.0))
#     optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)
#     model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(model=create_model, epochs=50, batch_size=10, verbose=0)
# # define the grid search parameters
# neurons = [1, 5, 10, 15, 20, 25, 30]
# param_grid = dict(model__neurons=neurons)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(X, Y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


def create_model_final():
    model = Sequential()
    model.add(Dense(5, input_shape=(8,), kernel_initializer='uniform', activation='relu',kernel_constraint=MaxNorm(5.0)))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.add(Dropout(0.0))
    optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0)
    model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
    return model

base_model = create_model()
base_model.fit(X, Y, epochs=50, batch_size=10, verbose=0)

_, accuracy = base_model.evaluate(X, Y, verbose=0)
print('Accuracy: %.2f' % (accuracy*100)) #Accuracy: 69.79

##############
final_model = create_model_final()
final_model.fit(X, Y, epochs=50, batch_size=10, verbose=0)

_, accuracy = final_model.evaluate(X, Y, verbose=0)
print('Accuracy: %.2f' % (accuracy*100)) #Accuracy: 70.44
