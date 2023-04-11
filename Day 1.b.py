"""
Regression Loss 함수들

    Regression Loss 함수들
        Mean Squared Error Loss
        Mean Squared Logarithmic Error Loss
        Mean Absolute Error Loss
    Binary Classification Loss 함수들
        Binary Cross-Entropy
        Hinge Loss
        Squared Hinge Loss
    Multi-Class Classification Loss 함수들
        Multi-Class Cross-Entropy Loss
        Sparse Multiclass Cross-Entropy Loss
        Kullback Leibler Divergence Loss

"""

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
#from keras.metrics import mean_squared_error
from keras.optimizers import SGD
from matplotlib import pyplot
from numpy import where

## Generate data from simple regression problem, data containing 20 inputs: 10 of them are meaningful and the rest are not.
x, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

## Standardize :
"""
실제 문제를 푸는데 있어서는 보통 트레이닝 데이터에만 스케일을 시키고 난 후 트레이닝 세트와 테스테 세트에 적용시키지만, 귀찮은 관계로 트레이닝과 테스트 세트를 나누기 전에 그냥 한꺼번에 시켜버림
"""
x = StandardScaler().fit_transform(x)
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]


## Split train and test
n_train = 500
trainx, testx = x[:n_train, :], x[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]


## Defining model:
"""
1 hidden layer with 25 nodes.
특정 예측값을 푸는 문제임으로  output layer는 1 node만 갖는다. 
"""
def dynamic_model_compiler(loss, metrics):
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))

    ## Compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    if metrics:
        model.compile(loss=loss, optimizer=opt, metrics=[metrics])
    else:
        model.compile(loss=loss, optimizer=opt)

    history = model.fit(trainx,
                        trainy,
                        validation_data=(testx, testy),
                        epochs=100,
                        verbose=0
                        )
    # history_dict = history.history
    # print(history_dict.keys())

    return model, history

def generate_plot(model, history, model_type):
    _, train_mse = model.evaluate(trainx, trainy, verbose=0)
    _, test_mse = model.evaluate(testx, testy, verbose=0)
    #model_type = model_type
    print('Loss Type: %s \n Train: %.3f, Test: %.3f' % (model_type, train_mse, test_mse))

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title(f'{model_type} Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot mse during training
    pyplot.subplot(212)
    pyplot.title(f'{model_type} Mean Squared Error')
    pyplot.plot(history.history['mse'], label='train')
    pyplot.plot(history.history['val_mse'], label='test')
    pyplot.legend()
    pyplot.draw()
    pyplot.ioff()
    pyplot.show(block = False)
    pyplot.pause(1)
    pyplot.close()

### MSE
model, history = dynamic_model_compiler(loss='mean_squared_error',
                                        metrics=None
                                        )
## Evaluate the model
train_mse = model.evaluate(trainx,trainy, verbose=0)
test_mse = model.evaluate(testx, testy, verbose=0)
model_type = 'MSE'
print('Loss Type: %s \n Train: %.3f, Test: %.3f' % (model_type ,train_mse, test_mse))

## plot loss during training
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.ioff()
pyplot.show(block = False)
pyplot.pause(1)
pyplot.close()



### MSLE
"""
아웃풋 값의 분산이 큰 경우 mse를 바로 적용하게 되면 모델의 웨이트가 크게 요동치게 된다. 그걸 피하기 위해선, 우선 모든 예측값에 로그를 값을 구하고 그에 mse를 씌워서 하는 것이 유리하다. 
그 방법이 Mean Squared Logarithmic Error loss, or MSLE임

보통, 모델이 스케일링 되지 않은 양을 직접 측정할때 많이 쓰임. 
"""
model, history = dynamic_model_compiler(loss ='mean_squared_logarithmic_error',
                                        metrics='mse')
generate_plot(model, history, model_type = 'MSLE')


### MAE:
"""
크거나 또는 작은 아웃라이어들이 많은 경우에 보통 쓰임. 
"""
model, history = dynamic_model_compiler(loss='mean_absolute_error',
                                        metrics=['mse'])
generate_plot(model, history, model_type = 'MAE')


## Binary Cross Entropy Loss
from sklearn.datasets import make_circles
from numpy import where

x, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
for i in range(2):
 samples_ix = where(y == i)
 pyplot.scatter(x[samples_ix, 0], x[samples_ix, 1], label=str(i))
pyplot.legend()
pyplot.show(block = False)
pyplot.pause(1)
pyplot.close()

n_train = 500
trainx, testx = x[:n_train, :], x[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]


def dynamic_model_complier_for_binary(activation, loss):
    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation=activation))

    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    history = model.fit(trainx, trainy,
                        validation_data=(testx, testy),
                        epochs=200,
                        verbose=0)
    return model, history

def generate_plots_for_binary(model, history, model_type):
    _, train_acc = model.evaluate(trainx, trainy, verbose=0)
    _, test_acc = model.evaluate(testx, testy, verbose=0)
    print('Accuracy Type: %s\nTrain: %.3f, Test: %.3f' % (model_type,train_acc, test_acc))

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title(f'{model_type} Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title(f'{model_type} Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.draw()
    pyplot.show(block=False)
    pyplot.pause(1)
    pyplot.close()

"""
Cross-entropy는 binary classification 문제들의 디폴트 값.
binary classification함은 아웃풋 값이 {0, 1}인 경우를 뜻한다
"""
model, history = dynamic_model_complier_for_binary(activation='sigmoid',
                                                   loss = 'binary_crossentropy')

generate_plots_for_binary(model= model, history= history, model_type = 'Binary Crossentropy')

## HINGE LOSS:
"""
Support Vector Machine (SVM) models을 위해서 만들어진 함수. binary crossentropy와 다른 점은 아웃풋 값이 {0, 1}이 아닌 {-1, 1}를 갖는다는 점
"""
# change y from {0,1} to {-1,1}
y[where(y == 0)] = -1
model, history = dynamic_model_complier_for_binary(activation='tanh',## {-1, 1}에 따라서 이것 또한 업데이트 됨!
                                                   loss= 'hinge')

generate_plots_for_binary(model= model, history= history, model_type = 'Hinge Loss')

## Squared Hinge Loss:
"""
hinge loss함수의 제곱 값인데, 이의 장점은 에러값들이 hinge함수에 비해 보다 수치화 하기 쉽다는 점이 있다. 
hinge함수를 썼는데 결과치가 만족스럽지 않다면 squared hinge loss를 써보는 것도 나쁘지 않다
"""
model, history = dynamic_model_complier_for_binary(activation='tanh',
                                                   loss= 'squared_hinge')

generate_plots_for_binary(model= model, history= history, model_type = 'Squared Hinge Loss')


### MULTI CLASS Classification Loss Function
from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
for i in range(3):
 samples_ix = where(y == i)
 pyplot.scatter(x[samples_ix, 0], x[samples_ix, 1])
pyplot.show(block = False)
pyplot.pause(1)
pyplot.close()

from keras.utils import to_categorical
y = to_categorical(y)## Cross-entropy를 위해서 필요함

n_train = 500
trainx, testx = x[:n_train, :], x[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]



def dynamic_model_compiler_for_multiclass(activation,loss):
    model = Sequential()
    model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(3, activation=activation))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    # fit model
    history = model.fit(trainx, trainy,
                        validation_data=(testx, testy),
                        epochs=100, verbose=0)

    return model, history

def generate_plot_for_multiclass(model, history, model_type):
    _, train_acc = model.evaluate(trainx, trainy, verbose=0)
    _, test_acc = model.evaluate(testx, testy, verbose=0)
    print('Loss Type: %s\n Train: %.3f, Test: %.3f' % (model_type, train_acc, test_acc))
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title(f'{model_type} Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title(f'{model_type} Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show(block=False)
    pyplot.pause(1)
    pyplot.close()
"""
Cross-entropy가 디폴트 값임.
multi-class classification, 즉 아웃풋 값이 {0, 1, 3, …, n} 각각의 숫자에 대응하는 꼴.
Cross-entropy는 모든 클레스에 대하여 예측값과 실제값의 차이에 대한 평균을 계산한고, 0에 아까울수록 좋다

categorical_crossentropy의 경우 아웃풋 값이 n nodes (각각 하나씩)있어야 한다. 즉, 아웃풋 값은 반드시 one hot encoded가 되어야한다. 
또한 각각의 클레스별 확률을 구하기 위해서 'softmax'를 사용
"""

model, history = dynamic_model_compiler_for_multiclass(activation='softmax',
                                                       loss='categorical_crossentropy')

generate_plot_for_multiclass(model, history, model_type='categorical_crossentropy')


## Sparse Multiclass Crossentropy
"""
앞서 이야기 한것처럼, cross-entropy를 multi-class classification에 대입할 경우, 아웃풋 값을 one hot encode해야 해서 메모리를 많이 잡아 먹는다. 
예를들어, 특정 단어를 예측하느데 있어 수백 수 천개의 카타고리가 있을 수 있는데 이는 개인 피씨를 통해서 풀기 어려워 질 수가 있다. 
이때 쓸 수 있는게바로, Sparse cross-entropy임! 

"""

x, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)

n_train = 500
trainx, testx = x[:n_train, :], x[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
model, history = dynamic_model_compiler_for_multiclass(activation='softmax',
                                                       loss = 'sparse_categorical_crossentropy')
generate_plot_for_multiclass(model, history, model_type='Sparse Multiclass Crossentropy')


## Kullback Leibler Divergence Loss
"""
Kullback Leibler Divergence, or KL Divergence for short,는 확률분산이 베이스라인에서 얼마나 다른지를 측정한다.
loss== 0의 경우 분포가 일치한다는 말임. cross-entropy와 매우 유사함. 
일반적으로 단순 multi-class classification보다 컴플랙스한 문제를 풀때 쓰인다.
확률이 높을수록 → 매우 당연하게 일어날 사건
확률이 낮으면  → 자주 일어나지 않는 특별한 사건

such as in the case of an autoencoder used for learning a dense feature representation under a model that must reconstruct the original input. 
In this case, KL divergence loss would be preferred. Nevertheless, it can be used for multi-class classification, in which case it is functionally equivalent to multi-class cross-entropy.
"""
y = to_categorical(y)## 다시 한번 적용

n_train = 500
trainx, testx = x[:n_train, :], x[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

#kullback_leibler_divergence
model, history = dynamic_model_compiler_for_multiclass(activation='softmax',
                                      loss = 'kullback_leibler_divergence')

generate_plot_for_multiclass(model, history, 'kullback_leibler_divergence')