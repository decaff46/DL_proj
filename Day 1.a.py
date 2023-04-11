"""
단순 딥 러닝 모델 테스트
"""

from datetime import date
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# today = date.today()
# print("Today's date:", today)
"""
Input Variables (X):

    Number of times pregnant
    Plasma glucose concentration at 2 hours in an oral glucose tolerance test
    Diastolic blood pressure (mm Hg)
    Triceps skin fold thickness (mm)
    2-hour serum insulin (mu U/ml)
    Body mass index (weight in kg/(height in m)^2)
    Diabetes pedigree function
    Age (years)

Output Variables (y):

    Class variable (0 or 1)
"""
dat = loadtxt('data/pima-indians-diabetes.csv', delimiter=',')
x = dat[:,0:8]
y = dat[:,8]

## Defining model:
"""
첫번째 레이어에서 input_shape 대한 옵션: 
input_shape을 명시 하지 않을 경우, 모델은 트레이닝/이벨류에이션을 선언하기 전까지는 특정한 무게를 갖지 않는다. 
input_shape을 명시 한 경우, 모델은 지속적으로 특정한 무게를 갖고 빌드업이 된다.
"""
model = Sequential()
model.add(Dense(12, input_shape = (8,), activation = 'relu'))
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

## Compiling model with loss, optimizer, and metrics:
"""
모델의 아웃풋이 바이너리 (1 | 0)이기에 loss = binary_crossentropy는 선언을 한다. 비슷하게, 클레서피케이션의 문제를 다루는 것이기에 이에 따른 metrics = 'accuracy'를 선언한다.
optimizer 경우, stochastic gradient descent algorithm 중 하나인 “adam“을 선택함.
adam은 자동적으로 튜닝을 함에 따른 최적의 결과치를 내놓기때문에 최근 그레이디언 디센딩 중 가장 핫함!
"""
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## Training model:
"""
    Epoch: 쉽게 생각해서 몇번의 트레이닝을 하고 싶은가에 대한 파라미터
    Batch: 한개 이상의 샘플을 한번에 트레이닝에 넣고 돌릴지에 대한 파라미터. 이때 트레이닝은 같은 웨이트를 갖게 되고 지속적으로 업데이트가 된다
"""
model.fit(x, y, epochs=150, batch_size=10, verbose=0)

## Evaluating model performance, accuracy in this case
_, accuracy = model.evaluate(x, y, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

#predictions = model.predict(x)
predictions = (model.predict(x) > 0.5).astype(int)
rounded = [round(x[0]) for x in predictions]

for i in range(5):
 print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))