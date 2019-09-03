## **Why Keras?**
Keras是一個開放原始碼，高階深度學習程式庫，使用Python編寫，能夠運行在TensorFlow或Theano之上。  
Keras可以使用最少的程式碼，花費最少的時間，就可以建立深度學習模型，並進行訓練、評估準確率，進行預測。相對的如果使用Tensorflow，雖然可以完全控制各種深度學習模型的細節，但是需要更多程式碼，花費更多時間開發，才能達成。

## **Keras安裝**
首先確認已安裝了Numpy和Scipy  
叫出terminal  
#如果是python2.X  
$pip install keras  
#如果是python3.X  
$pip3 install keras  

## **Backend後端**
就是Keras是基於甚麼來做運算，主要有兩個Tensorflow和Theano
### 如何知道自己用的是甚麼backend
import keras
### 如何更改backend
#### Method1
打開terminal  
輸入~/.keras/keras.json  
將文件內容重新貼上以下內容
    
    {
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
    }  
    
不能直接更改，要整段內容在編輯器事先複製下來再貼上。
#### Method2
在每次開始前先輸入  	
        
        import os
        os.environ['KERAS_BACKEND']='theano'

## **Keras model建構概念**
Define Network(定義) -> Compile Network(編譯) ->Fit Network(訓練) -> Evalute Network(評估) -> Make Predictions(預測)

## **用Keras建立一個回歸的神經網路**
	import numpy as np
	np.random.seed(1337)  # for reproducibility
	from keras.models import Sequential   #按順序(一層一層)建立的神經網路
	from keras.layers import Dense   #全連接層
	import matplotlib.pyplot as plt

	# create some data
	X = np.linspace(-1, 1, 200)
	np.random.shuffle(X)    # randomize the data
	Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
	# plot data
	plt.scatter(X, Y)
	plt.show()

	X_train, Y_train = X[:160], Y[:160]     # first 160 data points
	X_test, Y_test = X[160:], Y[160:]       # last 40 data points

	# build a neural network from the 1st layer to the last layer
	#define model
	model = Sequential()    #先用Sequential()建立Model

	model.add(Dense(units=1, input_dim=1))  #再用model.add()建立全連結層

	#compile model
	# choose loss function and optimizing method
	model.compile(loss='mse', optimizer='sgd')

	#fit model
	#training
	print('Training -----------')
	for step in range(301):
    		cost = model.train_on_batch(X_train, Y_train)
    		if step % 100 == 0:
        		print('train cost: ', cost)

	#evalution model
	# test
	print('\nTesting ------------')
	cost = model.evaluate(X_test, Y_test, batch_size=40)
	print('test cost:', cost)
	W, b = model.layers[0].get_weights()
	print('Weights=', W, '\nbiases=', b)

	#prediction model
	# plotting the prediction
	Y_pred = model.predict(X_test)
	plt.scatter(X_test, Y_test)
	plt.plot(X_test, Y_pred)
	plt.show()
