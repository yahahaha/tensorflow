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

## **Classifier分類(mnist手寫字識別)**		
	import numpy as np
	np.random.seed(1337)  # for reproducibility
	from keras.datasets import mnist
	from keras.utils import np_utils
	from keras.models import Sequential
	from keras.layers import Dense, Activation
	from keras.optimizers import RMSprop

	# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
	# X shape (60,000 28x28), y shape (10,000, )
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# data pre-processing
	X_train = X_train.reshape(X_train.shape[0], -1) / 255   # normalize
	X_test = X_test.reshape(X_test.shape[0], -1) / 255     # normalize
	y_train = np_utils.to_categorical(y_train, num_classes=10)
	y_test = np_utils.to_categorical(y_test, num_classes=10)

	# Another way to build your neural net
	model = Sequential([Dense(32, input_dim=784),Activation('relu'),Dense(10),Activation('softmax'),])

	# Another way to define your optimizer
	rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

	# We add metrics to get more results you want to see
	model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])

	print('Training ------------')
	# Another way to train the model
	model.fit(X_train, y_train, epochs=2, batch_size=32)

	print('\nTesting ------------')
	# Evaluate the model with the metrics we defined earlier
	loss, accuracy = model.evaluate(X_test, y_test)

	print('test loss: ', loss)
	print('test accuracy: ', accuracy)

## **CNN卷積神經網路(Convolutional Neural Networks)**

	import numpy as np
	np.random.seed(1337)  # for reproducibility
	from keras.datasets import mnist
	from keras.utils import np_utils
	from keras.models import Sequential
	from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
	from keras.optimizers import Adam

	# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
	# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	# data pre-processing
	X_train = X_train.reshape(-1, 1,28, 28)/255.
	X_test = X_test.reshape(-1, 1,28, 28)/255.
	y_train = np_utils.to_categorical(y_train, num_classes=10)
	y_test = np_utils.to_categorical(y_test, num_classes=10)

	# Another way to build your CNN
	model = Sequential()

	# Conv layer 1 output shape (32, 28, 28)
	model.add(Convolution2D(batch_input_shape=(None, 1, 28, 28),filters=32,kernel_size=5,strides=1,padding='same',data_format='channels_first',))
	model.add(Activation('relu'))

	# Pooling layer 1 (max pooling) output shape (32, 14, 14)
	model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first',))

	# Conv layer 2 output shape (64, 14, 14)
	model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
	model.add(Activation('relu'))

	# Pooling layer 2 (max pooling) output shape (64, 7, 7)
	model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

	# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))

	# Fully connected layer 2 to shape (10) for 10 classes
	model.add(Dense(10))
	model.add(Activation('softmax'))

	# Another way to define your optimizer
	adam = Adam(lr=1e-4)

	# We add metrics to get more results you want to see
	model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

	print('Training ------------')
	# Another way to train the model
	model.fit(X_train, y_train, epochs=1, batch_size=64,)

	print('\nTesting ------------')
	# Evaluate the model with the metrics we defined earlier
	loss, accuracy = model.evaluate(X_test, y_test)

	print('\ntest loss: ', loss)
	print('\ntest accuracy: ', accuracy)

## **循環神經網路(Recurrent Neural Network, RNN)&長短期記憶模型（LSTM）**
對於語言的理解，我們在閱讀一篇文章的時候，通常不是逐字逐句分開了解，而是有辦法從過往知識或是文章的上下文來理解文意。在機器學習模型的發展中，	引入這種遞歸 (recurrent) 的概念，是遞歸神經網路與其他神經網路模型 (如 CNN) 相比，較為創新的地方。  
長短期記憶模型則是改善了遞歸神經網路在長期記憶上的一些不足，因為單純的RNN因為無法處理隨著遞歸，權重指數級爆炸或梯度消失問題，難以捕捉長期時間關聯，而結合不同的LSTM可以很好解決這個問題。  
現在已大量運用在自然語言理解 (例如語音轉文字，翻譯，產生手寫文字)，圖像與影像辨識等應用。
### 圖解長短期記憶模型 (LSTM)
我們可以將遞迴神經網路想像成相似的單元，不斷地將過往資訊往下傳遞，所以我們可以用過往的資料來預測或瞭解現在的現象。這樣的鏈結可以透過下圖來表示：  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E5%9C%96%E8%A7%A3%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B%20(LSTM).png)
X 代表輸入的資料，h 為輸出，昨天預測的結果可作爲今天預測的資料，形成一條長鏈。    
但實際上，RNN 在長期記憶的表現並不如預期。    
而 LSTMs 就是設計用來改善 RNN 在長期記憶的不足。RNN 雖然構成一個龐大的神經網路，但如果我們將標準的 RNN 內部的單元放大來，裡面是個相當單純的架構，通常只有一層，包含一個稱為激勵函數的方程式，這個方程式有幾種選擇，這邊用雙曲函數 tanh 表示：  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E5%9C%96%E8%A7%A3%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B%20(LSTM)2.png)
LSTMs 也是由不斷重複的單元彼此相連所構成，但內部的單元設計比較複雜，有四層 (黃色長方形)，彼此之間會交互作用，如下圖示：  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E5%9C%96%E8%A7%A3%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B%20(LSTM)3.png)
在討論那四層的單元如何互相影響之前，先介紹 LSTMs 最為核心的一個概念：透過閘門來做調控。下圖由左至右的水平線用來表示單元狀態 (cell state)，可以將它想像成是一個貫穿所有單元的一條道路，將資訊從一個單元帶到下一個單元 ，  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E5%9C%96%E8%A7%A3%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B%20(LSTM)4.png)  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E5%9C%96%E8%A7%A3%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B%20(LSTM)5.png)
所以第一步是，決定哪些資訊要忘掉，透過這個黃色的邏輯函數我們可以決定從上一層的輸出帶進來的資訊，以及這一層新增加的資訊，有多少比例要被忘掉，不被帶進下一層。  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E5%9C%96%E8%A7%A3%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B%20(LSTM)6.png)
下一步是將新帶進來的資料紀錄到主要的單元狀態中。這邊又分成兩步驟：決定什麼要被記錄下來，以及更新我們的主要單元。  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E5%9C%96%E8%A7%A3%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B%20(LSTM)7.png)
這個第二步驟用來解釋主要單元被更新，從數學公式可以看出，這邊也考慮了上一個遺忘步驟的結果 ft  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E5%9C%96%E8%A7%A3%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B%20(LSTM)8.png)		
最後，我們決定有多少資訊要被輸出。從上一步驟的結果 Ct，再透過邏輯函數以及 tanh 的調控，來決定主要單元上要被輸出，用作明天預測的資料是什麼。  
![image](https://github.com/yahahaha/tensorflow/blob/master/img/%E5%9C%96%E8%A7%A3%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B%20(LSTM)9.png)

參考資料:https://medium.com/@tengyuanchang/%E6%B7%BA%E8%AB%87%E9%81%9E%E6%AD%B8%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-rnn-%E8%88%87%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B-lstm-300cbe5efcc3

