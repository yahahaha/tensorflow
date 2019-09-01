## **什麼是Tensorflow?**  
TensorFlow 是 Google透過使用資料流 (flow) 圖像，來進行數值演算的新一代開源機器學習工具。

## **Tensorflow安裝**  
### MacOS/Linux安裝  
用pip安裝(需先確認電腦中已安裝了pip。若電腦中已安裝了python3.X，因為pip已經自帶在python的模組裡，因此pip也已安裝了)  
#### CPU版本  
開啟terminal    
#如果安裝的是python2.X  
$pip install tensorflow  
#如果安裝的是python3.X  
$pip3 install tensorflow  
### Windows安裝  
支持python3.5(64bit)版本  
開啟command  
c:\>pip install tensorflow  

## **tensorflow數據流圖**
![image](img.gif)

## **實例**
	import tensorflow as tf
	import numpy as np

	#create data
	x_data=np.random.rand(100).astype(np.float32)     #生成100個隨機數列，在tensorflow中大部分的數據的type是float32的形式
	y_data=x_data*0.1+0.3

	#create tensorflow structure start
	Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))  #tf.random_uniform(結構,左範圍,右範圍)，初始值是-1~1的數
	biases=tf.Variable(tf.zeros([1]))   #初始值是0

	y=Weights*x_data+biases

	loss=tf.reduce_mean(tf.square(y-y_data))   #誤差，預測的y和實際的y_data的差別
	optimizer=tf.train.GradientDescentOptimizer(0.5)   #利用optimizer減少誤差，GradientDescentOptimizer(學習效率)，學習效率<1
	train=optimizer.minimize(loss)

	init=tf.initiallize_all_variables()    #初始化變量
	#create tensorflow structure end

	sess=tf.Session()
	sess.run(init)         #very important

	for step in range(201):
		session.run(train)
		if step%20==0:
			print(step,sess.run(Weights),sess.run(biases))
## **Session**
Tensorflow是基於圖架構進行運算的深度學習框架，Session是圖和執行者之間的媒介，首先透過Session來啟動圖，而Session.run()是用來進行操作的，Session再使用完過後需要透過close來釋放資源，或是透過with as的方式來讓他自動釋放。
	
	import tensorflow as tf

	matrix1=tf.constant([[3,3]])         #Constant就是不可變的常數
	matrix2=tf.constant([[2],[2]])
	product=tf.matmul(matrix1,matrix2)

	#method 1
	session=tf.Session()
	result=sess.run(product)
	print(result)
	sess.close()

	#method 2
	with tf.Session() as sess:
		result2= sess.run(product)
		print(result2)
## **Variable**
將值宣告賦值給變數（Variables）讓使用者能夠動態地進行相同的計算來得到不同的結果，在TensorFlow中是以tf.Variable()來完成。  
在TensorFlow的觀念之中，宣告變數張量並不如Python那麼單純，它需要兩個步驟：
1.宣告變數張量的初始值、類型與外觀 
2.初始化變數張量
