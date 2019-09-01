## **什麼是tensorflow?**  
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
