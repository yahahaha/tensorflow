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
