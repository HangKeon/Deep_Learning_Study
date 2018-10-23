#We learn the basic ingredients of tensorflow
import tensorflow as tf

#tf.constant : 말 그대로 상수를 의미
hello=tf.constant('Hello, TensorFlow!')
print(hello)

a=tf.constant(10)
b=tf.constant(32)
c=tf.add(a,b)     #a+b로도 쓸 수 있음.
print(c)

#위에서 변수 & 수식들을 정의했지만, 실행이 정의한 시점에서 실행되는 건 아님.
#다음처럼 Session 객체와 run 메소드를 사용할 때 계산됨.
#따라서 모델을 구성하는 것과, 실행하는 것을 분리하여 프로그램을 깔끔하게 분리!
#그래프를 실행할 세션을 구성.
sess=tf.Session()
#sess.run : 설정한 텐서 그래프(변수 or 수식 등등)를 실행!
print(sess.run(hello))
print(sess.run([a,b,c]))

#세션을 닫는다.
sess.close()
