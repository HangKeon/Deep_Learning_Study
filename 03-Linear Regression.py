#x와 y의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들고 실행!
import tensorflow as tf

x_data=[1,2,3]
y_data=[1,2,3]

w=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.random_uniform([1],-1.0,1.0))

#name : 나중에 텐서보드 등으로 값의 변화를 추적 or 살펴보기 쉽게 하기 위해 이름을 붙음
x=tf.placeholder(tf.float32,name="x")
y=tf.placeholder(tf.float32,name="y")
print(x)
print(y)

#x와 y의 상관 관계를 분석하기 위한 가설 수식을 작성
#y=w*x+b
#w와 x가 행렬이 아니므로 tf.matmul이 아니라 기본 곱셈 기호를 사용
hypothesis=w*x+b

#손실 함수를 작성
#mean(h-y)^2 : 예측값과 실제값의 거리를 비용(손실) 함수로 정함
cost=tf.reduce_mean(tf.square(hypothesis-y))

#텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법의 최적화를 수행
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)

#비용을 최소화하는 것이 최종 목표
train_op=optimizer.minimize(cost)

#세션을 생성하고 초기화
with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      #최적화를 100번 수행
      for step in range(100):
            #sess.run을 통해 train_op와 cost 그래프를 계산
            #이 때, 가설 수식에 넣어야 할 실제값을 feed_dict를 통해 전달!
            _,cost_val=sess.run([train_op,cost],feed_dict={x:x_data,y:y_data})

            print(step,cost_val,sess.run(w),sess.run(b))

      #최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인
            print("\n===Test===")
            print("x:5,y:",sess.run(hypothesis,feed_dict={x:5}))
            print("x:2.5,y:",sess.run(hypothesis,feed_dict={x:2.5}))
