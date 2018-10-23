#플레이스홀더 & 변수의 개념 익히기
import tensorflow as tf

#tf.placeholder : 계산을 실행할 때 입력값을 받는 변수
#None 은 크기가 정해지지 않았음을 의미!
X=tf.placeholder(tf.float32,[None,3])
print(X)

# X 플레이스홀더에 넣을 값이다.
# 플레이스홀더에서 설정한 것처럼, 두 번째 차원의 요소의 갯수는 3개이다.
x_data=[[1,2,3],[4,5,6]]

#tf.Variable : 그래프를 계산하면서 최적화 할 변수! 이 값이 바로 신경망을 좌우하는 값
#tf.random_normal : 각 변수들의 초기값을 정규분포 랜덤 값으로 초기화
w=tf.Variable(tf.random_normal([3,2]))
b=tf.Variable(tf.random_normal([2,1]))

#입력값과 변수들을 계산할 수식을 작성!
#tf.matmul 처럼 mat*로 되어 있는 함수로 행렬 계산 수행!
expr=tf.matmul(X,w)+b

sess=tf.Session()
#위에서 설정한 Variable 들의 값들을 최기화 하기 위해
# 처음에 tf.global_variable_initializer를 한 번 실행해야 한다.
sess.run(tf.global_variables_initializer())

print("===x_data===")
print(x_data)
print("===w===")
print(sess.run(w))
print("===b===")
print(sess.run(b))
print("===expr===")
#expr 수식에는 X라는 값이 필요
#따라서 expr 실행시에는 이 변수에 대한 실제 입력값을 다음처럼 넣어줘야 한다.
print(sess.run(expr,feed_dict={X:x_data}))

sess.close()
