''' 
자료형

int     정수
float   실수
str     문자열
bool    참과 거짓
list    배열
tuple   배열(수정 불가능)
ndarray 행렬

type(조사하고 픈 자료형) 자료형이 나옴

------------------------------------

문자열 수치 함께 표시

print('A = {0} kg B = {1} kg'. format(x, y))
-> A = x kg B = y kg

------------------------------------

list 길이

len(list name)

------------------------------------

연속된 정수 데이터 작성

range(Start Number, End Number + 1)

ex) 2 ~ 5
    ran = range(2, 6)

range 형은 list 형과 같은 방법으로 요소를 참조할 수 있지만 요소를 수정할 수는 없다
range에 y[2] = 2 라는 명령을 내리면 오류가 난다
시작 숫자를 생략하고 range(EndNumber + 1)를 입력하면 0부터 시작되는 수열이 만들어진다

------------------------------------

tuple의 이용

a = (1, 2, 3)

참조는 list 형과 같은 방식을 사용

길이가 1인 tuple은 a = (1, ) 와 같이 사용한다

------------------------------------

if, else 조건문 이용

if x > 10 :

else :
와 같이 사용

조건을 동시에 사용해야할 경우 and와 or을 사용한다

------------------------------------

enumerate의 이용

num = [2, 4, 6, 8, 10]
for i in range(len(num)) :
    num[i] *= 2
print(num)

num = [2, 4, 6, 8, 10]
for i in enumerate(num) :
    num[i] *= 2
print(num)

------------------------------------

vector의 이용

in [1, 2] + [3, 4]

out [1, 2, 3, 4]

------------------------------------

NumPy의 이용

import numpy as np

as np 부분은 NumPy를 np로 생략해서 사용한다는 의미 np 대신 npy를 입력해도 됩니다
별명은 사용자가 원하는 대로 결정할 수 있습니다

------------------------------------

Vector의 정의

Vector(1차원 배열)은 np.array(list 형)으로 정의합니다

in x = np array([1, 2, 3])
   x

out array([1, 2, 3])

print(x)를 통해 x를 표시하면 요소 사이의 ','가 생략되어 깔끔하게 보인다

out [1 2 3]

참고로 np.array 형이 아니라 list 형을 print 하면 요소 간에 ','이 남는다

in y = np.array([4, 5, 6])
   print(x + y)

out [5 7 9]

in type(x)

out numpy.ndarray

------------------------------------

연속되는 정수 백터의 생성

in print(np.arange(10))

out [0 1 2 3 4 5 6 7 9]

------------------------------------

ndarray 형의 주의점 

ndarray 형의 내용을 복사하려면 b = a가 아니라 b = a.copy()를 사용해야한다

b = a 는 c++의 *b = &a 와 같다

------------------------------------

행렬의 정의

in x = np.array([[1, 2, 3], [4, 5, 6]])

out [[1 2 3]
     [4 5 6]]

------------------------------------

행렬 크기

in x = np.array([[1, 2, 3], [4, 5, 6]])
   x.shape

out (2, 3)

위 출력은 ()로 둘러싸여 있으므로 tuple 형으로 나타납니다

in w, h = x.shape
   print(w , h)

out 2 3

------------------------------------

요소가 0과 1인 ndarray 만들기

in print(np.zeros(10))

out [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

in print(np.ones((2, 10)))

out [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]


------------------------------------

요소가 랜덤인 행렬 생성

in print(np.random.rand(2, 3))

out [[0.68238015 0.18823192 0.06178311]
    [0.44144063 0.52308275 0.43164664]]

'''

import numpy as np

print(np.random.rand(100 , 200, 5))
