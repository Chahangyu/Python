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

np.random.rand(size)를 사용해 평균 0 분산 1의 가우스 분포로 난수를 생성할 수 있습니다
또한 np.random.randint(low, high, size)를 이용하면 low에서 high까지의 임의의 정숫값으로 이뤄진 size 크기의 행렬을 생성할 수 있습니다

------------------------------------

행렬의 크기 변경

행렬의 크기를 변경하는 경우 변수명.reshape(n, m)를 사용합니다

in a = np.arange(10)
   print(a)

out [0 1 2 3 4 5 6 7 8 9]

in a.reshape(2, 5)

out array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

------------------------------------

행렬(ndarray)의 사칙 연산

+, -, *, / 는 해당되는 요소 전체에 적용됩니다.

로그 함수의 역함수 np.array(x), 제곱근 np.sqrt(x), 로그 np.log(x), 반올림 np.round(x, 유효자릿수)
평균 np.mean(x), 표준편차 np.st-d(x), 최댓값 np.max(x), 최솟값 np.min(x) 

------------------------------------

행렬 곱의 계산

변수명1.dot(변수명2)

in v = np.array([[1, 2, 3], [4, 5, 6]])
   w = np.array([[1, 1], [2, 2], [3, 3]])
   print(v.dot(w))

out [[14 14]
     [32 32]]
     

------------------------------------

슬라이싱의 이용

in x = np.arange(10)
   print(x)
   print(x[:5])

out [0 1 2 3 4 5 6 7 8 9]
    [0 1 2 3 4]

in print(x[5:])

out [5 6 7 8 9]

in print(x[3:8])

out [3 4 5 6 7]

변수명[n1:n2:dn] n1 ~ n2 - 1 의 요소까지 dn 간격으로 참조

in print(x[3:8:2])

out [3 5 7]

in print(x[::-1])

out [9 8 7 6 5 4 3 2 1 0]

------------------------------------

bool 배열 사용

in x = np.array([1, 1, 2, 3, 5, 8, 13])
   print(x > 3)

out [False False False False  True  True  True]

in print(x[x > 3])

out [ 5 8 13]

in x[x > 3] = 999
   print(x)

out [ 1 1 2 3 999 999 999]

------------------------------------

help 사용

함수에는 많은 종류가 있고 기능 또한 다양하다
help(함수명) 은 함수의 설명을 확인하는 함수이다

in help(np.random.randint)

out randint(...) method of numpy.random.mtrand.RandomState instance
    randint(low, high=None, size=None, dtype=int)

    Return random integers from `low` (inclusive) to `high` (exclusive).

    -- More  --

------------------------------------

함수의 사용

함수는 def 함수명() : 으로 시작하여, 함수의 내용은 들여쓰기로 정의한다

in def MyFunc1() : 
      print("Hi!")

   MyFunc1()

out Hi!

------------------------------------

인수와 반환값

in def MyFunc1(D) : 
   m = np.mean(D) //D의 평균을 구하는 함수
   s = np.std(D)  //D의 표준 편차를 구하는 함수
   return m, s

   data = np.random.randn(100)
   data_mean, data_std = MyFunc1(data)
   print('mean : {0:3.2f}, std : {1:3.2f}'.format(data_mean, data_std))

out mean : 0.22, std : 0.97

반환값이 여러 개라도 하나의 변수로 받을 수 있습니다
이 경우 반환값은 tuple 형이 되고, 각 요소에 함수의 반환값이 저장됩니다
난수이므로 실행할 때마다 결과가 달라집니다

in data = np.random.randn(100)
   output = MyFunc1(data)
   print(output)
   print(type(output))
   print('mean : {0:3.2f}, std : {1:3.2f}'.format(output[0], output[1]))

out (0.18037480307984086, 0.9334758314465706)
    <class 'tuple'>
    mean : 0.18, std : 0.93

------------------------------------

하나의 ndarray 형을 저장

np.save('파일명.np', 변수명)을 사용합니다
파일 확장자는 .npy입니다
데이터를 로드하려면 np.load('파일명.npy')를 사용합니다

in data = np.random.randn(5)
   print(data)
   np.save('datafile.npy', data)    # 세이브
   data = []                        # 데이터 삭제
   print(data)
   data = np.load('datafile.npy')   # 로드
   print(data)  

out [ 1.4248736  -1.71207187  0.63063983  0.07438093  1.06553757]
    []
    [ 1.4248736  -1.71207187  0.63063983  0.07438093  1.06553757]

------------------------------------

여러 ndarray 형을 저장

np.savez('파일명.npz', 변수명1 = 변수명1, 변수명2 = 변수명2, ...)을 사용합니다

in data1 = np.array([1, 2, 3])
   data2 = np.array([10, 20, 30])
   np.savez('datafile2.npz', data1 = data1, data2 = data2)  # 세이브
   data1 = []
   data2 = []
   outfile = np.load('datafile2.npz')                       # 로드
   print(outfile.files)
   data1 = outfile['data1'] # data1 꺼내기
   data2 = outfile['data2'] # data2 꺼내기
   print(data1, data2)

out ['data1', 'data2']
    [1 2 3] [10 20 30]

데이터를 np.load로 불러오면 저장된 모든 변수가 outfile에 저장되어 outfile['변수명']으로 각각의 변수를 참조할 수 있습니다
outfile.files로 저장된 변수의 목록을 볼 수 있습니다

------------------------------------
   
임의의 그래프 그리기

in import numpy as np
   import matplotlib.pyplot as plt

   np.random.seed(1) # 난수 고정
   x = np.arange(10)
   y = np.random.rand(10)

   # 그래프 표시
   plt.plot(x, y) # 꺽은선 그래프를 등록
   plt.show()     # 그래프 그리기

out 그래프

------------------------------------

3차 함수 f(x) = (x - 2) x (x + 2) 그리기

in def f(x) : 
      return (x - 2) * x * (x + 2)

   print(f(np.array([1, 2, 3])))

out [-3  0 15]

------------------------------------

그리는 범위를 결정하기

in x = np.arange(-3, 3.5, 0.5)
   print(x)

out [-3.  -2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2.   2.5  3. ]

np.arange(-3, 3, 0.5)로 하면 2.5까지의 범위를 다루므로, 3보다 큰 np.arange(-3, 3.5, 0.5)로 설정한다

그런데 그래프의 x를 정의하는 경우에는 arange보다 linspace라는 명령이 사용하기 쉬울지도 모른다
linspace(n1, n2, n)하면 범위 n1에서 n2 사이를 일정 간격 n개의 구간으로 나눈 값을 돌려준다

in x = np.linspace(-3, 3, 10)
   print(np.round(x, 2))

out [-3.   -2.33 -1.67 -1.   -0.33  0.33  1.    1.67  2.33  3.  ]

------------------------------------

그래프 그리기 

in def f(x) : 
      return (x - 2) * x * (x + 2)

   x = np.linspace(-3, 3, 10)
   print(np.round(x, 2))

   plt.plot(x, f(x))
   plt.show()

out 그래프

------------------------------------

그래프를 장식하기

def f(x, w) : 
   return (x - w) * x * (x + 2)

x = np.linspace(-3, 3, 100)

plt.plot(x, f(x, 2), color = 'black', label = '$ w = 2 $')
plt.plot(x, f(x, 1), color = 'cornflowerblue', label = '$ w = 1 $')

plt.legend(loc = "upper left")   # 범례 표시
plt.ylim(-15, 15)                # y축 범위
plt.title('$ f(x) $')            # 제목
plt.xlabel('$ x $')              # x 라벨
plt.ylabel('$ y $')              # y 라벨
plt.grid(True)                   # 그리드

plt.show()

범례의 위치는 자동으로 정해지지만 loc을 사용하여 지정할 수 있습니다
오른쪽 위는 "upper right"
왼쪽 위는 "upper left"
왼쪽 아래는 "lower left"
오른쪽 아래는 "lower right"로 지정

label과 title의 문자열은 '$'로 묶어보기 좋은 TeX 기반의 수식으로 지정할 수 있습니다

------------------------------------

색상 목록

print(matplotlib.colors.cnames)

------------------------------------

그래프 여러 개 보여주기

plt.subplot(n1, n2, n)를 사용하여 전체를 세로 n1, 가로 n2로 나눈 n번째에 그래프가 그려집니다
plt.subplot의 n은 특별하게도 0이 아닌 1부터 시작하므로 주의, 0을 지정하면 오류가 발생한다


'''

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
print(matplotlib.colors.cnames)

# def f(x, w) : 
#    return (x - w) * x * (x + 2)

# x = np.linspace(-3, 3, 100)

# plt.plot(x, f(x, 2), color = 'black', label = '$ w = 2 $')
# plt.plot(x, f(x, 1), color = 'cornflowerblue', label = '$ w = 1 $')

# plt.legend(loc = "upper left")   # 범례 표시
# plt.ylim(-15, 15)                # y축 범위
# plt.title('$ f(x) $')            # 제목
# plt.xlabel('$ x $')              # x 라벨
# plt.ylabel('$ y $')              # y 라벨
# plt.grid(True)                   # 그리드

# plt.show()

