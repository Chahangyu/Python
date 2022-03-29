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

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def f(x, w) : 
   return (x - w) * x * (x + 2)

x = np.linspace(-3, 3, 100)

plt.figure(figsize = (10, 3)) # 전체 영역의 크기 지정
plt.subplots_adjust(wspace = 0.5, hspace = 0.5) # 그래프 간격을 지정 w = 좌우 간격, h = 상하 간격

for i in range(6) : 
   plt.subplot(2, 3, i + 1)
   plt.title(i + 1)
   plt.plot(x, f(x, i), 'k')
   plt.ylim(-20, 20)
   plt.grid(True)

plt.show()

------------------------------------

이변수 함수

f(x0, x1) = (2 x 0^2) exp (-(2 x 0^2 + x1^2))

def f(x0, x1) : 
   r = (2 * (x0 ** 2)) + (x1 ** 2)
   ans = r * np.exp(-r)
   return ans

xn = 9
x0 = np.linspace(-2, 2, xn)
x1 = np.linspace(-2, 2, xn)

y = np.zeros((len(x0), len(x1)))

for i0 in range(xn) :
   for i1 in range(xn) :
      y[i1, i0] = f(x0[i0], x1[i1])

print(x0)
print(np.round(y, 1))

[-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
[[0.  0.  0.  0.  0.1 0.  0.  0.  0. ]
 [0.  0.  0.1 0.2 0.2 0.2 0.1 0.  0. ]
 [0.  0.  0.1 0.3 0.4 0.3 0.1 0.  0. ]
 [0.  0.  0.2 0.4 0.2 0.4 0.2 0.  0. ]
 [0.  0.  0.3 0.3 0.  0.3 0.3 0.  0. ]
 [0.  0.  0.2 0.4 0.2 0.4 0.2 0.  0. ]
 [0.  0.  0.1 0.3 0.4 0.3 0.1 0.  0. ]
 [0.  0.  0.1 0.2 0.2 0.2 0.1 0.  0. ]
 [0.  0.  0.  0.  0.1 0.  0.  0.  0. ]]

------------------------------------

수치를 색으로 표현하기 : pcolor

plt.figure(figsize = (3.5, 3))
plt.gray()
plt.pcolor(y)
plt.colorbar()

plt.show()

------------------------------------

함수를 표면에 표시 : surface

from mpl_toolkits.mplot3d import Axes3D

xx0, xx1 = np.meshgrid(x0, x1) # meshgrid(메쉬 그리드) : 벡터 x 및 y에 포함된 좌표를 바탕으로 2차원 그리드 좌표를 반환합니다

plt.figure(figsize = (5, 3.5))
ax = plt.subplot(1, 1, 1, projection = '3d')
ax.plot_surface(xx0, xx1, y, rstride = 1, cstride = 1, alpha = 0.3, color = 'blue', edgecolor = 'black')
ax.set_zticks((0, 0.2))
ax.view_init(75, -95)

plt.show()


surface를 표시하는 부분은 ax.plot_surface 입니다
옵션 rstride와 cstride에 자연수를 부여해 가로 및 세로로 몇 개의 선을 긋는지 지정할 수 있습니다
수가 적을수록 선의 간격이 조밀해집니다
alpha는 면의 투명도를 지정하는 옵션입니다 0과 1 사이의 실수로 면의 투명도를 지정하며 1에 가까울수록 불투명해집니다

print(x0)
print(x1)

print(xx0)
print(xx1)

[-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]

[-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]

[[-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]
 [-2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]]

[[-2.  -2.  -2.  -2.  -2.  -2.  -2.  -2.  -2. ]
 [-1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5]
 [-1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1. ]
 [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5]
 [ 0.   0.   0.   0.   0.   0.   0.   0.   0. ]
 [ 0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5]
 [ 1.   1.   1.   1.   1.   1.   1.   1.   1. ]
 [ 1.5  1.5  1.5  1.5  1.5  1.5  1.5  1.5  1.5]
 [ 2.   2.   2.   2.   2.   2.   2.   2.   2. ]]

------------------------------------

등고이선으로 표시 : contour

xx0, xx1 = np.meshgrid(x0, x1) # meshgrid(메쉬 그리드) : 벡터 x 및 y에 포함된 좌표를 바탕으로 2차원 그리드 좌표를 반환합니다

plt.figure(1, figsize = (4, 4))

cont = plt.contour(xx0, xx1, y, 5, colos = 'black')
cont.clabel(fmt = '%3.2f', fontsize = 8)

plt.xlabel('$ x_0 $', fontsize = 14)
plt.ylabel('$ x_1 $', fontsize = 14)

plt.show()

------------------------------------

머신러닝에 필요한 수학 skip

------------------------------------

1차원 입력 직선 모델

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed = 1)
X_min = 4   # X의 하한
X_max = 30  # X의 상한
X_n = 16    # X의 상한
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170, 108, 0.2] # 생성 매개 변수
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
np.savez('ch5_data.npz', X = X, X_min = X_min, X_max = X_max, X_n = X_n, T = T)

print(np.round(X, 2), '\n')
print(np.round(T, 2))

[15.43 23.01  5.   12.56  8.67  7.31  9.66 13.64 14.92 18.47 15.48 22.13
 10.11 26.95  5.68 21.76] 

[170.91 160.68 129.   159.7  155.46 140.56 153.65 159.43 164.7  169.65
 160.71 173.29 159.31 171.52 138.96 165.87]

------------------------------------

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed = 1)
X_min = 4   # X의 하한
X_max = 30  # X의 상한
X_n = 16    # X의 상한
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170, 108, 0.2] # 생성 매개 변수
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
np.savez('ch5_data.npz', X = X, X_min = X_min, X_max = X_max, X_n = X_n, T = T)

plt.figure(figsize = (4, 4))
plt.plot(X, T, marker = 'o', linestyle = 'None', markeredgecolor = 'black', color = 'cornflowerblue')
plt.xlim(X_min, X_max)
plt.grid(True)

plt.show()

------------------------------------

제곱의 오차 함수

제곱의 오차 함수식
https://url.kr/iadxz9

yn은 직선 모델에 xn을 넣었을 때의 출력을 나타낸다
yn = y(xn) = w0xn + w1
J는 평균 제곱의 오차로 위 링크와 같이 직선과 데이터 점의 차의 제곱의 평균이다
오차의 크기가 N에 의존하지 않는 평균 제곱 오차를 사용해 진행

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mse_line(x, t, w) :
    y = w[0] * x + w[1]
    mse = np.mean((y - t) ** 2)
    return mse

np.random.seed(seed = 1)
X_min = 4   # X의 하한
X_max = 30  # X의 상한
X_n = 16    # X의 상한
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170, 108, 0.2] # 생성 매개 변수
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
np.savez('ch5_data.npz', X = X, X_min = X_min, X_max = X_max, X_n = X_n, T = T)

xn = 100
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
J = np.zeros((len(x0), len(x1)))

for i0 in range(xn) : 
    for i1 in range(xn) :
        J[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))

plt.figure(figsize = (9.5, 4))
plt.subplots_adjust(wspace = 0.5)

ax = plt.subplot(1, 2, 1, projection = '3d')
ax.plot_surface(xx0, xx1, J, rstride = 10, cstride = 10, alpha = 0.3, color = 'blue', edgecolor = 'black')
ax.set_xticks([-20, 0, 20])
ax.set_yticks([120, 140, 160])
ax.view_init(20, -60)

plt.subplot(1, 2, 2)
cont = plt.contour(xx0, xx1, J, 30, color = 'black', levels = [100, 1000, 10000, 100000])
cont.clabel(fmt = '% 1.0f', fontsize = 8)
plt.grid(True)

plt.show()

------------------------------------

w = [10, 165] 의 기울기

def dmse_line(x, t, w) :
   y = w[0] * x + w[1]
   d_w0 = 2 * np.mean((y - t) * x)
   d_w1 = 2 * np.mean(y - t)
   return d_w0, d_w1


np.random.seed(seed = 1)
X_min = 4   # X의 하한
X_max = 30  # X의 상한
X_n = 16    # X의 상한
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170, 108, 0.2] # 생성 매개 변수
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
np.savez('ch5_data.npz', X = X, X_min = X_min, X_max = X_max, X_n = X_n, T = T)

d_w = dmse_line(X, T, [10, 165])
print(np.round(d_w, 1)) # [5046.3  301.8]

------------------------------------

경사 하강법

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mse_line(x, t, w) :
    y = w[0] * x + w[1]
    mse = np.mean((y - t) ** 2)
    return mse

def dmse_line(x, t, w) :
   y = w[0] * x + w[1]
   d_w0 = 2 * np.mean((y - t) * x)
   d_w1 = 2 * np.mean(y - t)
   return d_w0, d_w1

def fit_line_num(x, t) :
   w_init = [10.0, 165.0]  # 초기 매개 변수
   alpha = 0.001  # 학습률
   i_max = 100000 # 최대 반복수
   eps = 0.1   # 반복을 종료 기울기의 절대값의 한계
   w_i = np.zeros([i_max, 2])
   w_i[0, :] = w_init
   
   for i in range(1, i_max) :
      dmse = dmse_line(x, t, w_i[i - 1])
      w_i[i, 0] = w_i[i - 1, 0] - alpha * dmse[0]
      w_i[i, 1] = w_i[i - 1, 1] - alpha * dmse[1]
      if max(np.absolute(dmse)) < eps : # 종료 판정, np.absolute는 절대값
         break

   w0 = w_i[i, 0]
   w1 = w_i[i, 1]
   w_i = w_i[:i, :]
   return w0, w1, dmse, w_i

np.random.seed(seed = 1)
X_min = 4   # X의 하한
X_max = 30  # X의 상한
X_n = 16    # X의 상한
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170, 108, 0.2] # 생성 매개 변수
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
np.savez('ch5_data.npz', X = X, X_min = X_min, X_max = X_max, X_n = X_n, T = T)

xn = 100
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
J = np.zeros((len(x0), len(x1)))

plt.figure(figsize = (4, 4))
xn = 100
w0_range = [-25, 25]
w1_range = [120, 170]

x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)

xx0, xx1 = np.meshgrid(x0, x1)
J = np.zeros((len(x0), len(x1)))

for i0 in range(xn) : 
   for i1 in range(xn) :
      J[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))

cont = plt.contour(xx0, xx1, J, 30, colors = 'black', levels = (100, 1000, 10000, 100000))
cont.clabel(fmt = '%1.0f', fontsize = 8)

plt.grid(True)

W0, W1, dMSE, W_history = fit_line_num(X, T)

print('반복 횟수 {0}'. format(W_history.shape[0]))
print('W = [{0:.6f}, {1:.6f}]'. format(W0, W1))
print('dMSE = {0:.6f}'. format(dMSE[0], dMSE[1]))
print('MSE = {0:.6f}'. format(mse_line(X, T, [W0, W1])))

plt.plot(W_history[:, 0], W_history[:, 1], '.-', color = 'gray', markersize = 10, markeredgecolor = 'cornflowerblue')
plt.show()

반복 횟수 13820
W = [1.539947, 136.176160]
dMSE = [-0.005794, 0.099991]
MSE = 49.027452

------------------------------------

경사 하강법에 의한 직선 모델 fitting 결과

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mse_line(x, t, w) :
    y = w[0] * x + w[1]
    mse = np.mean((y - t) ** 2)
    return mse

def dmse_line(x, t, w) :
   y = w[0] * x + w[1]
   d_w0 = 2 * np.mean((y - t) * x)
   d_w1 = 2 * np.mean(y - t)
   return d_w0, d_w1

def fit_line_num(x, t) :
   w_init = [10.0, 165.0]  # 초기 매개 변수
   alpha = 0.001  # 학습률
   i_max = 100000 # 최대 반복수
   eps = 0.1   # 반복을 종료 기울기의 절대값의 한계
   w_i = np.zeros([i_max, 2])
   w_i[0, :] = w_init
   
   for i in range(1, i_max) :
      dmse = dmse_line(x, t, w_i[i - 1])
      w_i[i, 0] = w_i[i - 1, 0] - alpha * dmse[0]
      w_i[i, 1] = w_i[i - 1, 1] - alpha * dmse[1]
      if max(np.absolute(dmse)) < eps : # 종료 판정, np.absolute는 절대값
         break

   w0 = w_i[i, 0]
   w1 = w_i[i, 1]
   w_i = w_i[:i, :]
   return w0, w1, dmse, w_i

def show_line(w) :
   xb = np.linspace(X_min, X_max, 100)
   y = w[0] * xb + w[1]
   plt.plot(xb, y, color = (.5, .5, .5), linewidth = 4)

np.random.seed(seed = 1)
X_min = 4   # X의 하한
X_max = 30  # X의 상한
X_n = 16    # X의 상한
X = 5 + 25 * np.random.rand(X_n)
Prm_c = [170, 108, 0.2] # 생성 매개 변수
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
np.savez('ch5_data.npz', X = X, X_min = X_min, X_max = X_max, X_n = X_n, T = T)

xn = 100
w0_range = [-25, 25]
w1_range = [120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
J = np.zeros((len(x0), len(x1)))

plt.figure(figsize = (4, 4))
xn = 100
w0_range = [-25, 25]
w1_range = [120, 170]

x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)

xx0, xx1 = np.meshgrid(x0, x1)
J = np.zeros((len(x0), len(x1)))

for i0 in range(xn) : 
   for i1 in range(xn) :
      J[i1, i0] = mse_line(X, T, (x0[i0], x1[i1]))

W0, W1, dMSE, W_history = fit_line_num(X, T)

plt.figure(figsize = (4, 4))
W = np.array([W0, W1])
mse = mse_line(X, T, W)
print("w0 = {0:.3f}, w1 = {1:.3f}". format(W0, W1))
print("SD = {0:.3f} cm". format(np.sqrt(mse)))
show_line(W)

plt.plot(X, T, marker = 'o', linestyle = 'None', color = 'cornflowerblue', markeredgecolor = 'black')
plt.xlim(X_min, X_max)
plt.grid(True)
plt.show()


# cont = plt.contour(xx0, xx1, J, 30, colors = 'black', levels = (100, 1000, 10000, 100000))
# cont.clabel(fmt = '%1.0f', fontsize = 8)

# plt.grid(True)

# W0, W1, dMSE, W_history = fit_line_num(X, T)

# print('반복 횟수 {0}'. format(W_history.shape[0]))
# print('W = [{0:.6f}, {1:.6f}]'. format(W0, W1))
# print('dMSE = [{0:.6f}, {1:.6f}]'. format(dMSE[0], dMSE[1]))
# print('MSE = {0:.6f}'. format(mse_line(X, T, [W0, W1])))

# plt.plot(W_history[:, 0], W_history[:, 1], '.-', color = 'gray', markersize = 10, markeredgecolor = 'cornflowerblue')
# plt.show()

