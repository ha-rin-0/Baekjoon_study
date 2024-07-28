import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# 주어진 데이터
T_data = np.array([1567, 609, 399, 175, 536, 469, 448, 656, 336])
V_data = np.array([1.00E+00, 7.57E+06, 8.96E+06, 5.89E+05, 8.32E+05, 2.75E+05, 9.84E+04, 2.25E+05, 3.14E+04])
time_data = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12])  # 주차 데이터

# 초기 조건 설정
T0 = T_data[0]  # 초기 타겟 세포 수
I0 = 0          # 초기 감염 세포 수
V0 = V_data[0]  # 초기 바이러스 입자 수
y0 = [T0, I0, V0]

# 매개변수 설정
lambda_ = T0 * 0.05  # 예시로 설정한 lambda_
d = 0.05             # 예시로 설정한 d
delta = 0.01          # 예시로 설정한 delta
c = 23               # 예시로 설정한 c

# beta' 계산
beta = 1e-7          # 예시로 설정한 beta
beta_prime = lambda_ * beta

# 새로운 미분방정식 정의
def hiv_model_new(y, t, d, beta_prime, delta, p, c):
    x, y_var, z = y
    dx_dt = d - d * x - (beta_prime * p * x * z) / (c * d)
    dy_dt = beta_prime * p * x * z - delta * y_var
    dz_dt = c * y_var - c * z - (beta_prime * x * z) / d
    return [dx_dt, dy_dt, dz_dt]

# 모델 함수 정의
def model_func(t, beta_prime, p):
    solution = odeint(hiv_model_new, y0, t, args=(d, beta_prime, delta, p, c))
    return solution[:, 2]

# 최적화 (파라미터 추정)
popt, _ = curve_fit(model_func, time_data, V_data, p0=[beta_prime, 200])

# 최적화된 파라미터
beta_prime_opt, p_opt = popt
print(f"Estimated beta': {beta_prime_opt}")
print(f"Estimated p: {p_opt}")

# 최적화된 파라미터로 시뮬레이션
t = np.linspace(0, 12, 200)
solution = odeint(hiv_model_new, y0, t, args=(d, beta_prime_opt, delta, p_opt, c))
x = solution[:, 0]
y_var = solution[:, 1]
z = solution[:, 2]

# 결과 시각화
plt.figure(figsize=(12, 8))

# 건강한 T 세포와 바이러스 농도를 한 그래프에 표시
plt.plot(t, x, label='Simulated Healthy T cells (x)')
plt.scatter(time_data, T_data, color='red', label='Observed Healthy T cells (x)', zorder=5)
plt.plot(t, z, label='Simulated Virus (z)')
plt.scatter(time_data, V_data, color='blue', label='Observed Virus (z)', zorder=5)

plt.xlabel('Time (weeks)')
plt.ylabel('Concentration')
plt.title('Healthy T cells and Virus concentration over time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
