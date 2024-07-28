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
y0 = [T0, I0]

# 매개변수 설정
lambda_ = T0 * 0.05  # 예시로 설정한 lambda_
d = 0.05             # 예시로 설정한 d
delta = 0.1          # 예시로 설정한 delta
c = 23               # 예시로 설정한 c

# 미분 방정식 정의
def hiv_model(y, t, lambda_, d, beta, delta, p, c):
    T, I = y
    V = p * I / (c + beta * T)  # V의 근사식

    dT_dt = lambda_ - d * T - beta * T * V
    dI_dt = beta * T * V - delta * I

    return [dT_dt, dI_dt]

# 모델 함수 정의
def model_func(t, beta, p):
    solution = odeint(hiv_model, y0, t, args=(lambda_, d, beta, delta, p, c))
    I = solution[:, 1]
    V = p * I / (c + beta * solution[:, 0])
    return V

# 최적화 (파라미터 추정)
popt, _ = curve_fit(model_func, time_data, V_data, p0=[10^-7, 200])

# 최적화된 파라미터
beta_opt, p_opt = popt
print(f"Estimated beta: {beta_opt}")
print(f"Estimated p: {p_opt}")

# 최적화된 파라미터로 시뮬레이션
t = np.linspace(0, 12, 100)
solution = odeint(hiv_model, y0, t, args=(lambda_, d, beta_opt, delta, p_opt, c))
T = solution[:, 0]
I = solution[:, 1]
V = p_opt * I / (c + beta_opt * T)

# 결과 시각화
plt.figure(figsize=(12, 8))

# 건강한 T 세포 (T) 시각화
plt.subplot(2, 1, 1)
plt.plot(t, T, label='Simulated Healthy T cells (T)')
plt.scatter(time_data, T_data, color='red', label='Observed Healthy T cells (T)', zorder=5)
plt.xlabel('Time (weeks)')
plt.ylabel('Concentration')
plt.title('Healthy T cells over time')
plt.legend()
plt.grid()

# 바이러스 (V) 시각화
plt.subplot(2, 1, 2)
plt.plot(t, V, label='Simulated Virus (V)')
plt.scatter(time_data, V_data, color='red', label='Observed Virus (V)', zorder=5)
plt.xlabel('Time (weeks)')
plt.ylabel('Concentration')
plt.title('Virus concentration over time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
