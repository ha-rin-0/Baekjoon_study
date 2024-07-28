import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# 주어진 데이터
T_data = np.array([1567000, 609000, 399000, 175000, 536000, 469000, 448000, 656000, 336000])
V_data = np.array([1.00E+00, 7.57E+06, 8.96E+06, 5.89E+05, 8.32E+05, 2.75E+05, 9.84E+04, 2.25E+05, 3.14E+04])
time_data = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12])  # 주차 데이터

# 초기 조건 설정
T0 = T_data[0]  # 초기 타겟 세포 수
I0 = 0          # 초기 감염 세포 수
V0 = 200  # 초기 바이러스 입자 수
y0 = [T0, I0, V0]

# 새로운 미분방정식 정의
def hiv_model(y, t, d, beta_prime, delta, p, c):
    x, y_var, z = y
    dx_dt = d - d * x - (beta_prime * p * x * z) / (c * d)
    dy_dt = (beta_prime * p * x * z) / (c * d) - delta * y_var
    dz_dt = c * y_var - c * z - (beta_prime * x * z) / d
    return [dx_dt, dy_dt, dz_dt]

# 모델 함수 정의
def model_func(t, beta_prime, p):
    d = 0.05     # 고정된 값
    delta = 0.1  # 고정된 값
    c = 23       # 고정된 값
    
    solution = odeint(hiv_model, y0, t, args=(d, beta_prime, delta, p, c))
    return np.concatenate((solution[:, 0], solution[:, 2]))  # T값과 V값 반환

# 초기 추정값 설정
beta_prime_guess = 1e-7  # 초기 추정값 설정
p_guess = 200            # 초기 추정값 설정
initial_guesses = [beta_prime_guess, p_guess]

# curve_fit을 사용하여 파라미터 추정
params, _ = curve_fit(model_func, time_data, np.concatenate((T_data, V_data)), p0=initial_guesses)

# 추정된 파라미터
beta_prime_opt, p_opt = params
d = 0.05     # 고정된 값
delta = 0.1  # 고정된 값
c = 23       # 고정된 값

# lambda 계산
lambda_ = T0 * d

# beta 계산
beta_opt = beta_prime_opt / lambda_

print("Estimated parameters:")
print(f"beta': {beta_prime_opt}")
print(f"beta: {beta_opt}")
print(f"p: {p_opt}")
print(f"d: {d}")
print(f"delta: {delta}")
print(f"c: {c}")

# 시각화
t = np.linspace(0, 12, 200)
solution = odeint(hiv_model, y0, t, args=(d, beta_prime_opt, delta, p_opt, c))
x = solution[:, 0]
z = solution[:, 2]

plt.figure(figsize=(10, 6))

# 건강한 T 세포 (x) 시각화
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Simulated Healthy T cells (x)')
plt.scatter(time_data, T_data, color='red', label='Observed Healthy T cells (x)', zorder=5)
plt.xlabel('Time (weeks)')
plt.ylabel('Concentration (log scale)')
plt.title('Healthy T cells over time')
plt.yscale('log')  # y축을 로그 스케일로 설정
plt.ylim(1, T_data.max() * 1.1)  # y축의 범위 설정
plt.legend()
plt.grid()

# 바이러스 (z) 시각화
plt.subplot(2, 1, 2)
plt.plot(t, z, label='Simulated Virus (z)')
plt.scatter(time_data, V_data, color='red', label='Observed Virus (z)', zorder=5)
plt.xlabel('Time (weeks)')
plt.ylabel('Concentration (log scale)')
plt.title('Virus concentration over time')
plt.yscale('log')  # y축을 로그 스케일로 설정
plt.ylim(1, V_data.max() * 1.1)  # y축의 범위 설정
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
