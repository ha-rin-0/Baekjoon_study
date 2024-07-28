import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# 주어진 데이터 (로그 스케일을 사용하도록 변환)
T_data = np.array([1567000, 609000, 399000, 175000, 536000, 469000, 448000, 656000, 336000])
V_data = np.array([200, 7.57E+06, 8.96E+06, 5.89E+05, 8.32E+05, 2.75E+05, 9.84E+04, 2.25E+05, 3.14E+04])
log_T_data = np.log10(T_data)
log_V_data = np.log10(V_data)
time_data = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12])  # 주차 데이터

# 초기 조건 설정
T0 = T_data[0]  # 초기 타겟 세포 수
I0 = 0          # 초기 감염 세포 수
V0 = 200  # 초기 바이러스 입자 수
y0 = [T0, I0, V0]

# 새로운 미분방정식 정의
def hiv_model(y, t, lambda_, d, beta, delta, p, c):
    T, I, V = y
    dT_dt = lambda_ - d * T - beta * T * V
    dI_dt = beta * T * V - delta * I
    dV_dt = p * I - c * V - beta * T * V
    return [dT_dt, dI_dt, dV_dt]

# 모델 함수 정의
def model_func(t, beta, p):
    d = 0.05     # 고정된 값
    delta = 0.1  # 고정된 값
    c = 23       # 고정된 값
    lambda_ = T0 * d  # 초기 타겟 세포 수와 d로 lambda_ 계산
    
    solution = odeint(hiv_model, y0, t, args=(lambda_, d, beta, delta, p, c))
    log_T = np.log10(solution[:, 0])
    log_V = np.log10(solution[:, 2])
    return np.concatenate((log_T, log_V))  # 로그 스케일로 반환

# 초기 추정값 설정
beta_guess = 1e-7  # 초기 추정값 설정
p_guess = 200      # 초기 추정값 설정
initial_guesses = [beta_guess, p_guess]

# curve_fit을 사용하여 파라미터 추정
params, _ = curve_fit(model_func, time_data, np.concatenate((log_T_data, log_V_data)), p0=initial_guesses)

# 추정된 파라미터
beta_opt, p_opt = params
d = 0.05     # 고정된 값
delta = 0.1  # 고정된 값
c = 23       # 고정된 값

# lambda 계산
lambda_ = T0 * d

print("Estimated parameters:")
print(f"beta: {beta_opt}")
print(f"p: {p_opt}")
print(f"d: {d}")
print(f"delta: {delta}")
print(f"c: {c}")
print(f"lambda: {lambda_}")

# 시각화
t = np.linspace(0, 12, 200)
solution = odeint(hiv_model, y0, t, args=(lambda_, d, beta_opt, delta, p_opt, c))
log_T = np.log10(solution[:, 0])
log_V = np.log10(solution[:, 2])

plt.figure(figsize=(10, 6))

# 건강한 T 세포 (T) 시각화
plt.subplot(2, 1, 1)
plt.plot(t, log_T, label='Simulated Healthy T cells (T) [log scale]')
plt.scatter(time_data, log_T_data, color='red', label='Observed Healthy T cells (T) [log scale]', zorder=5)
plt.xlabel('Time (weeks)')
plt.ylabel('Log10 Concentration')
plt.title('Healthy T cells over time [log scale]')
plt.ylim(0, np.log10(T_data.max() * 1.1))  # y축의 범위 설정
plt.legend()
plt.grid()

# 바이러스 농도 (V) 시각화
plt.subplot(2, 1, 2)
plt.plot(t, log_V, label='Simulated Virus (V) [log scale]', color='green')
plt.scatter(time_data, log_V_data, color='orange', label='Observed Virus (V) [log scale]', zorder=5)
plt.xlabel('Time (weeks)')
plt.ylabel('Log10 Concentration')
plt.title('Virus over time [log scale]')
plt.ylim(0, np.log10(V_data.max() * 1.1))  # y축의 범위 설정
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
