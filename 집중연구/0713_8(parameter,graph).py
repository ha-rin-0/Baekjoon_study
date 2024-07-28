import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 주어진 데이터
T_data = np.array([1567, 609, 399, 175, 536, 469, 448, 656, 336])
V_data = np.array([1.00E+00, 7.57E+06, 8.96E+06, 5.89E+05, 8.32E+05, 2.75E+05, 9.84E+04, 2.25E+05, 3.14E+04])
time_data = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12])  # 주차 데이터

# 시간 배열 생성 (200등분으로 나누기)
t = np.linspace(time_data.min(), time_data.max(), 200)

# 초기 조건 설정
T0 = T_data[0]  # 초기 타겟 세포 수
I0 = 0          # 초기 감염 세포 수
V0 = V_data[0]  # 초기 바이러스 입자 수
y0 = [T0, I0, V0]

# 모델 정의
def removal_model(y, t, beta, p):
    T, I, V = y
    lambda_ = T0 * 0.05  # 예시로 설정한 lambda_
    d = 0.05             # 예시로 설정한 d
    delta = 0.1          # 예시로 설정한 delta
    c = 23               # 예시로 설정한 c
    dT_dt = lambda_ - d*T - beta*T*V
    dI_dt = beta*T*V - delta*I
    dV_dt = p*I - c*V-beta*T*V
    return [dT_dt, dI_dt, dV_dt]

# 전체 모델 함수 정의
def full_model(t, beta, p):
    sol = odeint(removal_model, y0, t, args=(beta, p))
    return sol[:, 0], sol[:, 2]  # T값과 V값 반환

# 결합 데이터
def combined_model(t, beta, p):
    T_model, V_model = full_model(t, beta, p)
    return np.concatenate((T_model, V_model))

# 결합된 데이터 생성
combined_data = np.concatenate((T_data, V_data))

# 초기 추정값 설정
beta_guess = 1e-7  # beta 초기 추정값 설정
p_guess = 100      # p 초기 추정값 설정
p0 = [beta_guess, p_guess]

# curve_fit을 사용하여 파라미터 추정
params, params_covariance = curve_fit(combined_model, time_data, combined_data, p0)

# 추정된 파라미터 출력
estimated_beta, estimated_p = params
print("Estimated parameters:")
print(f"beta: {estimated_beta}")
print(f"p: {estimated_p}")

# 시각화
T_fit, V_fit = full_model(t, estimated_beta, estimated_p)

plt.figure(figsize=(10, 6))
plt.scatter(time_data, T_data, label='T Data', color='blue')
plt.plot(t, T_fit, label='T Fitted model', color='blue', linestyle='--')
plt.scatter(time_data, V_data, label='V Data', color='red')
plt.plot(t, V_fit, label='V Fitted model', color='red', linestyle='--')
plt.xlabel('Time (weeks)')
plt.ylabel('Count(log scale)')
plt.yscale('log')  # y축을 로그 스케일로 설정
plt.legend()
plt.show()