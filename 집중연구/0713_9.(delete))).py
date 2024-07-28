import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 주어진 데이터
T_data = np.array([1567, 609, 399, 175, 536, 469, 448, 656, 336])
V_data = np.array([1.00E+00, 7.57E+06, 8.96E+06, 5.89E+05, 8.32E+05, 2.75E+05, 9.84E+04, 2.25E+05, 3.14E+04])
time_data = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12])  # 주차 데이터

# 초기 조건 설정
T0 = T_data[0]  # 초기 타겟 세포 수
I0 = 0          # 초기 감염 세포 수
V0 = V_data[0]  # 초기 바이러스 입자 수
y0 = [T0, I0, V0]

# 모델 정의
def removal_model(y, t, beta, p, lambda_):
    T, I, V = y
    d = 0.05             # 예시로 설정한 d
    delta = 0.1          # 예시로 설정한 delta
    c = 23               # 예시로 설정한 c
    dT_dt = lambda_ - d*T - beta*T*V
    dI_dt = beta*T*V - delta*I
    dV_dt = p*I - c*V-beta*T*V
    return [dT_dt, dI_dt, dV_dt]

# 전체 모델 함수 정의
def full_model(t, beta, p, lambda_):
    sol = odeint(removal_model, y0, t, args=(beta, p, lambda_))
    return sol[:, 0], sol[:, 2]  # T값과 V값 반환

# 결합 데이터
def combined_model(params, t):
    beta, p, lambda_= params
    T_model, V_model = full_model(t, beta, p, lambda_)
    return np.concatenate((T_model, V_model))

# 결합된 데이터 생성
combined_data = np.concatenate((T_data, V_data))

# 초기 추정값 설정
beta_guess = 1e-7  # beta 초기 추정값 설정
p_guess = 200      # p 초기 추정값 설정
lambda_guess = 1567 * 0.05    # lambda 초기 추정값 설정
initial_guess = [beta_guess, p_guess, lambda_guess]

# 최적화 함수 정의
def objective_function(params):
    predicted_data = combined_model(params, time_data)
    residual = combined_data - predicted_data
    return np.sum(residual ** 2)

# 최적화 수행
result = minimize(objective_function, initial_guess, method='Nelder-Mead')

# 추정된 파라미터 출력
estimated_beta, estimated_p, estimated_lambda_ = result.x
print("Estimated parameters:")
print(f"beta: {estimated_beta}")
print(f"p: {estimated_p}")
print(f"lambda: {estimated_lambda_}")

# 시각화
t = np.linspace(time_data.min(), time_data.max(), 500)
T_fit, V_fit = full_model(t, estimated_beta, estimated_p, estimated_lambda_)

plt.figure(figsize=(10, 6))
plt.scatter(time_data, T_data, label='T Data', color='blue')
plt.plot(t, T_fit, label='T Fitted model', color='blue', linestyle='--')
plt.scatter(time_data, V_data, label='V Data', color='red')
plt.plot(t, V_fit, label='V Fitted model', color='red', linestyle='--')
plt.xlabel('Time (weeks)')
plt.ylabel('Count (log scale)')
plt.yscale('log')  # y축을 로그 스케일로 설정
plt.legend()
plt.show()
