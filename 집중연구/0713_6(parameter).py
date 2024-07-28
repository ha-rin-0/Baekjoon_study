import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 주어진 데이터
T_data = np.array([1567, 609, 399, 175, 536, 469, 448, 656, 336])
V_data = np.array([1.00E+00, 7.57E+06, 8.96E+06, 5.89E+05, 8.32E+05, 2.75E+05, 9.84E+04, 2.25E+05, 3.14E+04])

# 초기 조건 설정
T0 = T_data[0]  # 초기 타겟 세포 수
I0 = 0          # 초기 감염 세포 수
V0 = V_data[0]  # 초기 바이러스 입자 수
y0 = [T0, I0, V0]

# 모델 정의
def removal_model(y, t, beta, p):
    T, I, V = y
    lambda_ = T0 * 0.05  # 예시로 설정한 lambda_
    d = 0.05           # 예시로 설정한 d
    delta = 0.1        # 예시로 설정한 delta
    c = 23             # 예시로 설정한 c
    dT_dt = lambda_ - d*T - beta*T*V
    dI_dt = beta*T*V - delta*I
    dV_dt = p*I - c*V - beta*T*V
    return [dT_dt, dI_dt, dV_dt]

# V_model 함수 정의
def V_model(t, beta, p):
    sol = odeint(removal_model, y0, t, args=(beta, p))
    return sol[:, 2]  # V값만 반환

# 초기 추정값 설정
beta_guess = 1e-7  # beta 초기 추정값 설정
p_guess = 200     # p 초기 추정값 설정
p0 = [beta_guess, p_guess]

# 시간 배열 생성
t = np.linspace(0, len(T_data)-1, len(T_data))

# curve_fit을 사용하여 파라미터 추정
params, params_covariance = curve_fit(V_model, t, V_data, p0)

# 추정된 파라미터 출력
estimated_beta, estimated_p = params
print("Estimated parameters:")
print(f"beta: {estimated_beta}")
print(f"p: {estimated_p}")

# 시각화
plt.figure()
plt.scatter(t, V_data, label='Data')
plt.plot(t, V_model(t, estimated_beta, estimated_p), label='Fitted model', color='red')
plt.xlabel('Time')
plt.ylabel('Virus particles (log scale)')
plt.yscale('log')  # y축을 로그 스케일로 설정
plt.legend()
plt.show()
