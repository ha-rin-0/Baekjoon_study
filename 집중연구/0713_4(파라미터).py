import numpy as np
from scipy.optimize import curve_fit

# 주어진 데이터
T_data = np.array([1567, 609, 399, 175, 536, 469, 448, 656, 336])
V_data = np.array([1.00E+00, 7.57E+06, 8.96E+06, 5.89E+05, 8.32E+05, 2.75E+05, 9.84E+04, 2.25E+05, 3.14E+04])

# 초기 조건 설정
T0 = T_data[0]  # 초기 타겟 세포 수
I0 = 0          # 초기 감염 세포 수
V0 = V_data[0]  # 초기 바이러스 입자 수
y0 = [T0, I0, V0]

# 모델 정의
def removal_model(t, beta, p):
    T, I, V = y0  # y0는 초기 조건으로 위에서 설정한 값을 사용합니다.
    lambda_ = T0 * 0.05  # 예시로 설정한 lambda_
    d = 0.05           # 예시로 설정한 d
    delta = 0.1        # 예시로 설정한 delta
    c = 23             # 예시로 설정한 c
    dT_dt = lambda_ - d*T - beta*T*V
    dI_dt = beta*T*V - delta*I
    dV_dt = p*I - c*V - beta*T*V
    return [dT_dt, dI_dt, dV_dt]

# 초기 추정값 설정
beta_guess = 0.1  # 예시로 초기 추정값 설정
p_guess = 1.0     # 예시로 초기 추정값 설정

# curve_fit을 사용하여 beta와 p 추정
params, params_covariance = curve_fit(removal_model, T_data, V_data, p0=[beta_guess, p_guess])
print("Estimated beta:", params[0])
print("Estimated p:", params[1])