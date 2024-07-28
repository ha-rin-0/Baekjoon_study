import numpy as np
T_data=np.array([1567, 609, 399, 175, 536, 469, 448, 656, 336])
V_data=np.array([1.00E+00, 7.57E+06, 8.96E+06, 5.89E+05, 8.32E+05, 2.75E+05, 9.84E+04, 2.25E+05, 3.14E+04])
def removal_model(t, y, lambda_, d, beta, delta, p, c):
    T, I, V = y
    dT_dt = lambda_ - d*T - beta*T*V
    dI_dt = beta*T*V - delta*I
    dV_dt = p*I - c*V - beta*T*V
    return [dT_dt, dI_dt, dV_dt]

# 초기 조건
T0 = T_data(0)  # 초기 타겟 세포 수
I0 = 0    # 초기 감염 세포 수
V0 = 10   # 초기 바이러스 입자 수
y0 = [T0, I0, V0]

# 매개변수
d = 0.05       # 타겟 세포의 자연 사망 속도
lambda_ = T0 * d  # 새로운 타겟 세포 생성 속도
delta = 0.1    # 감염 세포의 사망 속도
c = 23         # 자유 바이러스 입자의 제거 속도

from scipy.optimize import curve_fit
params, params_covariance = curve_fit(removal_model, T_data, V_data)
