import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 미분 방정식 시스템 정의
def removal_model(t, y, lambda_, d, beta, delta, p, c):
    T, I, V = y
    dT_dt = lambda_ - d*T - beta*T*V
    dI_dt = beta*T*V - delta*I
    dV_dt = p*I - c*V - beta*T*V
    return [dT_dt, dI_dt, dV_dt]

# 초기 조건
T0 = 1e6  # 초기 타겟 세포 수
I0 = 0    # 초기 감염 세포 수
V0 = 10   # 초기 바이러스 입자 수
y0 = [T0, I0, V0]

# 매개변수
lambda_ = 1e4  # 새로운 타겟 세포 생성 속도
d = 0.01       # 타겟 세포의 자연 사망 속도
beta = 2e-7    # 타겟 세포의 감염 속도
delta = 0.7    # 감염 세포의 사망 속도
p = 100        # 바이러스 입자 생성 속도
c = 3          # 자유 바이러스 입자의 제거 속도

# 시뮬레이션 시간 범위
t_span = (0, 20)  # 20일
t_eval = np.linspace(*t_span, 400)  # 부드러운 곡선을 위한 400개의 시간 점

# 미분 방정식 풀이
sol = solve_ivp(removal_model, t_span, y0, args=(lambda_, d, beta, delta, p, c), t_eval=t_eval)

# 결과 추출
T, I, V = sol.y

plt.rcParams['font.family'] = 'Malgun Gothic'

# 결과 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(sol.t, T, label='타겟 세포 (T)')
plt.plot(sol.t, I, label='감염 세포 (I)')
plt.plot(sol.t, V, label='바이러스 입자 (V)')
plt.xlabel('시간 (일)')
plt.ylabel('인구')
plt.title('바이러스 감염 동력학')
plt.legend()
plt.grid()
plt.show()