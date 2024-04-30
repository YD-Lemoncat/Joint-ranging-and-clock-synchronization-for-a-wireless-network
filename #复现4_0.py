#构造T矩阵
#复现3.6
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定义参数
N = 4  # 节点数
K_values = list(range(5, 21))  # 两节点间通信次数的范围
M = N * (N - 1) // 2  # 唯一的节点间距离数
sigma = 0.1  # 噪声方差
c = 3e8  # 电磁波波速，假设为 3e8 m/s

# 初始化结果列表
mse_omega_hat = []  # 时钟漂移 MSE
mse_phi_hat = []  # 时钟偏移 MSE
mse_d_hat = []  # 节点间距离 MSE

# 重复 Monte Carlo 实验
for K in tqdm(K_values):  # 使用 tqdm 进行循环迭代
    # 初始化存储每次 Monte Carlo 实验结果的列表
    omega_hat_list = []
    phi_hat_list = []
    d_hat_list = []
    
    # Monte Carlo 运行次数
    num_mc_runs = 10
    
    for _ in range(num_mc_runs):
        # 初始化模拟数据
        clock_skew_range = np.random.uniform(0.998, 1.002, N)#生成节点的时钟漂移序列
        clock_offset_range = np.random.uniform(-1, 1, N)#生成节点的时钟偏移序列
        clock_skew = np.array(clock_skew_range).reshape(1, -1)#将时钟漂移序列转换为行向量（二维行向量1*N）
        clock_offset = np.array(clock_offset_range).reshape(1, -1)#将时钟偏移序列转换为行向量（二维行向量1*N）
        T = np.zeros((2*K*M, N))  # 初始化时间标记 T 矩阵
        E1 = np.zeros((2*K*M, N))  # 初始化 E1 矩阵为全零
        E2 = np.zeros((2*K*M, M))  # 初始化 E2 矩阵为全零
        tau = np.zeros((M, 1))  #初始化 tau 矩阵为全零

       # 设置噪声q
        q = np.random.normal(0, sigma, (2*K*M, 1))  #设置q序列
        q = np.array(q).reshape(-1, 1)  #将q序列变为列向量（2KM*1）

        # 设置alpha和beta
        one_N = np.ones((N, 1))  # 生成全 1 向量
        alpha = one_N / clock_skew.T
        beta = -clock_offset.T / clock_skew.T
        alpha[0] = 1  # 第一个节点的时钟漂移设置为 1
        beta[0] = 0 # 第一个节点的时钟偏移设置为 0

        # 构造 T 矩阵
        def construct_T(N, K):
            #生成线性分布的时间标记
            time_markers = np.linspace(1, 100, 2 * K)
            #填充矩阵
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    if i != j:
                        #计算行索引
                        row_index = ((max(i, j) - 1) * (max(i, j) - 2) // 2 + (min(i, j) - 1)) * 2 * K
                        #根据发送和接收时间的关系填充矩阵
                        for k in range(2 * K):
                            T[row_index + k, i - 1] = time_markers[k] if i < j else -time_markers[k]
                            T[row_index + k, j - 1] = -time_markers[k] if i < j else time_markers[k]

            return T
        T = construct_T(N, K)

        # 构造 E1 矩阵
        def construct_E1(N, K):
            # 填充E1矩阵
            for i in range(1, N):
                for j in range(2*K):
                    E1[(i-1)*2*K + j][0] = 1
                    E1[(i-1)*2*K + j][i] = -1
            return E1

        E1 = construct_E1(N, K)
        
        # 构造矩阵 E2            
        #1、构造矩阵 e(2K*1)
        e = np.ones((2*K, 1))
        e[0] = -1  # 第一个元素为 -1
        #2、构造单位矩阵 I_M(M*M)
        I_M = np.eye(M)
        #3、-I_M叉e乘得到E2
        E2 = -np.kron(I_M, e)

        # 设置 tau 矩阵
        E2_pinv = np.linalg.pinv(E2)
        tau = np.dot(E2_pinv, q - np.dot(T, alpha) - np.dot(E1, beta))

        # 构造 A_bar 矩阵        
        #1、构造 E1_bar 矩阵
        #初始化 E1_bar 矩阵为全零，大小为 (2*K*M, N-1)
        E1_bar = np.zeros((2*K*M, N-1))
        # 从第二列开始，将 E1 的除了第一列之外的所有列复制到 E1_bar 中
        for i in range(1, N):
            E1_bar[:, i-1] = E1[:, i]

        #2、构造 T_bar 矩阵
        T_bar = np.zeros((2*K*M, N-1))
        for i in range(1, N):
            T_bar[:, i-1] = T[:, i]

        #3、合成 A_bar 矩阵
        A_bar = np.hstack((T_bar, E1_bar, E2))  # 构造A_bar 矩阵
        t1 = T[:, 0].reshape(-1, 1)  # 取第一列作为 t1

        # 构造 theta 矩阵
        alpha_bar = alpha[1:]
        beta_bar = beta[1:]
        theta = np.vstack((alpha_bar, beta_bar, tau))
        
        # 计算 GLS 解
        theta_hat = np.linalg.pinv(A_bar.T @ A_bar) @ A_bar.T @ t1
        
        # 提取时钟偏移、时钟漂移和节点间距离
        alpha_hat = theta_hat[:N-1]
        beta_hat = theta_hat[N-1:2*(N-1)]
        tau_hat = theta_hat[2*(N-1):]
        
        # 转换为时钟漂移、时钟偏移和距离
        omega_hat = 1 / alpha_hat
        phi_hat = -beta_hat / alpha_hat
        d_hat = c * tau_hat  # 假设 c 是电磁波的速度
        
        # 存储每次实验的结果
        omega_hat_list.append(omega_hat)
        phi_hat_list.append(phi_hat)
        d_hat_list.append(d_hat)
    
    # 计算每个 K 值下的平均结果
    avg_omega = np.mean(omega_hat_list, axis=0)
    avg_phi = np.mean(phi_hat_list, axis=0)
    avg_d = np.mean(d_hat_list, axis=0)
    
    # 计算 MSE
    true_clock_skew = clock_skew  # 真实的时钟漂移
    true_clock_offset = clock_offset  # 真实的时钟偏移
    true_propagation_delay = tau  # 真实的传播延迟
    mse_omega = np.mean((avg_omega - true_clock_skew)**2)  # 时钟漂移的 MSE
    mse_phi = np.mean((avg_phi - true_clock_offset)**2)  # 时钟偏移的 MSE
    mse_d = np.mean((avg_d - c*true_propagation_delay)**2)  # 传播延迟的 MSE
    
    # 存储 MSE 结果
    mse_omega_hat.append(mse_omega)
    mse_phi_hat.append(mse_phi)
    mse_d_hat.append(mse_d)

# 输出 MSE 结果
for idx, K in enumerate(K_values):
    print(f"For K = {K}:")
    print("  MSE of Clock Skews (omega_hat):", mse_omega_hat[idx])
    print("  MSE of Clock Offsets (phi_hat):", mse_phi_hat[idx])
    print("  MSE of Propagation Delays (d_hat):", mse_d_hat[idx])
    print()

# 可视化 MSE 结果
plt.figure(figsize=(15, 5))

# 时钟漂移 MSE 可视化
plt.subplot(1, 3, 1)
plt.plot(K_values, mse_omega_hat, marker='o')
plt.xlabel('Number of Two Way Communications (K)')
plt.ylabel('MSE of Clock Skews (omega)')
plt.title('MSE of Clock Skews')
plt.grid(True)

# 传播延迟 MSE 可视化
plt.subplot(1, 3, 2)
plt.plot(K_values, mse_d_hat, marker='o')
plt.xlabel('Number of Two Way Communications (K)')
plt.ylabel('MSE of Propagation Delays (tau)')
plt.title('MSE of Propagation Delays')
plt.grid(True)

# 时钟偏移 MSE 可视化
plt.subplot(1, 3, 3)
plt.plot(K_values, mse_phi_hat, marker='o')
plt.xlabel('Number of Two Way Communications (K)')
plt.ylabel('MSE of Clock Offsets (phi)')
plt.title('MSE of Clock Offsets')
plt.grid(True)

plt.tight_layout()
plt.show()