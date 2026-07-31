import os
import numpy as np
import pandas as pd
from vmdpy import VMD
import time
from joblib import Parallel, delayed
import multiprocessing
from dotenv import load_dotenv

load_dotenv()

class VMD_Decompose:
    def __init__(self, data, alpha=2000, tau=0, K=int(os.getenv("VmdK", 4)), DC=0, init=1, tol=1e-7, n_jobs=int(os.getenv("n_jobs", -1))):
        self.data = data
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol
        self.n_jobs=n_jobs
        
    def vmd(self):
        """
        对多维风速数据进行VMD分解
        输入: data shape (T, H, W) - T个时间步，HxW的空间网格，1个通道（可省略）
        输出: imfs shape (T, K, H, W) - K个分解模态
        """
        # 获取数据维度
        T, height, width = self.data.shape
        
        # 初始化输出数组
        imfs_all = np.zeros((T, self.K, height, width))
        imfs_hat_all = np.zeros((T, self.K, height, width))
        omega_all = np.zeros((self.K, height, width))
        
        # 对每个空间位置进行VMD分解
        for i in range(height):
            print(f"正在处理第{i}行")
            for j in range(width):
                # 提取该位置的时间序列
                time_series = self.data[:, i, j]  # shape: (T,)
                
                # 对该时间序列进行VMD分解
                imfs, imfs_hat, omega = VMD(time_series, 
                                          alpha=self.alpha, 
                                          tau=self.tau, 
                                          K=self.K, 
                                          DC=self.DC, 
                                          init=self.init, 
                                          tol=self.tol)
                
                # VMD返回的imfs形状是(K, T)，需要转换为(T, K)
                imfs = imfs.T  # 转置为(T, K)
                # imfs_hat = imfs_hat.T  # 转置为(T, K)
                
                # omega的形状是(T, K)，我们取最后一个时间步的值作为中心频率
                omega_final = omega[-1, :]  # 取最后一行，形状为(K,)
                
                # 将结果存储到对应的位置
                imfs_all[:, :, i, j] = imfs  # shape: (T, K)
                imfs_hat_all[:, :, i, j] = imfs_hat  # shape: (T, K)
                omega_all[:, i, j] = omega_final  # shape: (K,)
        
        return imfs_all, imfs_hat_all, omega_all
    
    def vmd_parallel(self, verbose=True):
        """
        并行版本的VMD分解，使用多进程加速
        输入: data shape (T, H, W)
        输出: imfs shape (T, K, H, W)
        """
        
        n_jobs = self.n_jobs
        
        # 获取数据维度
        T, height, width = self.data.shape
        total_positions = height * width
        
        if verbose:
            print(f"开始并行VMD分解...")
            print(f"数据形状: {self.data.shape}")
            print(f"分解模态数: {self.K}")
            print(f"总位置数: {total_positions}")
            print(f"并行进程数: {n_jobs if n_jobs != -1 else multiprocessing.cpu_count()}")
            start_time = time.time()
        
        # 定义单个位置的VMD处理函数
        def process_position(i, j):
            time_series = self.data[:, i, j]
            imfs, imfs_hat, omega = VMD(time_series, 
                                      alpha=self.alpha, 
                                      tau=self.tau, 
                                      K=self.K, 
                                      DC=self.DC, 
                                      init=self.init, 
                                      tol=self.tol)
            
            # VMD返回的imfs形状是(K, T)，需要转换为(T, K)
            imfs = imfs.T  # 转置为(T, K)
            # imfs_hat = imfs_hat.T  # 转置为(T, K)
            
            # omega的形状是(T, K)，我们取最后一个时间步的值作为中心频率
            omega_final = omega[-1, :]  # 取最后一行，形状为(K,)
            
            return i, j, imfs, imfs_hat, omega_final
        
        # 并行处理所有位置
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(process_position)(i, j) 
            for i in range(height) 
            for j in range(width)
        )
        
        # 初始化输出数组
        imfs_all = np.zeros((T, self.K, height, width))
        imfs_hat_all = np.zeros((T, self.K, height, width))
        omega_all = np.zeros((self.K, height, width))
        
        # 整理结果
        for i, j, imfs, imfs_hat, omega in results:
            imfs_all[:, :, i, j] = imfs
            imfs_hat_all[:, :, i, j] = imfs_hat
            omega_all[:, i, j] = omega
        
        if verbose:
            end_time = time.time()
            print(f"VMD分解完成，耗时: {end_time - start_time:.2f}秒")
            print(f"输出形状: imfs={imfs_all.shape}, omega={omega_all.shape}")
        
        return imfs_all, imfs_hat_all, omega_all
    
    def compare_performance(self, test_size=1000):
        """
        比较串行和并行处理的性能
        """
        import multiprocessing
        
        print("性能测试开始...")
        print(f"测试数据大小: {test_size}个时间步")
        
        # 创建测试数据，自动适应空间维度
        _, height, width = self.data.shape
        test_data = np.random.rand(test_size, height, width)
        vmd_test = VMD_Decompose(test_data, K=4)
        
        # 测试串行处理
        print("\n1. 测试串行处理...")
        start_time = time.time()
        try:
            imfs_serial, _, _ = vmd_test.vmd()
            serial_time = time.time() - start_time
            print(f"串行处理完成，耗时: {serial_time:.2f}秒")
        except Exception as e:
            print(f"串行处理失败: {e}")
            serial_time = float('inf')
        
        # 测试并行处理
        print("\n2. 测试并行处理...")
        start_time = time.time()
        try:
            imfs_parallel, _, _ = vmd_test.vmd_parallel(verbose=False)
            parallel_time = time.time() - start_time
            print(f"并行处理完成，耗时: {parallel_time:.2f}秒")
        except Exception as e:
            print(f"并行处理失败: {e}")
            parallel_time = float('inf')
        
        # 比较结果
        if serial_time != float('inf') and parallel_time != float('inf'):
            speedup = serial_time / parallel_time
            print(f"\n性能比较:")
            print(f"串行时间: {serial_time:.2f}秒")
            print(f"并行时间: {parallel_time:.2f}秒")
            print(f"加速比: {speedup:.2f}x")
            print(f"CPU核心数: {multiprocessing.cpu_count()}")
            
            # 验证结果一致性
            if np.allclose(imfs_serial, imfs_parallel, rtol=1e-10):
                print("✓ 串行和并行结果一致")
            else:
                print("✗ 串行和并行结果不一致")
        
        return serial_time, parallel_time
    
    def get_optimal_n_jobs(self):
        """
        获取最优的并行进程数
        """
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        
        # 根据数据大小和CPU核心数推荐最优进程数
        T, height, width = self.data.shape
        total_positions = height * width
        
        if total_positions <= cpu_count:
            # 位置数少于CPU核心数，使用位置数作为进程数
            optimal_jobs = total_positions
        else:
            # 位置数多于CPU核心数，使用CPU核心数
            optimal_jobs = cpu_count
        
        print(f"推荐并行进程数: {optimal_jobs}")
        print(f"CPU核心数: {cpu_count}")
        print(f"总位置数: {total_positions}")
        
        return optimal_jobs
