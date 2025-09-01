import numpy as np
import time
import statistics
# export PYTHONPATH="$PWD"
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from envs.overcook_gym_env import OvercookPygameEnv

def test_overcooked2_env():
    print("\n| test_overcooked2_env()")
    # env = OvercookPygameEnv(map_name='4playersplit', ifrender=True, debug=True)
    env = OvercookPygameEnv(map_name='supereasy', ifrender=False, debug=False, lossless_obs=False, fps=15)
    # env = OvercookPygameEnv(map_name='supereasy', ifrender=True, debug=True)
    for _ in range(2):
        nobs, _, available_actions = env.reset()
        print("observation shape:", nobs[0].shape)
        done = False
        while not done:
            random_action = np.random.randint(0, 6, size=env.n_agents)
            nobs, _, rewards, dones, infos, available_actions = env.step(random_action)
            done = dones[0]
            print(nobs)
def test_function_performance():
    """
    测试 OvercookPygameEnv 中关键函数的性能
    """
    print("\n=== OvercookPygameEnv 性能测试 ===")
    
    # 创建环境实例
    env = OvercookPygameEnv(map_name='supereasy', ifrender=False, debug=False, lossless_obs=True, fps=30)
    
    # 初始化环境
    nobs, _, available_actions = env.reset()
    
    # 测试参数
    num_tests = 100
    test_actions = [np.random.randint(0, 6, size=env.n_agents) for _ in range(num_tests)]
    
    print(f"测试次数: {num_tests}")
    print(f"环境配置: map_name='supereasy', n_agents={env.n_agents}")
    print("-" * 60)
    
    # 1. 测试 step() 函数
    print("\n1. 测试 step() 函数性能...")
    step_times = []
    
    for i in range(num_tests):
        # 确保环境处于有效状态
        if i > 0:  # 第一次已经reset过了
            env.reset()
        
        action = test_actions[i]
        
        start_time = time.perf_counter()
        nobs, _, rewards, dones, infos, available_actions = env.step(action)
        end_time = time.perf_counter()
        
        step_times.append(end_time - start_time)
        
        # 如果回合结束，重置环境以保持一致性
        if dones[0]:
            env.reset()
    
    # 2. 测试 get_obs() 函数
    print("2. 测试 get_obs() 函数性能...")
    obs_times = []
    
    # 重置环境到一致状态
    env.reset()
    
    for i in range(num_tests):
        start_time = time.perf_counter()
        obs = env.get_obs()
        end_time = time.perf_counter()
        
        obs_times.append(end_time - start_time)
        
        # 偶尔执行一个动作以保持环境动态性
        if i % 10 == 0 and i > 0:
            action = np.random.randint(0, 6, size=env.n_agents)
            nobs, _, rewards, dones, infos, available_actions = env.step(action)
            if dones[0]:
                env.reset()
    
    # 3. 测试 calculate_reward() 函数
    print("3. 测试 calculate_reward() 函数性能...")
    reward_times = []
    
    # 重置环境到一致状态
    env.reset()
    
    for i in range(num_tests):
        start_time = time.perf_counter()
        sparse_reward, shaped_reward = env.calculate_reward()
        end_time = time.perf_counter()
        
        reward_times.append(end_time - start_time)
        
        # 偶尔执行一个动作以保持环境动态性
        if i % 10 == 0 and i > 0:
            action = np.random.randint(0, 6, size=env.n_agents)
            nobs, _, rewards, dones, infos, available_actions = env.step(action)
            if dones[0]:
                env.reset()
    
    # 计算统计信息
    def calculate_stats(times, func_name):
        times_ms = [t * 1000 for t in times]  # 转换为毫秒
        return {
            'function': func_name,
            'count': len(times),
            'mean_ms': statistics.mean(times_ms),
            'median_ms': statistics.median(times_ms),
            'min_ms': min(times_ms),
            'max_ms': max(times_ms),
            'std_ms': statistics.stdev(times_ms) if len(times_ms) > 1 else 0
        }
    
    # 收集所有统计信息
    results = [
        calculate_stats(step_times, 'step()'),
        calculate_stats(obs_times, 'get_obs()'),
        calculate_stats(reward_times, 'calculate_reward()')
    ]
    
    # 输出详细结果
    print("\n" + "=" * 80)
    print("                    性能测试结果汇总")
    print("=" * 80)
    
    header = f"{'函数名称':<20} {'测试次数':<10} {'平均耗时(ms)':<15} {'中位数(ms)':<15} {'最小值(ms)':<15} {'最大值(ms)':<15} {'标准差(ms)':<15}"
    print(header)
    print("-" * len(header))
    
    for result in results:
        row = f"{result['function']:<20} {result['count']:<10} {result['mean_ms']:<15.3f} {result['median_ms']:<15.3f} {result['min_ms']:<15.3f} {result['max_ms']:<15.3f} {result['std_ms']:<15.3f}"
        print(row)
    
    print("\n" + "=" * 80)
    
    # 输出简化的性能报告
    print("\n📊 性能分析摘要:")
    for result in results:
        print(f"  • {result['function']}: 平均 {result['mean_ms']:.3f} ms (测试 {result['count']} 次)")
    
    # 识别性能瓶颈
    slowest = max(results, key=lambda x: x['mean_ms'])
    fastest = min(results, key=lambda x: x['mean_ms'])
    
    print(f"\n⚡ 最快函数: {fastest['function']} ({fastest['mean_ms']:.3f} ms)")
    print(f"🐌 最慢函数: {slowest['function']} ({slowest['mean_ms']:.3f} ms)")
    print(f"📈 性能差异: {slowest['mean_ms'] / fastest['mean_ms']:.1f}x")
    
    # 清理资源
    env.close()
    
    return results

def test_performance_under_load():
    """
    在不同负载条件下测试性能
    """
    print("\n=== 负载测试 ===")
    
    test_configs = [
        {'map_name': 'supereasy', 'n_agents': 2, 'label': '简单地图-2智能体'},
        {'map_name': '2playerhard', 'n_agents': 2, 'label': '困难地图-2智能体'}
    ]
    
    for config in test_configs:
        print(f"\n测试配置: {config['label']}")
        print("-" * 40)
        
        try:
            env = OvercookPygameEnv(
                map_name=config['map_name'], 
                ifrender=False, 
                debug=False, 
                lossless_obs=True, 
                fps=30
            )
            
            env.reset()
            
            # 快速性能测试 (较少次数)
            num_tests = 20
            step_times = []
            
            for i in range(num_tests):
                if i > 0:
                    env.reset()
                
                action = np.random.randint(0, 6, size=env.n_agents)
                
                start_time = time.perf_counter()
                nobs, _, rewards, dones, infos, available_actions = env.step(action)
                end_time = time.perf_counter()
                
                step_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            avg_time = statistics.mean(step_times)
            print(f"  step() 平均耗时: {avg_time:.3f} ms")
            
            env.close()
            
        except Exception as e:
            print(f"  测试失败: {str(e)}")

if __name__ == '__main__':
    print('\n| test_env.py - 性能测试版本')
    
    # 运行原有测试
    test_overcooked2_env()
    
    # 运行性能测试
    # test_function_performance()
    
    # 运行负载测试
    # test_performance_under_load()
    