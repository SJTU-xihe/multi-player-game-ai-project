"""
搜索算法AI测试文件
测试不同搜索算法在各种游戏中的表现
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.ai_bots.search_ai import SearchAI
from agents.ai_bots.random_bot import RandomBot
from games.snake.snake_env import SnakeEnv
from games.sokoban.sokoban_env import SokobanEnv
from games.gomoku.gomoku_env import GomokuEnv


def test_search_ai_snake():
    """测试搜索AI在贪吃蛇游戏中的表现"""
    print("=== 测试搜索AI - 贪吃蛇游戏 ===")
    
    # 测试不同搜索算法
    algorithms = ["bfs", "dfs", "astar"]
    
    for algorithm in algorithms:
        print(f"\n测试 {algorithm.upper()} 算法:")
        
        # 创建环境和智能体
        env = SnakeEnv(board_size=10)
        search_ai = SearchAI(name=f"SearchAI-{algorithm.upper()}", 
                           player_id=1, 
                           search_algorithm=algorithm,
                           max_depth=15)
        random_ai = RandomBot(name="RandomBot", player_id=2)
        
        # 运行几局游戏
        wins = 0
        total_games = 5
        
        for game_idx in range(total_games):
            env.reset()
            game_over = False
            steps = 0
            max_steps = 200
            
            while not game_over and steps < max_steps:
                # 玩家1 (SearchAI) 行动
                obs = env._get_observation()
                action1 = search_ai.get_action(obs, env)
                
                if action1 is not None:
                    _, _, game_over, _, info = env.step(action1)
                    if game_over:
                        winner = env.get_winner()
                        if winner == 1:
                            wins += 1
                        break
                
                # 玩家2 (RandomBot) 行动
                if not game_over:
                    obs = env._get_observation()
                    action2 = random_ai.get_action(obs, env)
                    if action2 is not None:
                        _, _, game_over, _, info = env.step(action2)
                        if game_over:
                            winner = env.get_winner()
                            if winner == 1:
                                wins += 1
                            break
                
                steps += 1
            
            if steps >= max_steps:
                print(f"  游戏 {game_idx + 1}: 达到最大步数")
            else:
                winner = env.get_winner()
                if winner == 1:
                    wins += 1
                    print(f"  游戏 {game_idx + 1}: SearchAI 获胜")
                elif winner == 2:
                    print(f"  游戏 {game_idx + 1}: RandomBot 获胜")
                else:
                    print(f"  游戏 {game_idx + 1}: 平局")
        
        win_rate = wins / total_games * 100
        print(f"  {algorithm.upper()} 胜率: {win_rate:.1f}% ({wins}/{total_games})")
        
        # 显示AI信息
        ai_info = search_ai.get_info()
        print(f"  AI信息: {ai_info['description']}")


def test_search_ai_sokoban():
    """测试搜索AI在推箱子游戏中的表现"""
    print("\n=== 测试搜索AI - 推箱子游戏 ===")
    
    try:
        # 创建环境和智能体
        env = SokobanEnv(level_id=1)
        search_ai = SearchAI(name="SearchAI-Sokoban", 
                           player_id=1, 
                           search_algorithm="bfs",
                           max_depth=20)
        
        # 运行单人游戏测试
        env.reset()
        obs = env._get_observation()
        
        print(f"初始状态观察形状: {obs['board'].shape if isinstance(obs, dict) and 'board' in obs else 'N/A'}")
        
        # 测试几个动作
        for step in range(10):
            action = search_ai.get_action(obs, env)
            print(f"步骤 {step + 1}: AI选择动作 {action}")
            
            if action is not None:
                obs, reward, done, _, info = env.step(action)
                if done:
                    print(f"游戏结束，奖励: {reward}")
                    break
            else:
                print("AI无法选择有效动作")
                break
        
        # 显示AI信息
        ai_info = search_ai.get_info()
        print(f"AI信息: {ai_info['description']}")
        
    except Exception as e:
        print(f"推箱子测试出现错误: {e}")


def test_search_ai_gomoku():
    """测试搜索AI在五子棋游戏中的表现"""
    print("\n=== 测试搜索AI - 五子棋游戏 ===")
    
    try:
        # 创建环境和智能体
        env = GomokuEnv(board_size=9)
        search_ai = SearchAI(name="SearchAI-Gomoku", 
                           player_id=1, 
                           search_algorithm="bfs",
                           max_depth=5)
        random_ai = RandomBot(name="RandomBot", player_id=2)
        
        # 运行一局游戏
        env.reset()
        game_over = False
        step_count = 0
        max_steps = 40
        
        print("开始五子棋对局...")
        
        while not game_over and step_count < max_steps:
            # 玩家1 (SearchAI) 行动
            obs = env._get_observation()
            action1 = search_ai.get_action(obs, env)
            
            if action1 is not None and action1 in env.get_valid_actions():
                _, _, game_over, _, info = env.step(action1)
                step_count += 1
                print(f"步骤 {step_count}: SearchAI 选择位置 {action1}")
                
                if game_over:
                    winner = env.get_winner()
                    print(f"游戏结束! 获胜者: {'SearchAI' if winner == 1 else 'RandomBot' if winner == 2 else '平局'}")
                    break
            
            # 玩家2 (RandomBot) 行动
            if not game_over:
                obs = env._get_observation()
                action2 = random_ai.get_action(obs, env)
                if action2 is not None and action2 in env.get_valid_actions():
                    _, _, game_over, _, info = env.step(action2)
                    step_count += 1
                    print(f"步骤 {step_count}: RandomBot 选择位置 {action2}")
                    
                    if game_over:
                        winner = env.get_winner()
                        print(f"游戏结束! 获胜者: {'SearchAI' if winner == 1 else 'RandomBot' if winner == 2 else '平局'}")
                        break
        
        if step_count >= max_steps:
            print("达到最大步数，游戏结束")
        
        # 显示AI信息
        ai_info = search_ai.get_info()
        print(f"AI信息: {ai_info['description']}")
        
    except Exception as e:
        print(f"五子棋测试出现错误: {e}")


def test_search_algorithms_comparison():
    """比较不同搜索算法的性能"""
    print("\n=== 搜索算法性能比较 ===")
    
    algorithms = ["bfs", "dfs", "astar"]
    results = {}
    
    for algorithm in algorithms:
        print(f"\n测试 {algorithm.upper()} 算法性能...")
        
        # 创建AI实例
        search_ai = SearchAI(name=f"SearchAI-{algorithm.upper()}", 
                           player_id=1, 
                           search_algorithm=algorithm,
                           max_depth=10)
        
        # 创建简单测试环境
        env = SnakeEnv(board_size=8)
        env.reset()
        
        # 测试决策时间
        import time
        
        decision_times = []
        for _ in range(5):
            start_time = time.time()
            obs = env._get_observation()
            action = search_ai.get_action(obs, env)
            end_time = time.time()
            
            decision_time = (end_time - start_time) * 1000  # 转换为毫秒
            decision_times.append(decision_time)
        
        avg_time = sum(decision_times) / len(decision_times)
        results[algorithm] = {
            'avg_time_ms': avg_time,
            'min_time_ms': min(decision_times),
            'max_time_ms': max(decision_times)
        }
        
        print(f"  平均决策时间: {avg_time:.2f}ms")
        print(f"  最短决策时间: {min(decision_times):.2f}ms")
        print(f"  最长决策时间: {max(decision_times):.2f}ms")
    
    # 总结比较结果
    print(f"\n=== 算法性能总结 ===")
    for algorithm, metrics in results.items():
        print(f"{algorithm.upper()}: 平均 {metrics['avg_time_ms']:.2f}ms")


def main():
    """主测试函数"""
    print("搜索算法AI测试开始...")
    
    # 测试各种游戏
    test_search_ai_snake()
    test_search_ai_sokoban()
    test_search_ai_gomoku()
    
    # 性能比较
    test_search_algorithms_comparison()
    
    print("\n搜索算法AI测试完成!")


if __name__ == "__main__":
    main()
