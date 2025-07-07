"""
搜索算法AI使用示例
展示如何在不同游戏中使用各种搜索算法
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ai_bots.search_ai import SearchAI
from agents.ai_bots.random_bot import RandomBot
from games.snake.snake_env import SnakeEnv
from games.sokoban.sokoban_env import SokobanEnv


def example_bfs_snake():
    """示例：使用BFS算法的贪吃蛇AI"""
    print("=== BFS算法贪吃蛇AI示例 ===")
    
    # 创建使用BFS算法的AI
    bfs_ai = SearchAI(
        name="BFS-SnakeAI",
        player_id=1,
        search_algorithm="bfs",  # 广度优先搜索
        max_depth=15
    )
    
    # 创建对手（随机AI）
    random_opponent = RandomBot(name="RandomOpponent", player_id=2)
    
    # 创建游戏环境
    env = SnakeEnv(board_size=12)
    env.reset()
    
    print(f"AI信息: {bfs_ai.get_info()}")
    print("开始游戏...\n")
    
    game_over = False
    step = 0
    max_steps = 100
    
    while not game_over and step < max_steps:
        step += 1
        
        # BFS AI行动
        obs = env._get_observation()
        action1 = bfs_ai.get_action(obs, env)
        
        if action1 is not None:
            _, _, game_over, _, info = env.step(action1)
            print(f"步骤 {step}: BFS-AI 选择动作 {action1}")
            
            if game_over:
                winner = env.get_winner()
                print(f"游戏结束! 获胜者: {'BFS-AI' if winner == 1 else '随机AI' if winner == 2 else '平局'}")
                break
        
        # 随机AI行动
        if not game_over:
            obs = env._get_observation()
            action2 = random_opponent.get_action(obs, env)
            if action2 is not None:
                _, _, game_over, _, info = env.step(action2)
                
                if game_over:
                    winner = env.get_winner()
                    print(f"游戏结束! 获胜者: {'BFS-AI' if winner == 1 else '随机AI' if winner == 2 else '平局'}")
                    break
    
    if step >= max_steps:
        print("达到最大步数，游戏结束")


def example_astar_snake():
    """示例：使用A*算法的贪吃蛇AI"""
    print("\n=== A*算法贪吃蛇AI示例 ===")
    
    # 创建使用A*算法的AI
    astar_ai = SearchAI(
        name="AStar-SnakeAI",
        player_id=1,
        search_algorithm="astar",  # A*搜索
        max_depth=20
    )
    
    # 创建对手
    random_opponent = RandomBot(name="RandomOpponent", player_id=2)
    
    # 创建游戏环境
    env = SnakeEnv(board_size=10)
    env.reset()
    
    print(f"AI信息: {astar_ai.get_info()}")
    print("开始游戏...\n")
    
    # 运行游戏循环
    game_over = False
    step = 0
    max_steps = 100
    
    while not game_over and step < max_steps:
        step += 1
        
        # A* AI行动
        obs = env._get_observation()
        action1 = astar_ai.get_action(obs, env)
        
        if action1 is not None:
            _, _, game_over, _, info = env.step(action1)
            print(f"步骤 {step}: A*-AI 选择动作 {action1}")
            
            if game_over:
                winner = env.get_winner()
                print(f"游戏结束! 获胜者: {'A*-AI' if winner == 1 else '随机AI' if winner == 2 else '平局'}")
                break
        
        # 对手行动
        if not game_over:
            obs = env._get_observation()
            action2 = random_opponent.get_action(obs, env)
            if action2 is not None:
                _, _, game_over, _, info = env.step(action2)
                
                if game_over:
                    winner = env.get_winner()
                    print(f"游戏结束! 获胜者: {'A*-AI' if winner == 1 else '随机AI' if winner == 2 else '平局'}")
                    break


def example_dfs_sokoban():
    """示例：使用DFS算法的推箱子AI"""
    print("\n=== DFS算法推箱子AI示例 ===")
    
    try:
        # 创建使用DFS算法的AI
        dfs_ai = SearchAI(
            name="DFS-SokobanAI",
            player_id=1,
            search_algorithm="dfs",  # 深度优先搜索
            max_depth=25
        )
        
        # 创建推箱子环境
        env = SokobanEnv(level_id=1)
        env.reset()
        
        print(f"AI信息: {dfs_ai.get_info()}")
        print("开始推箱子游戏...\n")
        
        # 运行游戏
        done = False
        step = 0
        max_steps = 50
        
        while not done and step < max_steps:
            step += 1
            
            obs = env._get_observation()
            action = dfs_ai.get_action(obs, env)
            
            if action is not None:
                obs, reward, done, _, info = env.step(action)
                print(f"步骤 {step}: DFS-AI 选择动作 {action}, 奖励: {reward}")
                
                if done:
                    print(f"游戏结束! 最终奖励: {reward}")
                    if reward > 0:
                        print("恭喜！成功完成关卡！")
                    break
            else:
                print(f"步骤 {step}: AI无法选择有效动作")
                break
        
        if step >= max_steps:
            print("达到最大步数，游戏结束")
            
    except Exception as e:
        print(f"推箱子示例出现错误: {e}")


def example_algorithm_comparison():
    """示例：比较不同搜索算法"""
    print("\n=== 搜索算法比较示例 ===")
    
    algorithms = ["bfs", "dfs", "astar"]
    results = {}
    
    for algorithm in algorithms:
        print(f"\n测试 {algorithm.upper()} 算法:")
        
        # 创建AI
        ai = SearchAI(
            name=f"{algorithm.upper()}-AI",
            player_id=1,
            search_algorithm=algorithm,
            max_depth=15
        )
        
        # 创建环境
        env = SnakeEnv(board_size=8)
        
        # 运行多局游戏统计胜率
        wins = 0
        total_games = 3
        
        for game in range(total_games):
            env.reset()
            random_opponent = RandomBot(name="RandomBot", player_id=2)
            
            game_over = False
            steps = 0
            max_steps = 50
            
            while not game_over and steps < max_steps:
                # AI行动
                obs = env._get_observation()
                action1 = ai.get_action(obs, env)
                
                if action1 is not None:
                    _, _, game_over, _, info = env.step(action1)
                    if game_over:
                        winner = env.get_winner()
                        if winner == 1:
                            wins += 1
                        break
                
                # 对手行动
                if not game_over:
                    obs = env._get_observation()
                    action2 = random_opponent.get_action(obs, env)
                    if action2 is not None:
                        _, _, game_over, _, info = env.step(action2)
                
                steps += 1
        
        win_rate = wins / total_games * 100
        results[algorithm] = win_rate
        
        print(f"  胜率: {win_rate:.1f}% ({wins}/{total_games})")
        print(f"  AI描述: {ai.get_info()['description']}")
    
    # 显示比较结果
    print(f"\n=== 算法性能排名 ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (algorithm, win_rate) in enumerate(sorted_results, 1):
        print(f"{i}. {algorithm.upper()}: {win_rate:.1f}%")


def example_custom_search_ai():
    """示例：创建自定义配置的搜索AI"""
    print("\n=== 自定义搜索AI示例 ===")
    
    # 创建一个高深度的A*搜索AI
    custom_ai = SearchAI(
        name="HighDepth-A*-AI",
        player_id=1,
        search_algorithm="astar",
        max_depth=30  # 更大的搜索深度
    )
    
    print(f"自定义AI信息: {custom_ai.get_info()}")
    
    # 在小地图上测试高深度搜索的效果
    env = SnakeEnv(board_size=6)  # 较小的地图
    env.reset()
    
    print("在小地图上测试高深度搜索...")
    
    # 运行几步看看AI的决策
    for step in range(5):
        obs = env._get_observation()
        action = custom_ai.get_action(obs, env)
        
        if action is not None:
            print(f"步骤 {step + 1}: AI选择动作 {action}")
            _, _, done, _, info = env.step(action)
            
            if done:
                print("游戏结束")
                break
        else:
            print(f"步骤 {step + 1}: AI无法选择动作")
            break
    
    print("自定义AI测试完成")


def main():
    """主示例函数"""
    print("搜索算法AI使用示例")
    print("=" * 50)
    
    # 运行各种示例
    example_bfs_snake()
    example_astar_snake()
    example_dfs_sokoban()
    example_algorithm_comparison()
    example_custom_search_ai()
    
    print("\n" + "=" * 50)
    print("所有示例完成!")
    
    # 显示使用提示
    print("\n使用提示:")
    print("1. BFS适合寻找最短路径，但内存消耗大")
    print("2. DFS搜索深度大，但可能找不到最优路径")
    print("3. A*结合了BFS和启发式，通常性能最好")
    print("4. 可以通过调整max_depth参数控制搜索深度")
    print("5. 不同游戏类型会自动选择相应的搜索策略")


if __name__ == "__main__":
    main()
