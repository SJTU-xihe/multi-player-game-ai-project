"""
双人游戏AI框架主程序
"""

import argparse
import sys
import time
from typing import Dict, List, Any
import config

# 导入游戏模块
from games.gomoku import GomokuEnv
from games.snake import SnakeEnv

# 导入智能体模块
from agents import (
    HumanAgent, RandomBot, MinimaxBot, MCTSBot, RLBot, BehaviorTreeBot
)
from agents.ai_bots.search_ai import SearchAI


def create_agent(agent_type: str, player_id: int, name: str = None) -> Any:
    """创建智能体"""
    if name is None:
        name = f"{agent_type}_{player_id}"
    
    # 处理搜索AI的特殊参数
    if agent_type.startswith('search_'):
        algorithm = agent_type.split('_')[1] if '_' in agent_type else 'bfs'
        return SearchAI(name=name, player_id=player_id, search_algorithm=algorithm)
    
    agent_map = {
        'human': HumanAgent,
        'random': RandomBot,
        'minimax': MinimaxBot,
        'mcts': MCTSBot,
        'rl': RLBot,
        'behavior_tree': BehaviorTreeBot,
        'search': SearchAI,  # 默认BFS搜索
        'search_bfs': lambda name, player_id: SearchAI(name=name, player_id=player_id, search_algorithm='bfs'),
        'search_dfs': lambda name, player_id: SearchAI(name=name, player_id=player_id, search_algorithm='dfs'),
        'search_astar': lambda name, player_id: SearchAI(name=name, player_id=player_id, search_algorithm='astar'),
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"不支持的智能体类型: {agent_type}")
    
    agent_class = agent_map[agent_type]
    if callable(agent_class) and not isinstance(agent_class, type):
        # 如果是lambda函数，直接调用
        return agent_class(name, player_id)
    else:
        # 如果是类，实例化
        return agent_class(name=name, player_id=player_id)


def create_env(game_type: str, **kwargs) -> Any:
    """创建游戏环境"""
    env_map = {
        'gomoku': GomokuEnv,
        'snake': SnakeEnv,
        # 'mahjong': MahjongEnv,  # 待实现
    }

    if game_type not in env_map:
        raise ValueError(f"不支持的游戏类型: {game_type}")
    
    return env_map[game_type](**kwargs)


def play_single_game(env: Any, agent1: Any, agent2: Any, render: bool = True) -> Dict[str, Any]:
    """进行单局游戏"""
    print(f"\n=== 开始游戏 ===")
    print(f"玩家1: {agent1.name} ({agent1.__class__.__name__})")
    print(f"玩家2: {agent2.name} ({agent2.__class__.__name__})")
    print(f"游戏类型: {env.__class__.__name__}")
    
    # 重置环境
    observation, info = env.reset()
    
    agents = {1: agent1, 2: agent2}
    step_count = 0
    max_steps = 1000
    
    while not env.is_terminal() and step_count < max_steps:
        current_agent = agents[env.game.current_player]
        
        print(f"\n--- 第 {step_count + 1} 步 ---")
        print(f"当前玩家: {current_agent.name}")
        
        # 获取动作
        action = current_agent.get_action(observation, env)
        print(f"选择动作: {action}")
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        if render:
            env.render()
            time.sleep(0.5)  # 添加延迟以便观察
        
        step_count += 1
        
        if terminated or truncated:
            break
    
    # 获取游戏结果
    winner = env.get_winner()
    if winner == 1:
        result = "玩家1获胜"
        agent1.update_stats('win', 0)
        agent2.update_stats('lose', 0)
    elif winner == 2:
        result = "玩家2获胜"
        agent1.update_stats('lose', 0)
        agent2.update_stats('win', 0)
    else:
        result = "平局"
        agent1.update_stats('draw', 0)
        agent2.update_stats('draw', 0)
    
    print(f"\n=== 游戏结束 ===")
    print(f"结果: {result}")
    print(f"总步数: {step_count}")
    
    return {
        'winner': winner,
        'steps': step_count,
        'result': result,
        'info': info
    }


def evaluate_agents(env: Any, agent1: Any, agent2: Any, num_games: int = 100) -> Dict[str, Any]:
    """评估两个智能体的性能"""
    print(f"\n=== 开始评估 ===")
    print(f"游戏数量: {num_games}")
    print(f"玩家1: {agent1.name}")
    print(f"玩家2: {agent2.name}")
    
    results = {
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'total_steps': 0,
        'avg_steps': 0,
        'games': []
    }
    
    for i in range(num_games):
        if (i + 1) % 10 == 0:
            print(f"进度: {i + 1}/{num_games}")
        
        # 重置智能体
        agent1.reset()
        agent2.reset()
        
        # 进行游戏
        game_result = play_single_game(env, agent1, agent2, render=False)
        results['games'].append(game_result)
        results['total_steps'] += game_result['steps']
        
        if game_result['winner'] == 1:
            results['agent1_wins'] += 1
        elif game_result['winner'] == 2:
            results['agent2_wins'] += 1
        else:
            results['draws'] += 1
        
        # 重置环境
        env.reset()
    
    # 计算统计信息
    results['avg_steps'] = results['total_steps'] / num_games
    results['agent1_win_rate'] = results['agent1_wins'] / num_games
    results['agent2_win_rate'] = results['agent2_wins'] / num_games
    results['draw_rate'] = results['draws'] / num_games
    
    # 打印结果
    print(f"\n=== 评估结果 ===")
    print(f"总游戏数: {num_games}")
    print(f"{agent1.name} 获胜: {results['agent1_wins']} ({results['agent1_win_rate']:.2%})")
    print(f"{agent2.name} 获胜: {results['agent2_wins']} ({results['agent2_win_rate']:.2%})")
    print(f"平局: {results['draws']} ({results['draw_rate']:.2%})")
    print(f"平均步数: {results['avg_steps']:.1f}")
    
    return results


def compare_agents(env: Any, agents: List[Any], num_games: int = 50) -> Dict[str, Any]:
    """比较多个智能体"""
    print(f"\n=== 智能体比较 ===")
    print(f"智能体数量: {len(agents)}")
    print(f"每对对战游戏数: {num_games}")
    
    comparison_results = {}
    
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i >= j:  # 避免重复对战
                continue
            
            print(f"\n--- {agent1.name} vs {agent2.name} ---")
            results = evaluate_agents(env, agent1, agent2, num_games)
            
            key = f"{agent1.name}_vs_{agent2.name}"
            comparison_results[key] = results
    
    return comparison_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="双人游戏AI框架")
    parser.add_argument('--game', type=str, default='gomoku', 
                       choices=['gomoku', 'snake', 'mahjong'],
                       help='游戏类型')
    parser.add_argument('--player1', type=str, default='human',
                       choices=['human', 'random', 'minimax', 'mcts', 'rl', 'behavior_tree', 
                               'search', 'search_bfs', 'search_dfs', 'search_astar'],
                       help='玩家1类型')
    parser.add_argument('--player2', type=str, default='random',
                       choices=['human', 'random', 'minimax', 'mcts', 'rl', 'behavior_tree',
                               'search', 'search_bfs', 'search_dfs', 'search_astar'],
                       help='玩家2类型')
    parser.add_argument('--name1', type=str, help='玩家1名称')
    parser.add_argument('--name2', type=str, help='玩家2名称')
    parser.add_argument('--games', type=int, default=1, help='游戏数量')
    parser.add_argument('--evaluate', action='store_true', help='评估模式')
    parser.add_argument('--compare', action='store_true', help='比较模式')
    parser.add_argument('--no-render', action='store_true', help='不渲染游戏')
    
    # 游戏特定参数
    parser.add_argument('--board-size', type=int, default=15, help='棋盘大小（五子棋）')
    parser.add_argument('--win-length', type=int, default=5, help='获胜长度（五子棋）')
    parser.add_argument('--initial-length', type=int, default=3, help='初始长度（贪吃蛇）')
    parser.add_argument('--food-count', type=int, default=5, help='食物数量（贪吃蛇）')
    
    args = parser.parse_args()
    
    try:
        # 根据游戏类型设置参数
        if args.game == 'gomoku':
            env = create_env(args.game, board_size=args.board_size, win_length=args.win_length)
        elif args.game == 'snake':
            env = create_env(args.game, board_size=args.board_size, 
                           initial_length=args.initial_length, food_count=args.food_count)
        else:
            env = create_env(args.game)
        
        # 创建智能体
        agent1 = create_agent(args.player1, 1, args.name1)
        agent2 = create_agent(args.player2, 2, args.name2)
        
        if args.compare:
            # 比较模式
            all_agents = [agent1, agent2]
            if args.player1 != args.player2:
                # 创建更多智能体进行比较
                for agent_type in ['random', 'minimax', 'mcts']:
                    if agent_type not in [args.player1, args.player2]:
                        all_agents.append(create_agent(agent_type, len(all_agents) + 1))
            
            comparison_results = compare_agents(env, all_agents, args.games)
            
        elif args.evaluate or args.games > 1:
            # 评估模式
            evaluate_agents(env, agent1, agent2, args.games)
            
        else:
            # 单局游戏模式
            play_single_game(env, agent1, agent2, not args.no_render)
    
    except KeyboardInterrupt:
        print("\n游戏被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 