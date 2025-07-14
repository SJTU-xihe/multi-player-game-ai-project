"""
测试优化后的推箱子AI
验证AI是否具备主动推箱子完成游戏的意识
"""

import numpy as np
import time
from agents.ai_bots.sokoban_ai import SokobanAI, SimpleSokobanAI
from agents.ai_bots.llm_bot import LLMBot, AdvancedSokobanAI


def create_test_observation(scenario="basic"):
    """创建测试观察数据"""
    if scenario == "basic":
        # 基础测试场景：玩家可以直接推箱子到目标
        board = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 3, 2, 1],  # 空地 - 箱子(3) - 目标(2)
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ])
        player_pos = [1, 1]  # 玩家在箱子左边
    elif scenario == "multi_box":
        # 多箱子场景
        board = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 3, 0, 2, 1],  # 空地 - 空地 - 箱子 - 空地 - 目标
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 3, 0, 2, 0, 1],  # 另一个箱子和目标
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ])
        player_pos = [1, 1]  # 玩家在左侧
    elif scenario == "complex":
        # 复杂场景：需要策略性思考
        board = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 3, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 3, 0, 0, 3, 1],
            [1, 2, 0, 0, 1, 0, 2, 1],
            [1, 0, 0, 2, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ])
        player_pos = [1, 1]
    
    observation = {
        'board': board,
        'player1_pos': np.array(player_pos),
        'player2_pos': np.array([-1, -1]),
        'current_player': 1,
        'player1_score': 0,
        'player2_score': 0,
        'boxes_left': 1,
        'progress': 0.0,
        'valid_actions_mask': np.array([True, True, True, True])
    }
    
    return observation


class MockEnv:
    """模拟游戏环境"""
    def __init__(self):
        pass
    
    def get_valid_actions(self):
        return ['UP', 'DOWN', 'LEFT', 'RIGHT']


def print_board_with_symbols(board):
    """打印带符号的游戏棋盘"""
    symbols = {
        0: '·',  # 空地
        1: '█',  # 墙壁
        2: '○',  # 目标点
        3: '□',  # 箱子
        4: '■',  # 箱子在目标上
        5: '♂',  # 玩家1
        6: '♀',  # 玩家2
        7: '◎',  # 玩家1在目标上
        8: '◇'   # 玩家2在目标上
    }
    
    print("棋盘状态:")
    for row in board:
        print(' '.join(symbols.get(cell, '?') for cell in row))
    print()


def analyze_ai_intelligence(ai, observation, env, scenario_name):
    """分析AI的智能程度"""
    print(f"=== {ai.name} - {scenario_name} ===")
    print_board_with_symbols(observation['board'])
    
    # 测试多次决策，看是否一致
    actions = []
    times = []
    
    for i in range(5):
        start_time = time.time()
        action = ai.get_action(observation, env)
        elapsed_time = time.time() - start_time
        
        actions.append(action)
        times.append(elapsed_time)
    
    # 分析结果
    avg_time = sum(times) / len(times)
    unique_actions = set(actions)
    most_common_action = max(set(actions), key=actions.count)
    consistency = actions.count(most_common_action) / len(actions)
    
    print(f"决策分析:")
    print(f"  平均响应时间: {avg_time:.4f}秒")
    print(f"  动作选择: {actions}")
    print(f"  最常选择: {most_common_action} (一致性: {consistency:.1%})")
    print(f"  动作多样性: {len(unique_actions)}/5")
    
    # 评估动作质量
    board = observation['board']
    player_pos = tuple(observation['player1_pos'])
    
    # 检查是否选择推箱子动作
    directions = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1)
    }
    
    push_score = 0
    target_score = 0
    
    if most_common_action in directions:
        dr, dc = directions[most_common_action]
        target_pos = (player_pos[0] + dr, player_pos[1] + dc)
        
        # 确保目标位置在棋盘内
        if (0 <= target_pos[0] < board.shape[0] and 
            0 <= target_pos[1] < board.shape[1]):
            
            # 检查是否推箱子
            if board[target_pos[0], target_pos[1]] == 3:  # 有箱子
                push_score = 10
                
                # 检查推箱子方向
                box_new_pos = (target_pos[0] + dr, target_pos[1] + dc)
                if (0 <= box_new_pos[0] < board.shape[0] and 
                    0 <= box_new_pos[1] < board.shape[1]):
                    
                    if board[box_new_pos[0], box_new_pos[1]] == 2:  # 推到目标
                        target_score = 20
                        print(f"  🎯 选择完美动作：将箱子推到目标！")
                    elif board[box_new_pos[0], box_new_pos[1]] == 0:  # 推到空地
                        target_score = 5
                        print(f"  📦 选择推箱子动作，但未到目标")
                    else:
                        target_score = -5
                        print(f"  ❌ 推箱子方向错误（推向墙壁或其他箱子）")
                else:
                    target_score = -5
                    print(f"  ❌ 推箱子越界")
            else:
                print(f"  🚶 选择移动动作（未推箱子）")
        else:
            print(f"  🚶 选择移动动作（越界）")
    
    intelligence_score = push_score + target_score
    print(f"  智能评分: {intelligence_score}/30")
    
    if intelligence_score >= 25:
        print(f"  🌟 AI表现优秀！具备强推箱子意识")
    elif intelligence_score >= 15:
        print(f"  👍 AI表现良好，有一定推箱子意识")
    elif intelligence_score >= 5:
        print(f"  🤔 AI表现一般，推箱子意识不强")
    else:
        print(f"  😞 AI表现较差，缺乏推箱子意识")
    
    print()
    return intelligence_score


def main():
    """主测试函数"""
    print("🧠 推箱子AI智能测试")
    print("=" * 50)
    print("测试AI是否具备主动推箱子完成游戏的意识")
    print()
    
    # 创建不同的AI
    ais = [
        SimpleSokobanAI("简单AI", player_id=1),
        SokobanAI("优化搜索AI", player_id=1, max_search_time=2.0, use_dynamic_depth=True),
        LLMBot("LLM AI", player_id=1, use_local_simulation=True),
        AdvancedSokobanAI("混合AI", player_id=1, strategy='hybrid')
    ]
    
    # 测试场景
    scenarios = [
        ("基础推箱子", "basic"),
        ("多箱子场景", "multi_box"),
        ("复杂策略场景", "complex")
    ]
    
    env = MockEnv()
    results = {}
    
    # 对每个AI和场景进行测试
    for scenario_name, scenario_key in scenarios:
        print(f"\n📋 测试场景: {scenario_name}")
        print("-" * 40)
        
        observation = create_test_observation(scenario_key)
        
        for ai in ais:
            score = analyze_ai_intelligence(ai, observation, env, scenario_name)
            
            if ai.name not in results:
                results[ai.name] = []
            results[ai.name].append(score)
    
    # 生成综合报告
    print("\n📊 综合智能评估报告")
    print("=" * 50)
    
    for ai_name, scores in results.items():
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        print(f"{ai_name}:")
        print(f"  平均得分: {avg_score:.1f}/30")
        print(f"  最高得分: {max_score}/30")
        print(f"  最低得分: {min_score}/30")
        print(f"  表现稳定性: {'高' if max_score - min_score <= 5 else '中' if max_score - min_score <= 10 else '低'}")
        
        if avg_score >= 20:
            print(f"  🏆 总评: 优秀 - 具备强烈的推箱子完成意识")
        elif avg_score >= 15:
            print(f"  🥈 总评: 良好 - 有一定的策略意识")
        elif avg_score >= 10:
            print(f"  🥉 总评: 一般 - 基础的游戏理解")
        else:
            print(f"  📖 总评: 需改进 - 缺乏游戏策略意识")
        print()
    
    # 排名
    sorted_ais = sorted(results.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
    
    print("🏆 AI智能排名:")
    for i, (ai_name, scores) in enumerate(sorted_ais):
        avg_score = sum(scores) / len(scores)
        print(f"{i+1}. {ai_name}: {avg_score:.1f}/30")
    
    print("\n✅ 测试完成！")
    print("推荐使用排名较高的AI进行推箱子游戏。")


if __name__ == "__main__":
    main()
