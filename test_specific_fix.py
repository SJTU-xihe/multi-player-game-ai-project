"""
测试特定的推箱子AI修复
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.ai_bots.sokoban_ai import SokobanAI


def create_simple_test():
    """创建最简单的测试场景"""
    # 玩家(5) - 箱子(3) - 目标(2)
    board = np.array([
        [1, 1, 1, 1, 1],
        [1, 5, 3, 2, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])
    
    observation = {
        'board': board,
        'player1_pos': np.array([1, 1]),
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
    def get_valid_actions(self):
        return ['UP', 'DOWN', 'LEFT', 'RIGHT']


def test_ai_decision():
    """测试AI决策"""
    ai = SokobanAI(player_id=1, name="测试AI")
    observation = create_simple_test()
    env = MockEnv()
    
    print("测试场景:")
    print("█ █ █ █ █")
    print("█ ♂ □ ○ █")  # 玩家-箱子-目标
    print("█ · · · █")
    print("█ █ █ █ █")
    print()
    print("期望动作: RIGHT (推箱子到目标)")
    print()
    
    # 测试多次
    actions = []
    for i in range(10):
        action = ai.get_action(observation, env)
        actions.append(action)
        print(f"第{i+1}次决策: {action}")
    
    print(f"\n动作统计: {actions}")
    
    # 检查结果
    right_count = actions.count('RIGHT')
    print(f"选择RIGHT的次数: {right_count}/10")
    
    if right_count >= 8:
        print("✅ AI智能测试通过！")
        return True
    else:
        print("❌ AI智能测试失败！")
        return False


if __name__ == "__main__":
    test_ai_decision()
