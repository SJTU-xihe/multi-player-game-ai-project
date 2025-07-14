"""
快速调试测试
"""
import numpy as np
from agents.ai_bots.sokoban_ai import SokobanAI, SimpleSokobanAI
from agents.ai_bots.llm_bot import LLMBot, AdvancedSokobanAI

def quick_test():
    # 创建基础测试场景
    board = np.array([
        [1, 1, 1, 1, 1],
        [1, 5, 3, 2, 1],  # 玩家(5) - 箱子(3) - 目标(2)
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

    class MockEnv:
        def get_valid_actions(self):
            return ['UP', 'DOWN', 'LEFT', 'RIGHT']

    env = MockEnv()

    # 测试所有AI
    ais = [
        SimpleSokobanAI('简单AI', player_id=1),
        SokobanAI('优化搜索AI', player_id=1),
        LLMBot('LLM AI', player_id=1, use_local_simulation=True),
        AdvancedSokobanAI('混合AI', player_id=1, strategy='hybrid')
    ]

    for ai in ais:
        try:
            action = ai.get_action(observation, env)
            print(f'{ai.name}: {action}')
        except Exception as e:
            print(f'{ai.name}: 错误 - {e}')

if __name__ == "__main__":
    quick_test()
