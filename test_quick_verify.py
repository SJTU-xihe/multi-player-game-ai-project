#!/usr/bin/env python3
"""
快速验证AI选择
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from games.gomoku.gomoku_game import GomokuGame
from games.gomoku.gomoku_env import GomokuEnv
from agents.ai_bots.gomoku_minimax_bot import GomokuMinimaxBot

def quick_test():
    # 创建测试局面
    game = GomokuGame(board_size=15)
    env = GomokuEnv(board_size=15)
    
    # 设置活三局面
    test_board = np.zeros((15, 15), dtype=int)
    test_board[7, 5] = 1  # AI
    test_board[7, 6] = 1  # AI  
    test_board[7, 7] = 1  # AI
    
    game.board = test_board.copy()
    game.current_player = 1
    env.game = game
    
    # 创建AI，使用最小搜索深度
    ai = GomokuMinimaxBot(name="QuickBot", player_id=1, max_depth=1, timeout=1.0)
    
    print("简单活三局面测试:")
    print("AI应该选择 (7,4) 或 (7,8) 来形成冲四")
    
    action = ai.get_action(None, env)
    print(f"\nAI选择: {action}")
    
    if action in [(7, 4), (7, 8)]:
        print("✅ 测试通过！AI正确选择了进攻性着法")
        return True
    else:
        print(f"❌ 测试失败！AI应该选择 (7,4) 或 (7,8)")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 Gomoku AI进攻能力改进成功！")
        print("主要改进:")
        print("- 增强了活三识别和进攻倾向")
        print("- 优化了动作排序，优先考虑进攻机会")
        print("- 动态调整防守权重，平衡攻守")
        print("- 修复了威胁检测的准确性")
    else:
        print("\n❌ 还需要进一步调试")
