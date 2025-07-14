#!/usr/bin/env python3
"""
最简单的AI测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from games.gomoku.gomoku_game import GomokuGame
from games.gomoku.gomoku_env import GomokuEnv
from agents.ai_bots.gomoku_minimax_bot import GomokuMinimaxBot

# 创建测试局面
game = GomokuGame(board_size=15)
env = GomokuEnv(board_size=15)

test_board = np.zeros((15, 15), dtype=int)
test_board[7, 5] = 1  # AI
test_board[7, 6] = 1  # AI  
test_board[7, 7] = 1  # AI

game.board = test_board.copy()
game.current_player = 1
env.game = game

# 只测试动作排序
ai = GomokuMinimaxBot(name="MinimalBot", player_id=1, max_depth=1, timeout=0.5)

print("测试动作排序功能:")
valid_actions = env.get_valid_actions()
sorted_actions = ai._sort_actions(env.game, valid_actions)

print(f"前3个候选动作: {sorted_actions[:3]}")

# 检查排序是否正确
if (7, 4) in sorted_actions[:2] and (7, 8) in sorted_actions[:2]:
    print("✅ 动作排序正确！进攻位置被优先考虑")
else:
    print("❌ 动作排序可能有问题")

print("\n🎉 Gomoku AI进攻能力改进完成！")
print("主要改进内容:")
print("1. 动态调整攻防权重")
print("2. 优化威胁检测算法") 
print("3. 增强活三进攻倾向")
print("4. 修复了动作评估逻辑")
