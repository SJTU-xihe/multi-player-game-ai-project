#!/usr/bin/env python3
"""
简化的Gomoku AI进攻测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from games.gomoku.gomoku_game import GomokuGame
from games.gomoku.gomoku_env import GomokuEnv
from agents.ai_bots.gomoku_minimax_bot import GomokuMinimaxBot

def test_simple_attack():
    """测试简单的活三进攻场景"""
    print("=" * 50)
    print("简化的进攻测试")
    print("=" * 50)
    
    # 创建测试局面
    game = GomokuGame(board_size=15)
    env = GomokuEnv(board_size=15)
    
    # 设置一个简单的活三局面
    test_board = np.zeros((15, 15), dtype=int)
    
    # AI的三个连子，形成活三
    test_board[7, 5] = 1  # AI
    test_board[7, 6] = 1  # AI  
    test_board[7, 7] = 1  # AI
    # (7,4) 和 (7,8) 是空位，可以形成冲四
    
    game.board = test_board.copy()
    game.current_player = 1  # AI的回合
    env.game = game
    
    print("测试局面:")
    print_board(game.board)
    print()
    
    # 创建AI，使用更短的搜索深度
    ai = GomokuMinimaxBot(name="TestBot", player_id=1, max_depth=2, timeout=2.0)
    
    print("分析各个候选位置...")
    
    # 手动测试几个关键位置
    test_positions = [(7, 4), (7, 8), (6, 6), (8, 8)]
    
    for pos in test_positions:
        row, col = pos
        if game.board[row, col] == 0:  # 空位
            # 模拟在这个位置下棋
            temp_board = game.board.copy()
            temp_board[row, col] = 1
            
            print(f"\n位置{pos}:")
            print(f"  下棋后棋盘:")
            print_board_compact(temp_board, highlight_pos=pos)
            
            # 检查威胁
            threats = ai._count_threats(temp_board, 1)
            print(f"  威胁分析: 冲四={threats['rush_four']}, 活三={threats['live_three']}")
            
            # 检查连线长度
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                length = ai._count_consecutive_length(temp_board, row, col, dr, dc, 1)
                if length >= 3:
                    print(f"  方向({dr},{dc}): 连线长度={length}")
    
    print(f"\n{'='*50}")
    print("AI最终选择:")
    action = ai.get_action(None, env)
    print(f"AI选择: {action}")
    
    # 验证选择是否合理
    if action in [(7, 4), (7, 8)]:
        print("✅ AI正确选择了进攻性着法！")
    else:
        print(f"❌ AI可能错过了最佳进攻机会")
        print(f"   推荐位置: (7,4) 或 (7,8)")

def print_board(board):
    """打印完整棋盘"""
    print("  ", end="")
    for i in range(board.shape[1]):
        print(f"{i:2d}", end="")
    print()
    
    for i in range(board.shape[0]):
        print(f"{i:2d}", end="")
        for j in range(board.shape[1]):
            if board[i, j] == 0:
                print(" .", end="")
            elif board[i, j] == 1:
                print(" ●", end="")
            else:
                print(" ○", end="")
        print()

def print_board_compact(board, highlight_pos=None):
    """打印局部棋盘"""
    start_row = max(0, highlight_pos[0] - 2) if highlight_pos else 5
    end_row = min(board.shape[0], start_row + 5)
    start_col = max(0, highlight_pos[1] - 3) if highlight_pos else 3
    end_col = min(board.shape[1], start_col + 7)
    
    print("   ", end="")
    for j in range(start_col, end_col):
        print(f"{j:2d}", end="")
    print()
    
    for i in range(start_row, end_row):
        print(f"{i:2d} ", end="")
        for j in range(start_col, end_col):
            if highlight_pos and i == highlight_pos[0] and j == highlight_pos[1]:
                if board[i, j] == 0:
                    print(" +", end="")  # 高亮新位置
                elif board[i, j] == 1:
                    print(" ◆", end="")  # 高亮AI棋子
                else:
                    print(" ◇", end="")  # 高亮对手棋子
            else:
                if board[i, j] == 0:
                    print(" .", end="")
                elif board[i, j] == 1:
                    print(" ●", end="")
                else:
                    print(" ○", end="")
        print()

if __name__ == "__main__":
    test_simple_attack()
