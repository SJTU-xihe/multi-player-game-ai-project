#!/usr/bin/env python3
"""
测试改进后的Gomoku AI的进攻能力
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from games.gomoku.gomoku_game import GomokuGame
from games.gomoku.gomoku_env import GomokuEnv
from agents.ai_bots.gomoku_minimax_bot import GomokuMinimaxBot

def test_attack_scenario():
    """测试AI在活三情况下的进攻能力"""
    print("=" * 60)
    print("测试Gomoku AI的进攻能力")
    print("=" * 60)
    
    # 创建一个有活三机会的测试局面
    game = GomokuGame(board_size=15)
    env = GomokuEnv(board_size=15)
    
    # 设置一个测试局面：AI有活三机会
    # AI是玩家1 (黑子)，人类是玩家2 (白子)
    test_board = np.zeros((15, 15), dtype=int)
    
    # 创建一个AI有活三机会的局面
    # AI的三个连子
    test_board[7, 5] = 1  # AI
    test_board[7, 6] = 1  # AI  
    test_board[7, 7] = 1  # AI
    # (7,4) 和 (7,8) 是空位，形成活三
    
    # 对手的一些棋子
    test_board[6, 6] = 2  # 对手
    test_board[8, 7] = 2  # 对手
    test_board[6, 8] = 2  # 对手
    
    # 设置测试局面
    game.board = test_board.copy()
    game.current_player = 1  # AI的回合
    env.game = game  # 重要：将修改后的game赋值给env
    
    print("初始测试局面:")
    print_board(game.board)
    print()
    
    # 创建AI
    ai = GomokuMinimaxBot(name="AttackBot", player_id=1, max_depth=4)
    
    # 获取AI的动作
    print("AI分析中...")
    action = ai.get_action(None, env)
    
    print(f"\nAI选择的动作: {action}")
    
    # 执行AI动作
    if action:
        game.step(action)
        print("\n执行AI动作后的局面:")
        print_board(game.board)
        
        # 检查AI是否形成了冲四或更强的威胁
        threats = ai._count_threats(game.board, 1)
        print(f"\nAI执行动作后的威胁分析:")
        print(f"连五: {threats['five']}")
        print(f"活四: {threats['live_four']}")
        print(f"冲四: {threats['rush_four']}")
        print(f"活三: {threats['live_three']}")
        print(f"眠三: {threats['sleep_three']}")
        
        # 判断AI是否选择了进攻性的着法
        if action == (7, 4) or action == (7, 8):
            print("\n✅ AI正确选择了进攻性着法，延续活三！")
        else:
            print(f"\n❌ AI可能错过了进攻机会，应该考虑(7,4)或(7,8)")
    
    return action

def test_double_threat_scenario():
    """测试AI在可以形成双威胁时的选择"""
    print("\n" + "=" * 60)
    print("测试AI双威胁进攻能力")
    print("=" * 60)
    
    game = GomokuGame(board_size=15)
    env = GomokuEnv(board_size=15)
    
    # 创建一个可以形成双威胁的局面
    test_board = np.zeros((15, 15), dtype=int)
    
    # AI的棋子布局，可以形成双活三
    test_board[7, 5] = 1  # AI
    test_board[7, 6] = 1  # AI
    # 如果在(7,7)下子，可以与(7,4)形成一个活三
    # 同时，在对角线上也有机会
    test_board[6, 6] = 1  # AI
    test_board[8, 8] = 1  # AI
    # 如果在(7,7)下子，也会在对角线形成威胁
    
    # 对手的棋子
    test_board[6, 7] = 2  # 对手
    test_board[8, 5] = 2  # 对手
    
    game.board = test_board.copy()
    game.current_player = 1
    env.game = game  # 重要：将修改后的game赋值给env
    
    print("双威胁测试局面:")
    print_board(game.board)
    print()
    
    ai = GomokuMinimaxBot(name="AttackBot", player_id=1, max_depth=4)
    action = ai.get_action(None, env)
    
    print(f"\nAI选择的动作: {action}")
    
    if action:
        game.step(action)
        print("\n执行AI动作后的局面:")
        print_board(game.board)
        
        threats = ai._count_threats(game.board, 1)
        print(f"\nAI执行动作后的威胁分析:")
        print(f"活三: {threats['live_three']}")
        print(f"冲四: {threats['rush_four']}")
        
        if threats['live_three'] >= 2:
            print("\n✅ AI成功形成了双活三威胁！")
        elif threats['live_three'] >= 1 or threats['rush_four'] >= 1:
            print("\n✅ AI形成了威胁，但可能还有更好的选择")
        else:
            print("\n❌ AI可能错过了进攻机会")

def print_board(board):
    """打印棋盘"""
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
                print(" ●", end="")  # AI (黑子)
            else:
                print(" ○", end="")  # 对手 (白子)
        print()

def main():
    """主测试函数"""
    try:
        print("开始测试改进后的Gomoku AI...")
        
        # 测试基本进攻场景
        action1 = test_attack_scenario()
        
        # 测试双威胁场景
        action2 = test_double_threat_scenario()
        
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        print("✅ 所有测试完成")
        print("✅ AI进攻能力改进已部署")
        print("✅ 建议在实际对局中验证效果")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
