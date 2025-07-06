"""
最终验证AI行为是否正常
"""

from agents.ai_bots.gomoku_minimax_bot import GomokuMinimaxBot
from games.gomoku.gomoku_env import GomokuEnv

def final_verification():
    """最终验证AI行为"""
    print("=== 最终AI行为验证 ===")
    
    ai = GomokuMinimaxBot(name="最终测试AI", player_id=2, max_depth=3, timeout=2.0)
    env = GomokuEnv(board_size=15)
    
    # 场景1：模拟实际对局
    print("\n场景1：模拟实际对局开始几步")
    env.reset()
    
    moves = [
        (7, 7, 1, "人类玩家开局中心"),
        (None, None, 2, "AI第一步"),
        (6, 6, 1, "人类玩家进攻"),
        (None, None, 2, "AI第二步"),
        (8, 8, 1, "人类玩家继续"),
        (None, None, 2, "AI第三步")
    ]
    
    for i, (row, col, player, desc) in enumerate(moves):
        if player == 1:
            # 人类玩家走子
            env.step((row, col))
            print(f"{desc}: ({row}, {col})")
        else:
            # AI走子
            action = ai.get_action(env.game.get_state(), env)
            env.step(action)
            print(f"{desc}: {action}")
        
        # 打印当前棋盘状态
        if i % 2 == 1:  # 每两步打印一次
            print("当前棋盘:")
            print_simple_board(env.game.board)
            print()
    
    # 场景2：威胁与防守
    print("\n场景2：威胁与防守测试")
    env.reset()
    
    # 设置一个对手快要获胜的局面
    threat_moves = [(6, 6), (6, 7), (6, 8)]  # 对手三连
    for move in threat_moves:
        env.game.board[move[0], move[1]] = 1
    
    env.game.current_player = 2  # AI回合
    print("危险局面（对手三连）:")
    print_simple_board(env.game.board)
    
    action = ai.get_action(env.game.get_state(), env)
    print(f"AI防守选择: {action}")
    print(f"预期位置: (6, 5)或(6, 9)")

def print_simple_board(board):
    """简化版棋盘打印"""
    for i in range(5, 10):  # 只打印中心区域
        print(f"{i} ", end="")
        for j in range(5, 10):
            if board[i, j] == 0:
                print(".", end=" ")
            elif board[i, j] == 1:
                print("X", end=" ")
            else:
                print("O", end=" ")
        print()
    print("  5 6 7 8 9")

if __name__ == "__main__":
    final_verification()
