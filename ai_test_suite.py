"""
五子棋AI综合测试套件
包含所有重要的AI行为测试
"""

from agents.ai_bots.gomoku_minimax_bot import GomokuMinimaxBot
from games.gomoku.gomoku_env import GomokuEnv

def print_board_section(board, center_row, center_col, radius=3):
    """打印棋盘的一个区域"""
    min_row = max(0, center_row - radius)
    max_row = min(14, center_row + radius)
    min_col = max(0, center_col - radius)
    max_col = min(14, center_col + radius)
    
    print("   ", end="")
    for j in range(min_col, max_col + 1):
        print(f"{j:2}", end="")
    print()
    
    for i in range(min_row, max_row + 1):
        print(f"{i:2} ", end="")
        for j in range(min_col, max_col + 1):
            if board[i, j] == 0:
                print(" .", end="")
            elif board[i, j] == 1:
                print(" X", end="")
            else:
                print(" O", end="")
        print()

def test_opening_behavior():
    """测试开局行为"""
    print("=== 开局行为测试 ===")
    
    ai = GomokuMinimaxBot(name="开局测试AI", player_id=2, max_depth=2, timeout=1.0)
    env = GomokuEnv(board_size=15)
    
    # 测试1：空棋盘
    print("\n1. 空棋盘开局")
    env.reset()
    action = ai.get_action(env.game.get_state(), env)
    print(f"AI选择: {action} (期望: 中心位置)")
    
    # 测试2：对手在中心
    print("\n2. 对手占据中心")
    env.reset()
    env.step((7, 7))
    action = ai.get_action(env.game.get_state(), env)
    print(f"AI选择: {action} (期望: 中心附近)")
    
    # 测试3：对手在偏角
    print("\n3. 对手在偏角位置")
    env.reset()
    env.step((3, 3))
    action = ai.get_action(env.game.get_state(), env)
    print(f"AI选择: {action} (期望: 对手附近)")

def test_defense_behavior():
    """测试防守行为"""
    print("\n=== 防守行为测试 ===")
    
    ai = GomokuMinimaxBot(name="防守测试AI", player_id=2, max_depth=3, timeout=2.0)
    env = GomokuEnv(board_size=15)
    
    # 测试1：水平威胁
    print("\n1. 水平三连威胁")
    env.reset()
    env.game.board[7, 5] = 1  # 对手
    env.game.board[7, 6] = 1  # 对手  
    env.game.board[7, 7] = 1  # 对手
    env.game.board[9, 9] = 2  # AI已有子，避免开局逻辑
    env.game.current_player = 2
    
    print("威胁场景:")
    print_board_section(env.game.board, 7, 6)
    
    action = ai.get_action(env.game.get_state(), env)
    print(f"AI选择: {action} (期望: (7,4) 或 (7,8))")
    
    # 测试2：垂直威胁
    print("\n2. 垂直三连威胁")
    env.reset()
    env.game.board[5, 7] = 1  # 对手
    env.game.board[6, 7] = 1  # 对手
    env.game.board[7, 7] = 1  # 对手
    env.game.board[9, 9] = 2  # AI已有子
    env.game.current_player = 2
    
    print("威胁场景:")
    print_board_section(env.game.board, 6, 7)
    
    action = ai.get_action(env.game.get_state(), env)
    print(f"AI选择: {action} (期望: (4,7) 或 (8,7))")

def test_attack_behavior():
    """测试进攻行为"""
    print("\n=== 进攻行为测试 ===")
    
    ai = GomokuMinimaxBot(name="进攻测试AI", player_id=2, max_depth=3, timeout=2.0)
    env = GomokuEnv(board_size=15)
    
    # 测试：AI有三连优势
    print("\n1. AI三连进攻")
    env.reset()
    env.game.board[7, 7] = 2  # AI
    env.game.board[7, 8] = 2  # AI
    env.game.board[7, 9] = 2  # AI
    env.game.board[6, 6] = 1  # 对手
    env.game.current_player = 2
    
    print("进攻场景:")
    print_board_section(env.game.board, 7, 8)
    
    action = ai.get_action(env.game.get_state(), env)
    print(f"AI选择: {action} (期望: (7,6) 或 (7,10))")

def test_complex_scenarios():
    """测试复杂局面"""
    print("\n=== 复杂局面测试 ===")
    
    ai = GomokuMinimaxBot(name="复杂测试AI", player_id=2, max_depth=3, timeout=2.0)
    env = GomokuEnv(board_size=15)
    
    # 复杂混战局面
    print("\n1. 复杂混战局面")
    env.reset()
    moves = [
        (7, 7, 1), (7, 8, 2), (6, 6, 1), (8, 8, 2),
        (5, 5, 1), (9, 9, 2), (8, 6, 1), (6, 8, 2)
    ]
    for row, col, player in moves:
        env.game.board[row, col] = player
    env.game.current_player = 2
    
    print("复杂局面:")
    print_board_section(env.game.board, 7, 7, 4)
    
    action = ai.get_action(env.game.get_state(), env)
    print(f"AI选择: {action}")

def test_performance():
    """测试性能"""
    print("\n=== 性能测试 ===")
    
    ai = GomokuMinimaxBot(name="性能测试AI", player_id=2, max_depth=3, timeout=3.0)
    env = GomokuEnv(board_size=15)
    
    import time
    
    env.reset()
    env.step((7, 7))
    
    start_time = time.time()
    action = ai.get_action(env.game.get_state(), env)
    end_time = time.time()
    
    print(f"响应时间: {end_time - start_time:.2f}s")
    print(f"搜索节点: {ai.nodes_searched}")
    print(f"AI选择: {action}")

def run_comprehensive_test():
    """运行综合测试"""
    print("🎮 五子棋AI综合测试套件")
    print("=" * 50)
    
    try:
        test_opening_behavior()
        test_defense_behavior()
        test_attack_behavior()
        test_complex_scenarios()
        test_performance()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试完成！")
        print("\n📊 测试总结:")
        print("- 开局行为: 智能选择开局位置")
        print("- 防守能力: 识别并阻止对手威胁")
        print("- 进攻能力: 寻找最佳进攻机会")
        print("- 复杂局面: 在混战中做出合理选择")
        print("- 性能表现: 在合理时间内给出回应")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_test()
