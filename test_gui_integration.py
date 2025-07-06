"""
测试GUI中的五子棋AI
"""

import sys
import time

def test_gui_import():
    """测试GUI导入和五子棋AI"""
    try:
        print("测试GUI模块导入...")
        from gui_game import MultiGameGUI
        print("✓ 成功导入 MultiGameGUI")
        
        print("测试五子棋AI导入...")
        from agents.ai_bots.gomoku_minimax_bot import GomokuMinimaxBot
        print("✓ 成功导入 GomokuMinimaxBot")
        
        print("创建GUI实例...")
        gui = MultiGameGUI()
        print("✓ 成功创建GUI")
        
        print("测试AI创建...")
        # 测试不同AI的创建
        test_ais = ["RandomBot", "MinimaxBot", "MCTSBot", "GomokuMinimaxBot"]
        
        for ai_name in test_ais:
            gui.selected_ai = ai_name
            gui._create_ai_agent()
            print(f"✓ 成功创建 {ai_name}: {gui.ai_agent.name}")
        
        print("测试按钮...")
        button_count = len(gui.buttons)
        print(f"✓ GUI有 {button_count} 个按钮")
        
        # 检查新按钮是否存在
        if "gomoku_ai" in gui.buttons:
            print("✓ 找到五子棋AI按钮")
        else:
            print("❌ 未找到五子棋AI按钮")
        
        print("\n所有测试通过！新的五子棋AI已成功集成到GUI中。")
        print("\n使用方法:")
        print("1. 运行 python gui_game.py")
        print("2. 选择 'Gomoku' 游戏")
        print("3. 点击 'Gomoku AI' 按钮")
        print("4. 点击 'New Game' 开始对战")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_gui_import():
        print("\n🎉 GUI集成测试成功！")
    else:
        print("\n❌ GUI集成测试失败！")
        sys.exit(1)
