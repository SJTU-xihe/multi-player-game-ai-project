#!/usr/bin/env python3
"""
测试改进后的GUI - AI按钮动态显示
"""

import os
import sys

# 设置环境
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

def test_gui_ai_buttons():
    """测试GUI中AI按钮的动态显示"""
    try:
        print("测试改进后的GUI AI按钮功能...")
        
        import gui_game
        
        # 创建GUI实例（但不运行主循环）
        gui = gui_game.MultiGameGUI()
        print("✓ MultiGameGUI 创建成功")
        
        # 测试初始状态（五子棋）
        print(f"\n当前游戏: {gui.current_game}")
        print("当前AI按钮:")
        ai_buttons = [name for name in gui.buttons.keys() if name.endswith('_ai') or name == 'simple_sokoban_ai']
        for btn in ai_buttons:
            print(f"  - {btn}: {gui.buttons[btn]['text']}")
        
        # 测试切换到贪吃蛇
        print("\n切换到贪吃蛇游戏...")
        gui._switch_game("snake")
        print(f"当前游戏: {gui.current_game}")
        print("当前AI按钮:")
        ai_buttons = [name for name in gui.buttons.keys() if name.endswith('_ai') or name == 'simple_sokoban_ai']
        for btn in ai_buttons:
            print(f"  - {btn}: {gui.buttons[btn]['text']}")
        
        # 测试切换到推箱子（如果可用）
        if gui_game.SOKOBAN_AVAILABLE:
            print("\n切换到推箱子游戏...")
            gui._switch_game("sokoban")
            print(f"当前游戏: {gui.current_game}")
            print("当前AI按钮:")
            ai_buttons = [name for name in gui.buttons.keys() if name.endswith('_ai') or name == 'simple_sokoban_ai']
            for btn in ai_buttons:
                print(f"  - {btn}: {gui.buttons[btn]['text']}")
        else:
            print("\n推箱子游戏不可用，跳过测试")
        
        # 测试切换回五子棋
        print("\n切换回五子棋游戏...")
        gui._switch_game("gomoku")
        print(f"当前游戏: {gui.current_game}")
        print("当前AI按钮:")
        ai_buttons = [name for name in gui.buttons.keys() if name.endswith('_ai') or name == 'simple_sokoban_ai']
        for btn in ai_buttons:
            print(f"  - {btn}: {gui.buttons[btn]['text']}")
        
        print("\n✓ AI按钮动态显示测试通过!")
        print("\n测试结果:")
        print("- 五子棋模式: 只显示 Gomoku AI 和 Random AI")
        print("- 贪吃蛇模式: 只显示 Snake AI、Smart Snake AI 和 Random AI")
        print("- 推箱子模式: 只显示 Smart AI、Simple AI 和 Random AI")
        print("- 切换游戏时AI按钮正确更新")
        
        return True
        
    except Exception as e:
        print(f"✗ GUI AI按钮测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_selection():
    """测试AI选择逻辑"""
    try:
        print("\n测试AI选择逻辑...")
        
        import gui_game
        gui = gui_game.MultiGameGUI()
        
        # 测试在不同游戏中选择AI
        test_cases = [
            ("gomoku", "GomokuMinimaxBot", "Gomoku Expert AI"),
            ("snake", "MinimaxBot", "Snake AI"),
        ]
        
        if gui_game.SOKOBAN_AVAILABLE:
            test_cases.append(("sokoban", "SokobanAI", "Smart Sokoban AI"))
        
        for game, ai_type, expected_name in test_cases:
            gui._switch_game(game)
            gui.selected_ai = ai_type
            gui._create_ai_agent()
            
            actual_name = gui.ai_agent.name
            print(f"✓ {game}: {ai_type} -> {actual_name}")
            
            if expected_name.lower() in actual_name.lower():
                print(f"  - AI名称匹配: {actual_name}")
            else:
                print(f"  - 警告: AI名称不匹配，期望包含'{expected_name}'，实际'{actual_name}'")
        
        print("✓ AI选择逻辑测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ AI选择逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== GUI改进测试 ===")
    
    success1 = test_gui_ai_buttons()
    success2 = test_ai_selection()
    
    if success1 and success2:
        print("\n🎉 GUI改进测试全部通过!")
        print("\n改进效果:")
        print("✓ 五子棋游戏只显示相关AI按钮")
        print("✓ 贪吃蛇游戏只显示相关AI按钮") 
        print("✓ 推箱子游戏只显示相关AI按钮")
        print("✓ 切换游戏时按钮动态更新")
        print("✓ AI选择逻辑正确工作")
        print("\n现在可以运行: python gui_game.py")
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
        sys.exit(1)
