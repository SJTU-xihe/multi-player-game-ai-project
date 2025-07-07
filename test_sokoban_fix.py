"""
验证Sokoban在主GUI中的显示修复
"""

def test_sokoban_gui_integration():
    """测试Sokoban在主GUI中的集成"""
    print("=== 测试Sokoban主GUI显示修复 ===")
    
    try:
        # 测试导入
        from gui_game import MultiGameGUI
        print("✅ MultiGameGUI导入成功")
        
        # 测试Sokoban相关导入
        try:
            from games.sokoban import SokobanGame, SokobanEnv
            print("✅ Sokoban游戏模块导入成功")
            
            # 测试MultiGameGUI的Sokoban方法是否存在
            gui = MultiGameGUI()
            
            # 检查是否有_draw_sokoban方法
            if hasattr(gui, '_draw_sokoban'):
                print("✅ _draw_sokoban方法存在")
            else:
                print("❌ _draw_sokoban方法缺失")
            
            # 检查是否有_draw_sokoban_cell方法
            if hasattr(gui, '_draw_sokoban_cell'):
                print("✅ _draw_sokoban_cell方法存在")
            else:
                print("❌ _draw_sokoban_cell方法缺失")
            
            # 测试切换到Sokoban游戏
            gui._switch_game("sokoban")
            print("✅ 成功切换到Sokoban游戏")
            
            # 检查游戏环境是否正确创建
            if gui.env and hasattr(gui.env, 'game'):
                print("✅ Sokoban环境创建成功")
            else:
                print("❌ Sokoban环境创建失败")
                
        except ImportError as e:
            print(f"⚠️ Sokoban游戏模块不可用: {e}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    print("\n=== 修复验证完成 ===")
    return True

if __name__ == "__main__":
    test_sokoban_gui_integration()
