#!/usr/bin/env python3
"""
游戏启动脚本
让用户选择不同的游戏模式
"""

import sys
import os
import subprocess


def main():
    print("=" * 50)
    print("🎮 多游戏AI对战平台")
    print("=" * 50)
    print()
    print("请选择游戏模式:")
    print("1. 多游戏GUI - 五子棋、贪吃蛇和推箱子 (推荐)")
    print("2. 贪吃蛇专用GUI - 更好的贪吃蛇体验")
    print("3. 推箱子专用GUI - 专业推箱子体验 🎯")
    print("4. 五子棋命令行版本")
    print("5. 贪吃蛇命令行版本")
    print("6. 搜索算法AI演示 🔍")
    print("7. 运行测试")
    print("8. 退出")
    print()

    while True:
        try:
            choice = input("请输入选择 (1-8): ").strip()

            if choice == "1":
                print("\n🎯 启动多游戏图形界面...")
                print("支持:")
                print("- 五子棋: 鼠标点击落子")
                print("- 贪吃蛇: 方向键/WASD控制")
                print("- 推箱子: 方向键/WASD移动推箱子")
                print("- 多种AI难度选择")
                print("- 暂停/继续功能")
                print()

                # 检查GUI文件是否存在
                if os.path.exists("gui_game.py"):
                    subprocess.run([sys.executable, "gui_game.py"])
                else:
                    print("❌ GUI文件未找到，请检查项目文件")
                break

            elif choice == "2":
                print("\n🐍 启动贪吃蛇专用图形界面...")
                print("特性:")
                print("- 专为贪吃蛇优化的界面")
                print("- 更流畅的游戏体验")
                print("- 多种贪吃蛇AI算法")
                print("- 实时状态显示")
                print()

                if os.path.exists("snake_gui.py"):
                    subprocess.run([sys.executable, "snake_gui.py"])
                else:
                    print("❌ 贪吃蛇GUI文件未找到")
                break

            elif choice == "3":
                print("\n📦 启动推箱子专用图形界面...")
                print("特性:")
                print("- 专为推箱子优化的界面")
                print("- 双人对战模式 (竞争/合作)")
                print("- Smart AI 和 Simple AI")
                print("- 多个精心设计的关卡")
                print("- 关卡编辑器功能")
                print("- 提示和撤销功能")
                print()

                if os.path.exists("sokoban_gui.py"):
                    subprocess.run([sys.executable, "sokoban_gui.py"])
                else:
                    print("❌ 推箱子GUI文件未找到")
                break

            elif choice == "4":
                print("\n♟️  启动五子棋命令行版本...")
                subprocess.run(
                    [
                        sys.executable,
                        "main.py",
                        "--game",
                        "gomoku",
                        "--player1",
                        "human",
                        "--player2",
                        "random",
                    ]
                )
                break

            elif choice == "5":
                print("\n🐍 启动贪吃蛇命令行版本...")
                subprocess.run(
                    [
                        sys.executable,
                        "main.py",
                        "--game",
                        "snake",
                        "--player1",
                        "human",
                        "--player2",
                        "snake_ai",
                    ]
                )
                break

            elif choice == "6":
                print("\n🔍 搜索算法AI演示...")
                print("演示不同搜索算法的表现:")
                print("- BFS (广度优先搜索)")
                print("- DFS (深度优先搜索)")  
                print("- A* (启发式搜索)")
                print()
                
                if os.path.exists("test_search_ai.py"):
                    subprocess.run([sys.executable, "test_search_ai.py"])
                elif os.path.exists("examples/search_ai_examples.py"):
                    subprocess.run([sys.executable, "examples/search_ai_examples.py"])
                else:
                    print("❌ 搜索AI演示文件未找到")
                break

            elif choice == "7":
                print("\n🧪 运行项目测试...")
                subprocess.run([sys.executable, "test_project.py"])
                break

            elif choice == "8":
                print("\n👋 再见！")
                sys.exit(0)

            else:
                print("❌ 无效选择，请输入 1-8")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 再见！")
            sys.exit(0)


if __name__ == "__main__":
    main()
