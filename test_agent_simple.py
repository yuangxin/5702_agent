#!/usr/bin/env python3
"""
简化的Agent功能测试脚本
"""

def test_agent_import():
    """测试Agent模块导入"""
    print("🔍 测试Agent模块导入...")
    try:
        from plagiarism_checker.agent import SmartPlagiarismAgent, AgentAnalysis
        print("✅ Agent模块导入成功")
        return True
    except Exception as e:
        print(f"❌ Agent模块导入失败: {e}")
        return False


def test_config_file():
    """测试配置文件"""
    print("🔍 测试配置文件...")
    try:
        import json
        from pathlib import Path
        
        config_path = Path("api_config.json")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print("✅ 配置文件读取成功")
                print(f"   - 配置类型: {list(config.keys())}")
                return True
        else:
            print("❌ api_config.json 文件不存在")
            return False
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False


def test_agent_basic():
    """测试Agent基本初始化"""
    print("🔍 测试Agent基本初始化...")
    try:
        from plagiarism_checker.agent import SmartPlagiarismAgent
        agent = SmartPlagiarismAgent("api_config.json", dual_phase=False)
        print("✅ Agent初始化成功")
        print(f"   - Provider: {agent.provider}")
        print(f"   - Model: {agent.model}")
        return True
    except Exception as e:
        print(f"❌ Agent初始化失败: {e}")
        return False


def test_cli_integration():
    """测试CLI集成"""
    print("🔍 测试CLI集成...")
    try:
        from plagiarism_checker.cli import build_parser
        parser = build_parser()
        # 测试Agent参数是否存在
        test_args = [
            "--submissions-dir", "dataset",
            "--enable-agent",
            "--agent-threshold", "0.7"
        ]
        args = parser.parse_args(test_args)
        print("✅ CLI Agent参数解析成功")
        print(f"   - enable_agent: {args.enable_agent}")
        print(f"   - agent_threshold: {args.agent_threshold}")
        return True
    except Exception as e:
        print(f"❌ CLI集成测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 启动简化Agent功能测试\n")
    
    tests = [
        ("配置文件测试", test_config_file),
        ("Agent模块导入测试", test_agent_import),
        ("Agent初始化测试", test_agent_basic),
        ("CLI集成测试", test_cli_integration),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"🧪 {name}")
        print('='*40)
        
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} 出现异常: {e}")
            results.append((name, False))
    
    # 总结
    print(f"\n{'='*40}")
    print("📊 测试结果总结")
    print('='*40)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
        if success:
            passed += 1
    
    print(f"\n🎯 总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有基础测试通过！Agent已成功集成。")
        print("\n📝 后续使用步骤:")
        print("1. 启动Web界面: streamlit run app.py")
        print("2. 在侧边栏启用 'Enable Smart Agent'")
        print("3. 上传文件进行检测")
        print("4. 在 'Agent Analysis' 标签页查看AI分析结果")
    else:
        print("⚠️ 部分测试失败，请检查配置。")


if __name__ == "__main__":
    main()