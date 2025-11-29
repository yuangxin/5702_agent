#!/usr/bin/env python3
"""
Agent功能测试脚本：验证智能抄袭分析功能是否正常工作。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from plagiarism_checker.pipeline import PipelineConfig, PlagiarismPipeline


def test_agent_basic():
    """测试基本Agent功能"""
    print("🧪 测试Agent基本功能...")
    
    # 检查API配置文件
    api_config = Path("api_config.json")
    if not api_config.exists():
        print("❌ api_config.json 文件不存在！")
        return False
    
    try:
        # 尝试初始化Agent
        from plagiarism_checker.agent import SmartPlagiarismAgent
        agent = SmartPlagiarismAgent("api_config.json")
        print("✅ Agent初始化成功")
        return True
    except Exception as e:
        print(f"❌ Agent初始化失败: {e}")
        return False


def test_agent_pipeline():
    """测试完整的Agent流水线"""
    print("\n🔄 测试Agent检测流程...")
    
    # 配置参数
    config = PipelineConfig(
        submissions_dir=Path("dataset"),
        device="cpu",
        use_parallel=False,
        similarity_threshold=0.80,
        enable_agent=True,
        agent_threshold=0.60,  # 降低阈值以确保触发
        api_config_path="api_config.json",
        agent_dual_phase=False,
        agent_max_reports=2,
        output_dir=Path("test_output"),
    )
    
    try:
        pipeline = PlagiarismPipeline(config)
        
        # 运行带Agent的检测
        print("📊 开始检测...")
        sent_stats, sent_details, agent_reports = pipeline.run_with_agent()
        
        print(f"✅ 检测完成:")
        print(f"   - 发现 {len(sent_stats)} 个可疑文本对")
        print(f"   - 生成 {len(agent_reports)} 个Agent分析报告")
        
        # 显示Agent报告摘要
        if agent_reports:
            print("\n📝 Agent分析摘要:")
            for i, report in enumerate(agent_reports, 1):
                pair = report['pair']
                report_text = report['report']
                # 提取判定结果
                if "检测到抄袭嫌疑" in report_text:
                    result = "⚠️ 检测到抄袭嫌疑"
                else:
                    result = "✅ 未检测到明显抄袭"
                print(f"   {i}. {pair[0]} ⟷ {pair[1]}: {result}")
        else:
            print("💡 未生成Agent报告（可能风险分数都低于阈值）")
        
        return True
        
    except Exception as e:
        print(f"❌ 流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_single_pair():
    """测试单个文本对的Agent分析"""
    print("\n🎯 测试单个文本对分析...")
    
    try:
        from plagiarism_checker.agent import SmartPlagiarismAgent
        
        agent = SmartPlagiarismAgent("api_config.json", dual_phase=False)
        
        # 模拟数据
        text_a = """Artificial intelligence, often abbreviated as AI, is a crucial field within computer science that focuses on understanding and replicating aspects of human cognition. This discipline involves the study and creation of algorithms, models, and systems that can perceive, reason, learn, and make decisions. Over the years, AI has evolved to include subfields such as machine learning, natural language processing, 
        computer vision, and robotics. Researchers in this area aim to design technologies that can not only mimic human intelligence but also augment it, enhancing efficiency and accuracy across various domains. From predictive analytics to autonomous systems, AI plays an increasingly significant role in transforming industries, improving problem-solving capabilities, and enabling innovative solutions to complex challenges."""
        text_b = """Artificial intelligence, a rapidly advancing domain of computer science, is devoted to the study, development, and implementation of techniques that simulate and enhance human intellectual abilities. This includes designing computational frameworks and intelligent systems capable of learning, reasoning, and adapting to diverse scenarios. AI encompasses numerous subfields, including machine learning, natural language understanding, robotics, and computer vision, each contributing to the creation of smarter technologies. The ultimate goal of AI research is to produce tools that can assist or complement human decision-making, boost productivity, and solve intricate problems in areas ranging from healthcare 
        and finance to transportation and scientific discovery. By leveraging AI, organizations are increasingly able to make data-driven decisions and develop innovative solutions that were previously unattainable."""
        
        similarity_hits = [
            {
                'text_i': text_a,
                'text_j': text_b,
                'sim': 0.92,
                'sent_id_i': 1,
                'sent_id_j': 1,
                'citation_penalty': 1.0
            }
        ]
        
        statistics = {
            'count': 1,
            'mean_sim': 0.92,
            'max_sim': 0.92,
            'coverage_min': 0.8
        }
        
        print("🤔 开始AI分析...")
        analysis = agent.analyze_suspicious_pair(
            text_a=text_a,
            text_b=text_b,
            similarity_hits=similarity_hits,
            statistics=statistics,
            left_name="文档A",
            right_name="文档B",
            dual_phase=False
        )
        
        print("✅ 分析完成:")
        print(f"   - 是否抄袭: {analysis.is_plagiarism}")
        print(f"   - 置信度: {analysis.confidence:.1%}")
        print(f"   - 推理过程: {analysis.reasoning[:100]}...")
        print(f"   - 关键证据: {analysis.key_evidence} ")
        
        return True
        
    except Exception as e:
        print(f"❌ 单对分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 启动Agent功能测试\n")
    
    # 创建输出目录
    Path("test_output").mkdir(exist_ok=True)
    
    # 运行测试
    tests = [
        ("基本功能测试", test_agent_basic),
        ("单对分析测试", test_agent_single_pair)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🔍 {name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} 出现异常: {e}")
            results.append((name, False))
    
    # 总结
    print(f"\n{'='*50}")
    print("📊 测试结果总结")
    print('='*50)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
        if success:
            passed += 1
    
    print(f"\n🎯 总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！Agent功能正常工作。")
    else:
        print("⚠️ 部分测试失败，请检查配置和网络连接。")


if __name__ == "__main__":
    main()