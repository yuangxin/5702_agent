#!/usr/bin/env python3
"""
Agent问题诊断脚本：分析为什么没有生成Agent报告
"""

import sys
import json
from pathlib import Path

def diagnose_agent_issue():
    """诊断Agent问题"""
    print("🔍 诊断Agent无报告问题...\n")
    
    issues = []
    
    # 检查1: API配置文件
    print("1️⃣ 检查API配置文件...")
    config_file = Path("api_config.json")
    if not config_file.exists():
        issues.append("❌ api_config.json 文件不存在")
        print("   ❌ api_config.json 文件不存在")
    else:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print("   ✅ api_config.json 文件存在且格式正确")
            
            # 检查配置内容
            if 'modelscope' in config:
                ms_config = config['modelscope']
                if not ms_config.get('api_key'):
                    issues.append("❌ ModelScope API key为空")
                    print("   ❌ API key为空")
                else:
                    print(f"   ✅ API key已配置: {ms_config['api_key'][:10]}...")
                print(f"   ✅ Base URL: {ms_config.get('base_url', 'N/A')}")
                print(f"   ✅ Model: {ms_config.get('model', 'N/A')}")
            else:
                issues.append("❌ api_config.json 缺少有效配置")
                print("   ❌ 缺少ModelScope配置")
                
        except Exception as e:
            issues.append(f"❌ 配置文件解析错误: {e}")
            print(f"   ❌ 配置文件解析错误: {e}")
    
    # 检查2: Agent模块
    print("\n2️⃣ 检查Agent模块...")
    try:
        from plagiarism_checker.agent import SmartPlagiarismAgent, AgentAnalysis
        print("   ✅ Agent模块导入成功")
        
        # 尝试初始化
        try:
            agent = SmartPlagiarismAgent("api_config.json", dual_phase=False)
            print(f"   ✅ Agent初始化成功 (Provider: {agent.provider})")
        except Exception as e:
            issues.append(f"❌ Agent初始化失败: {e}")
            print(f"   ❌ Agent初始化失败: {e}")
            
    except Exception as e:
        issues.append(f"❌ Agent模块导入失败: {e}")
        print(f"   ❌ Agent模块导入失败: {e}")
    
    # 检查3: 数据集
    print("\n3️⃣ 检查数据集...")
    dataset_dir = Path("dataset")
    if dataset_dir.exists():
        files = list(dataset_dir.glob("*.txt"))
        print(f"   ✅ 数据集目录存在，包含 {len(files)} 个txt文件")
        if len(files) < 2:
            issues.append("❌ 数据集文件数量不足（需要至少2个文件）")
            print("   ❌ 文件数量不足，需要至少2个文件进行比较")
    else:
        issues.append("❌ dataset目录不存在")
        print("   ❌ dataset目录不存在")
    
    # 检查4: 模拟配置参数
    print("\n4️⃣ 检查配置参数...")
    from plagiarism_checker.pipeline import PipelineConfig
    
    try:
        config = PipelineConfig(
            submissions_dir=dataset_dir,
            enable_agent=True,
            agent_threshold=0.7,
            api_config_path="api_config.json",
            agent_max_reports=3,
            agent_dual_phase=False
        )
        print("   ✅ PipelineConfig创建成功")
        print(f"   ✅ enable_agent: {config.enable_agent}")
        print(f"   ✅ agent_threshold: {config.agent_threshold}")
        print(f"   ✅ agent_max_reports: {config.agent_max_reports}")
        
        if config.agent_max_reports == 0:
            issues.append("❌ agent_max_reports设置为0，将不生成报告")
            
    except Exception as e:
        issues.append(f"❌ PipelineConfig创建失败: {e}")
        print(f"   ❌ PipelineConfig创建失败: {e}")
    
    # 总结
    print(f"\n{'='*50}")
    print("📊 诊断结果总结")
    print('='*50)
    
    if not issues:
        print("🎉 恭喜！没有发现明显问题，Agent应该能正常工作。")
        print("\n💡 可能的原因：")
        print("1. 检测到的文本对风险分数都低于阈值（默认0.7）")
        print("2. 网络连接问题导致API调用失败")
        print("3. API服务暂时不可用")
        print("\n🔧 建议解决方案：")
        print("1. 在Web界面中降低Agent Analysis Threshold到0.5")
        print("2. 检查网络连接和API服务状态")
        print("3. 查看控制台是否有错误信息")
    else:
        print("⚠️ 发现以下问题：")
        for issue in issues:
            print(f"   {issue}")
        
        print("\n🔧 修复建议：")
        if "api_config.json" in str(issues):
            print("• 确保api_config.json文件存在且格式正确")
            print("• 验证API key是否有效")
        if "Agent模块" in str(issues):
            print("• 检查依赖包是否正确安装")
            print("• 确保所有Python文件无语法错误")
        if "数据集" in str(issues):
            print("• 确保dataset目录存在并包含足够的文件")
        if "agent_max_reports设置为0" in str(issues):
            print("• 这是主要问题！agent_max_reports被错误设置为0")
            print("• 已在代码中修复，重新运行应该可以解决")


if __name__ == "__main__":
    diagnose_agent_issue()