#!/usr/bin/env python3
"""
VideoMultiAgents - Gemini Implementation Verification Script
验证所有Gemini迁移组件是否正确安装和配置

使用方法:
  python verify_gemini_setup.py
  
预期输出: ✅ ALL SYSTEMS GO
"""

import sys
import os


def check_environment():
    """检查环境变量"""
    print("\n[1/5] 环境检查...")
    
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        print(f"  ✅ GEMINI_API_KEY 已设置 (前10字符: {gemini_key[:10]}...)")
        return True
    else:
        print("  ⚠️  GEMINI_API_KEY 未设置")
        print("     运行: export GEMINI_API_KEY='your_key_here'")
        return False


def check_packages():
    """检查必需的包"""
    print("\n[2/5] 包依赖检查...")
    
    packages = [
        ("google-generativeai", "google.generativeai"),
        ("langchain", "langchain"),
        ("langchain-core", "langchain_core"),
        ("langgraph", "langgraph"),
        ("langchain-community", "langchain_community"),
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - 未安装")
            all_installed = False
    
    if not all_installed:
        print("\n  安装缺失包:")
        print("    pip install google-generativeai langchain langchain-core langgraph langchain-community")
    
    return all_installed


def check_modules():
    """检查项目模块"""
    print("\n[3/5] 项目模块检查...")
    
    modules_to_test = [
        ("langchain_gemini_wrapper", "ChatGemini"),
        ("langchain_gemini_agent", "create_gemini_tools_agent"),
        ("util", "ask_gemini_omni"),
    ]
    
    all_imported = True
    for module_name, symbol in modules_to_test:
        try:
            exec(f"from {module_name} import {symbol}")
            print(f"  ✅ {module_name}.{symbol}")
        except Exception as e:
            print(f"  ❌ {module_name} - {str(e)[:50]}")
            all_imported = False
    
    return all_imported


def check_agents():
    """检查多智能体框架"""
    print("\n[4/5] 多智能体框架检查...")
    
    # 不设置GEMINI_API_KEY时会导入失败,这是预期行为
    # 这里只检查模块存在性
    agents = [
        "multi_agent_report",
        "multi_agent_debate",
        "multi_agent_report_star",
        "multi_agent_star",
    ]
    
    all_exist = True
    for agent_name in agents:
        try:
            # 临时设置API KEY以便导入
            old_key = os.environ.get("GEMINI_API_KEY")
            os.environ["GEMINI_API_KEY"] = "temp_for_check"
            
            exec(f"import {agent_name}")
            
            # 恢复原状
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            
            print(f"  ✅ {agent_name}")
        except Exception as e:
            # 恢复原状
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
            
            print(f"  ❌ {agent_name} - {str(e)[:50]}")
            all_exist = False
    
    return all_exist


def check_api_connectivity():
    """检查API连接(仅当API KEY设置时)"""
    print("\n[5/5] API连接检查...")
    
    if not os.getenv("GEMINI_API_KEY"):
        print("  ⚠️  跳过 - GEMINI_API_KEY 未设置")
        print("     运行: export GEMINI_API_KEY='your_key'")
        print("     然后: python verify_gemini_setup.py")
        return True
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # 简单的API调用测试
        response = model.generate_content("Respond with 'OK' only.")
        
        if "OK" in response.text or response.text.strip():
            print(f"  ✅ Gemini API 连接成功")
            print(f"     响应: {response.text.strip()[:50]}")
            return True
        else:
            print(f"  ❌ Gemini API 响应异常")
            return False
    except Exception as e:
        print(f"  ❌ Gemini API 连接失败: {str(e)[:50]}")
        return False


def main():
    print("=" * 70)
    print("VideoMultiAgents - Gemini Implementation Verification")
    print("=" * 70)
    
    results = [
        ("环境配置", check_environment()),
        ("包依赖", check_packages()),
        ("项目模块", check_modules()),
        ("多智能体框架", check_agents()),
        ("API连接", check_api_connectivity()),
    ]
    
    print("\n" + "=" * 70)
    print("验证结果总结:")
    print("=" * 70)
    
    all_passed = all(result for _, result in results)
    
    for check_name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL SYSTEMS GO - 已准备就绪")
        print("\n使用方法:")
        print("  export GEMINI_API_KEY='your_gemini_api_key_here'")
        print("  python main.py --dataset=demo --modality=all")
        print("=" * 70)
        return 0
    else:
        print("❌ 某些检查失败 - 请查看上述错误并修复")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
