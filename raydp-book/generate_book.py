#!/usr/bin/env python3
"""
RayDP源码分析电子书生成脚本
生成PDF和EPUB格式的发布版本
"""

import os
import re
import markdown
from ebooklib import epub
from weasyprint import HTML
import tempfile

class BookGenerator:
    def __init__(self, book_dir):
        self.book_dir = book_dir
        self.source_dir = os.path.join(book_dir, "../raydp/source_code_analysis")
        self.chapters = [
            # 核心架构分析
            "00_OVERVIEW_ARCHITECTURE.md",
            "01_PYTHON_CONTEXT_MODULE.md",
            "02_PYTHON_SPARK_CLUSTER_MODULE.md",
            "03_PYTHON_RAY_SPARK_MASTER_MODULE.md",
            "04_JAVA_SPARK_ON_RAY_CONFIGS_MODULE.md",
            "05_SPARK_ON_RAY_CREATION_FLOW.md",
            "06_CORE_PROJECT_STRUCTURE_AND_CLASS_RELATIONSHIP.md",
            
            # Spark-on-Ray 运行原理深度分析 - 基础原理系列
            "spark_on_ray_principles/01_ARCHITECTURE_PRINCIPLES.md",
            "spark_on_ray_principles/02_RESOURCE_MANAGEMENT_PRINCIPLES.md",
            "spark_on_ray_principles/03_COMMUNICATION_MECHANISM_PRINCIPLES.md",
            "spark_on_ray_principles/04_DATA_EXCHANGE_MECHANISM_PRINCIPLES.md",
            "spark_on_ray_principles/05_FAULT_TOLERANCE_PERFORMANCE_OPTIMIZATION.md",
            
            # Spark-on-Ray 运行原理深度分析 - Spark Master 创建过程深度分析
            "spark_on_ray_principles/06_SPARK_MASTER_CREATION_OVERVIEW.md",
            "spark_on_ray_principles/07_PYTHON_API_LAYER.md",
            "spark_on_ray_principles/08_PY4J_GATEWAY_COMMUNICATION.md",
            "spark_on_ray_principles/09_JAVA_PROCESS_LAUNCH.md",
            "spark_on_ray_principles/10_APP_MASTER_BRIDGE_INTERFACE.md",
            "spark_on_ray_principles/11_RAY_APP_MASTER_CORE_IMPLEMENTATION.md",
            "spark_on_ray_principles/12_SPARK_SESSION_INTEGRATION_AND_PYSPARK_INTERACTION.md",
            "spark_on_ray_principles/13_SPARK_MASTER_CREATION_SUMMARY.md",
            "spark_on_ray_principles/14_RAYDP_EXECUTOR_CREATION_AND_INVOCATION.md",
            
            # Spark-on-Ray 运行原理深度分析 - RayDP Executor 系列
            "spark_on_ray_principles/15_00_RAYDP_EXECUTOR_CREATION_AND_EXECUTION_OVERVIEW.md",
            "spark_on_ray_principles/15_01_RAYAPPMASTER_AND_EXECUTOR_MANAGEMENT.md",
            "spark_on_ray_principles/15_02_RAYDP_EXECUTOR_CREATION_AND_INITIALIZATION.md",
            "spark_on_ray_principles/15_03_EXECUTOR_REGISTRATION_AND_COMMUNICATION.md",
            "spark_on_ray_principles/15_04_EXECUTOR_STARTUP_AND_ENVIRONMENT_PREPARATION.md",
            "spark_on_ray_principles/15_05_SPARK_INTEGRATION_AND_TASK_EXECUTION.md",
            "spark_on_ray_principles/15_06_FAULT_HANDLING_AND_RECOVERY.md",
            
            # Spark Run on Ray 详细分析
            "spark_run_on_ray/01_RAYDP_ARCHITECTURE_OVERVIEW.md",
            "spark_run_on_ray/02_PYTHON_INITIALIZATION_FLOW.md",
            "spark_run_on_ray/03_JAVA_APPMASTER_INITIALIZATION.md",
            "spark_run_on_ray/04_RAY_ACTOR_CREATION_AND_MANAGEMENT.md",
            "spark_run_on_ray/05_SPARK_DRIVER_AND_EXECUTOR_COMMUNICATION.md",
            "spark_run_on_ray/06_RESOURCE_MANAGEMENT_AND_ALLOCATION.md",
            "spark_run_on_ray/07_FAULT_TOLERANCE_AND_RECOVERY.md",
            "spark_run_on_ray/08_PERFORMANCE_OPTIMIZATION.md",
            "spark_run_on_ray/RAYDP_DETAILED_RUN_FLOW_ANALYSIS.md"
        ]
        
    def read_chapter(self, filename):
        """读取章节内容"""
        filepath = os.path.join(self.source_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 移除导航链接
        content = re.sub(r'\[←.*?\]\s*\|\s*\[.*?→\].*?\n-{3,}\n', '', content)
        
        return content
    
    def merge_chapters(self):
        """合并所有章节"""
        full_content = ""
        
        for chapter in self.chapters:
            print(f"正在处理: {chapter}")
            content = self.read_chapter(chapter)
            
            # 添加章节标题
            title_match = re.match(r'^# (.+)$', content.strip().split('\n')[0])
            if title_match:
                chapter_title = title_match.group(1)
                full_content += f"\n# {chapter_title}\n\n"
            
            # 添加章节内容
            full_content += content + "\n\n"
        
        return full_content
    
    def generate_pdf(self, output_path="raydp-source-code-analysis.pdf"):
        """生成PDF版本"""
        print("正在生成PDF版本...")
        
        # 合并内容
        content = self.merge_chapters()
        
        # 转换为HTML
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        
        # 添加CSS样式
        css = """
        <style>
            body { font-family: "Microsoft YaHei", sans-serif; line-height: 1.6; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
            h2 { color: #34495e; }
            h3 { color: #7f8c8d; }
            code { background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
        """
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>RayDP源码分析</title>
            {css}
        </head>
        <body>
            <h1>RayDP源码分析</h1>
            {html_content}
        </body>
        </html>
        """
        
        # 生成PDF
        HTML(string=full_html).write_pdf(output_path)
        print(f"PDF版本已生成: {output_path}")
    
    def generate_epub(self, output_path="raydp-source-code-analysis.epub"):
        """生成EPUB版本"""
        print("正在生成EPUB版本...")
        
        # 创建EPUB书籍
        book = epub.EpubBook()
        
        # 设置元数据
        book.set_identifier('raydp-source-code-analysis')
        book.set_title('RayDP源码分析')
        book.set_language('zh')
        book.add_author('RayDP社区')
        
        # 创建章节
        chapters = []
        
        for i, chapter_file in enumerate(self.chapters):
            content = self.read_chapter(chapter_file)
            
            # 提取标题
            title_match = re.match(r'^# (.+)$', content.strip().split('\n')[0])
            if title_match:
                chapter_title = title_match.group(1)
            else:
                chapter_title = f"第{i}章"
            
            # 转换为HTML
            html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
            
            # 创建EPUB章节
            chapter = epub.EpubHtml(
                title=chapter_title,
                file_name=f'chapter_{i:02d}.xhtml',
                lang='zh'
            )
            
            chapter.content = f"""
            <h1>{chapter_title}</h1>
            {html_content}
            """
            
            book.add_item(chapter)
            chapters.append(chapter)
        
        # 定义目录结构
        book.toc = chapters
        
        # 添加导航文件
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # 定义阅读顺序
        book.spine = ['nav'] + chapters
        
        # 写入文件
        epub.write_epub(output_path, book)
        print(f"EPUB版本已生成: {output_path}")

def main():
    """主函数"""
    book_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建生成器实例
    generator = BookGenerator(book_dir)
    
    # 生成PDF版本
    try:
        generator.generate_pdf("raydp-source-code-analysis.pdf")
    except Exception as e:
        print(f"PDF生成失败: {e}")
        print("尝试使用简化版本...")
        
    # 生成EPUB版本
    try:
        generator.generate_epub("raydp-source-code-analysis.epub")
    except Exception as e:
        print(f"EPUB生成失败: {e}")
        
    print("\n生成完成！")

if __name__ == "__main__":
    main()