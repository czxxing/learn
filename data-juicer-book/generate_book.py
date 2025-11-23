#!/usr/bin/env python3
"""
Data-Juicer技术指南电子书生成脚本
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
        self.chapters = [
            "00-目录与索引.md",
            "01-整体架构.md", 
            "02-数据处理核心机制.md",
            "03-算子系统详解.md",
            "04-数据集管理.md",
            "05-模型调用机制.md",
            "06-分析功能详解.md",
            "07-总结与展望.md"
        ]
        
    def read_chapter(self, filename):
        """读取章节内容"""
        filepath = os.path.join(self.book_dir, filename)
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
    
    def generate_pdf(self, output_path="data-juicer技术指南.pdf"):
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
            <title>Data-Juicer技术指南</title>
            {css}
        </head>
        <body>
            <h1>Data-Juicer技术指南</h1>
            {html_content}
        </body>
        </html>
        """
        
        # 生成PDF
        HTML(string=full_html).write_pdf(output_path)
        print(f"PDF版本已生成: {output_path}")
    
    def generate_epub(self, output_path="data-juicer技术指南.epub"):
        """生成EPUB版本"""
        print("正在生成EPUB版本...")
        
        # 创建EPUB书籍
        book = epub.EpubBook()
        
        # 设置元数据
        book.set_identifier('data-juicer-guide')
        book.set_title('Data-Juicer技术指南')
        book.set_language('zh')
        book.add_author('Data-Juicer社区')
        
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
        generator.generate_pdf("data-juicer技术指南.pdf")
    except Exception as e:
        print(f"PDF生成失败: {e}")
        print("尝试使用简化版本...")
        
    # 生成EPUB版本
    try:
        generator.generate_epub("data-juicer技术指南.epub")
    except Exception as e:
        print(f"EPUB生成失败: {e}")
        
    print("\n生成完成！")

if __name__ == "__main__":
    main()