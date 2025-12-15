"""
通信分析命令模块
"""

import time
from pathlib import Path

from ..validators import parse_show_options
from ...analyzer import analyze_communication_performance


class CommCommand:
    """通信分析命令处理器"""

    def run(self, args) -> int:
        """运行通信性能分析"""
        print(f"=== 通信性能分析 ===")
        print(f"Pod目录: {args.pod_dir}")
        print(f"Step: {args.step if args.step is not None else '所有step'}")
        print(f"通信Kernel前缀: {args.kernel_prefix}")
        print(f"输出目录: {args.output_dir}")
        print()
        
        # 验证pod目录
        pod_path = Path(args.pod_dir)
        if not pod_path.exists():
            print(f"错误: Pod目录不存在: {args.pod_dir}")
            return 1
        
        if not pod_path.is_dir():
            print(f"错误: 路径不是目录: {args.pod_dir}")
            return 1
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            start_time = time.time()
            
            # 运行通信性能分析
            generated_files = analyze_communication_performance(
                pod_dir=str(pod_path),
                step=args.step,
                kernel_prefix=args.kernel_prefix,
                output_dir=str(output_dir),
            )
            
            total_time = time.time() - start_time
            print(f"\n分析完成，总耗时: {total_time:.2f} 秒")
            
            # 显示生成的文件
            print("\n生成的文件:")
            for file_path in generated_files:
                print(f"  {file_path}")
            
            return 0
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            return 1
