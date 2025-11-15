#!/usr/bin/env python3
"""
Generate compliance evidence report from benchmark results.
"""

import sys
import argparse
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from compliance import ComplianceMapper, ReportBuilder


def find_latest_metrics_file(metrics_dir: str = 'metrics') -> Path:
    """Find the most recent metrics file."""
    metrics_path = Path(metrics_dir)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")
    
    jsonl_files = list(metrics_path.glob('results_*.jsonl'))
    if not jsonl_files:
        raise FileNotFoundError(f"No metrics files found in {metrics_dir}")
    
    # Sort by modification time, most recent first
    latest = sorted(jsonl_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return latest


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate EU AI Act compliance evidence report'
    )
    parser.add_argument(
        '--metrics',
        help='Path to metrics JSONL file (default: latest in metrics/)'
    )
    parser.add_argument(
        '--output-dir',
        default='reports',
        help='Output directory for report (default: reports/)'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'json'],
        default='markdown',
        help='Report format (default: markdown)'
    )
    parser.add_argument(
        '--matrix',
        default='compliance/eu_ai_act_matrix.csv',
        help='Path to compliance matrix CSV'
    )
    
    args = parser.parse_args()
    
    # Find metrics file
    if args.metrics:
        metrics_file = Path(args.metrics)
    else:
        try:
            metrics_file = find_latest_metrics_file()
            print(f"Using latest metrics file: {metrics_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease run benchmarks first using: python run_benchmarks.py")
            sys.exit(1)
    
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        sys.exit(1)
    
    # Check matrix file
    if not Path(args.matrix).exists():
        print(f"Error: Compliance matrix not found: {args.matrix}")
        sys.exit(1)
    
    # Initialize mapper and builder
    print(f"\nLoading compliance matrix from {args.matrix}...")
    mapper = ComplianceMapper(args.matrix)
    
    print(f"Loading metrics from {metrics_file}...")
    builder = ReportBuilder(mapper)
    
    # Generate report
    print(f"Generating {args.format} report...")
    report_file = builder.generate_report(
        str(metrics_file),
        output_dir=args.output_dir,
        format=args.format
    )
    
    print(f"\n✓ Report generated successfully!")
    print(f"  Location: {report_file}")
    print(f"  Format: {args.format}")
    
    # Print quick summary
    metrics = builder.load_metrics(str(metrics_file))
    coverage = mapper.get_coverage_summary()
    
    print(f"\nQuick Summary:")
    print(f"  - Requirements covered: {coverage['covered']}/{coverage['total_requirements']}")
    print(f"  - Overall coverage: {coverage['partial_coverage_percentage']:.1f}%")
    print(f"  - Tests conducted: {len(metrics)}")
    print(f"  - Models evaluated: {len(set(m['model_name'] for m in metrics))}")


if __name__ == '__main__':
    main()

