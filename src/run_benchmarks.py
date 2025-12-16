#!/usr/bin/env python3
"""
Main script to run trust benchmarks on GPT-4o mini and Claude Haiku.

Supports two evaluation frameworks:
- TrustEval: Safety, Fairness, Robustness, Privacy, Truthfulness
- DecodingTrust: Toxicity, Stereotype Bias, Adversarial Robustness, 
                 OOD Robustness, Privacy, Machine Ethics, Fairness
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List
import yaml
from dotenv import load_dotenv

# Add src to path if running as script
if __name__ == "__main__":
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from benchmarks import TrustEvalRunner, DecodingTrustRunner, BenchmarkResult


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(results: List[BenchmarkResult], output_dir: str = 'metrics'):
    """Save benchmark results to JSONL file."""
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_file = output_path / f'results_{timestamp}.jsonl'
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + '\n')
    
    print(f"\n✓ Results saved to {output_file}")
    return output_file


def print_summary(results: List[BenchmarkResult]):
    """Print summary of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Group by model
    by_model = {}
    for result in results:
        if result.model_name not in by_model:
            by_model[result.model_name] = []
        by_model[result.model_name].append(result)
    
    for model_name, model_results in by_model.items():
        print(f"\n{model_name.upper()}")
        print("-" * 80)
        
        # Group by framework
        by_framework = {}
        for result in model_results:
            if result.framework not in by_framework:
                by_framework[result.framework] = []
            by_framework[result.framework].append(result)
        
        for framework, fw_results in by_framework.items():
            print(f"\n  {framework}:")
            for result in fw_results:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(f"    {result.test_name:30s} Score: {result.score:.2f}  {status}")
            
            # Framework subtotals
            fw_passed = sum(1 for r in fw_results if r.passed)
            fw_avg = sum(r.score for r in fw_results) / len(fw_results) if fw_results else 0
            print(f"    {'─' * 50}")
            print(f"    Subtotal: {fw_passed}/{len(fw_results)} passed | Avg: {fw_avg:.2f}")
        
        # Overall stats
        total = len(model_results)
        passed = sum(1 for r in model_results if r.passed)
        avg_score = sum(r.score for r in model_results) / total if total > 0 else 0
        
        print(f"\n  {'═' * 50}")
        print(f"  OVERALL: {passed}/{total} tests passed | Average score: {avg_score:.2f}")
    
    print("\n" + "="*80)


async def run_benchmarks_for_model(
    model_name: str,
    api_key: str,
    config: dict,
    frameworks: List[str]
) -> List[BenchmarkResult]:
    """Run all benchmarks for a single model."""
    results = []
    
    print(f"\n{'='*80}")
    print(f"Running benchmarks for {model_name}")
    print(f"{'='*80}")
    
    if 'trust_eval' in frameworks or 'all' in frameworks:
        print(f"\n→ Running TrustEval tests...")
        try:
            runner = TrustEvalRunner(model_name, api_key, config)
            trust_results = await runner.run_all_tests()
            results.extend(trust_results)
            print(f"  Completed {len(trust_results)} TrustEval tests")
        except Exception as e:
            print(f"  Error running TrustEval: {e}")
    
    if 'decoding_trust' in frameworks or 'all' in frameworks:
        print(f"\n→ Running DecodingTrust tests...")
        try:
            runner = DecodingTrustRunner(model_name, api_key, config)
            decoding_results = await runner.run_all_tests()
            results.extend(decoding_results)
            print(f"  Completed {len(decoding_results)} DecodingTrust tests")
        except Exception as e:
            print(f"  Error running DecodingTrust: {e}")
    
    return results


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run trust benchmarks on LLMs')
    parser.add_argument(
        '--model',
        choices=['gpt-4o-mini', 'claude-haiku', 'both'],
        default='both',
        help='Model to benchmark'
    )
    parser.add_argument(
        '--framework',
        choices=['trust_eval', 'decoding_trust', 'all'],
        default='all',
        help='Framework to use'
    )
    parser.add_argument(
        '--output-dir',
        default='metrics',
        help='Output directory for results'
    )
    parser.add_argument(
        '--test',
        help='Run a specific test (e.g., safety, toxicity, fairness)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config()
    
    # Get API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not openai_key and args.model in ['gpt-4o-mini', 'both']:
        print("Error: OPENAI_API_KEY not found in environment")
        sys.exit(1)
    
    if not anthropic_key and args.model in ['claude-haiku', 'both']:
        print("Error: ANTHROPIC_API_KEY not found in environment")
        sys.exit(1)
    
    # Determine which frameworks to run
    frameworks = ['all'] if args.framework == 'all' else [args.framework]
    
    # Run benchmarks
    all_results = []
    
    if args.model in ['gpt-4o-mini', 'both']:
        gpt_results = await run_benchmarks_for_model(
            'gpt-4o-mini',
            openai_key,
            config,
            frameworks
        )
        all_results.extend(gpt_results)
    
    if args.model in ['claude-haiku', 'both']:
        claude_results = await run_benchmarks_for_model(
            'claude-haiku',
            anthropic_key,
            config,
            frameworks
        )
        all_results.extend(claude_results)
    
    if not all_results:
        print("No benchmark results collected. Check your configuration and API keys.")
        sys.exit(1)
    
    # Save results
    output_file = save_results(all_results, args.output_dir)
    
    # Print summary
    print_summary(all_results)
    
    print(f"\n✓ Benchmarking complete! Results saved to {output_file}")
    
    # Return exit code based on pass rate
    pass_rate = sum(1 for r in all_results if r.passed) / len(all_results)
    return 0 if pass_rate >= 0.5 else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
