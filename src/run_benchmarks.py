#!/usr/bin/env python3
"""
Main script to run trust benchmarks on multiple LLMs.

Supports:
- OpenAI: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- Anthropic: Claude Opus, Claude Sonnet, Claude Haiku
- Google: Gemini Pro, Gemini Flash

Evaluation frameworks:
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
from typing import List, Dict, Optional
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


def get_available_models(config: dict) -> Dict[str, dict]:
    """Get all available models from config."""
    return config.get('models', {})


def get_api_key_for_provider(provider: str) -> Optional[str]:
    """Get API key for a specific provider."""
    key_mapping = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
    }
    env_var = key_mapping.get(provider)
    if env_var:
        return os.getenv(env_var)
    return None


def save_results(results: List[BenchmarkResult], output_dir: str = 'metrics', suffix: str = ''):
    """Save benchmark results to JSONL file."""
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f'results_{timestamp}{suffix}.jsonl' if suffix else f'results_{timestamp}.jsonl'
    output_file = output_path / filename
    
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
    
    # Print comparison table if multiple models
    if len(by_model) > 1:
        print("\nMODEL COMPARISON")
        print("-" * 80)
        print(f"{'Model':<20} {'Tests Passed':<15} {'Avg Score':<12} {'Pass Rate':<12}")
        print("-" * 60)
        for model_name, model_results in by_model.items():
            total = len(model_results)
            passed = sum(1 for r in model_results if r.passed)
            avg_score = sum(r.score for r in model_results) / total if total > 0 else 0
            pass_rate = passed / total if total > 0 else 0
            print(f"{model_name:<20} {passed}/{total:<13} {avg_score:<12.2f} {pass_rate*100:<11.1f}%")
        print("="*80)


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
    # Load environment variables first
    load_dotenv()
    
    # Load configuration
    config = load_config()
    available_models = get_available_models(config)
    model_names = list(available_models.keys())
    
    parser = argparse.ArgumentParser(
        description='Run trust benchmarks on LLMs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  OpenAI:    gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
  Anthropic: claude-opus, claude-sonnet, claude-haiku
  Google:    gemini-pro, gemini-flash

Examples:
  python run_benchmarks.py --model gpt-4o-mini
  python run_benchmarks.py --model claude-sonnet --framework trust_eval
  python run_benchmarks.py --models gpt-4o-mini claude-haiku
  python run_benchmarks.py --all-models
        """
    )
    parser.add_argument(
        '--model',
        choices=model_names + ['all'],
        help='Single model to benchmark'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=model_names,
        help='Multiple models to benchmark'
    )
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Run benchmarks on all available models'
    )
    parser.add_argument(
        '--provider',
        choices=['openai', 'anthropic', 'google'],
        help='Run all models from a specific provider'
    )
    parser.add_argument(
        '--framework',
        choices=['trust_eval', 'decoding_trust', 'all'],
        default='all',
        help='Framework to use (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        default='metrics',
        help='Output directory for results (default: metrics)'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("\nAvailable Models:")
        print("-" * 60)
        for name, model_config in available_models.items():
            provider = model_config.get('provider', 'unknown')
            model_id = model_config.get('model_id', name)
            api_key = get_api_key_for_provider(provider)
            status = "✓" if api_key else "✗ (missing API key)"
            print(f"  {name:<20} {provider:<12} {model_id:<30} {status}")
        print()
        return 0
    
    # Determine which models to run
    models_to_run = []
    
    if args.all_models or args.model == 'all':
        models_to_run = model_names
    elif args.provider:
        models_to_run = [
            name for name, cfg in available_models.items()
            if cfg.get('provider') == args.provider
        ]
    elif args.models:
        models_to_run = args.models
    elif args.model:
        models_to_run = [args.model]
    else:
        # Default: run gpt-4o-mini and claude-haiku
        models_to_run = ['gpt-4o-mini', 'claude-haiku']
    
    # Filter to only models with valid API keys
    valid_models = []
    for model_name in models_to_run:
        model_config = available_models.get(model_name, {})
        provider = model_config.get('provider')
        api_key = get_api_key_for_provider(provider)
        
        if api_key:
            valid_models.append(model_name)
        else:
            print(f"⚠️  Skipping {model_name}: No API key for {provider}")
    
    if not valid_models:
        print("\nError: No models have valid API keys configured.")
        print("Please set the required environment variables:")
        print("  - OPENAI_API_KEY for OpenAI models")
        print("  - ANTHROPIC_API_KEY for Anthropic models")
        print("  - GOOGLE_API_KEY for Google models")
        sys.exit(1)
    
    print(f"\n🚀 Running benchmarks for {len(valid_models)} model(s): {', '.join(valid_models)}")
    
    # Determine which frameworks to run
    frameworks = ['all'] if args.framework == 'all' else [args.framework]
    
    # Run benchmarks for each model
    all_results = []
    
    for model_name in valid_models:
        model_config = available_models.get(model_name, {})
        provider = model_config.get('provider')
        api_key = get_api_key_for_provider(provider)
        
        model_results = await run_benchmarks_for_model(
            model_name,
            api_key,
            config,
            frameworks
        )
        all_results.extend(model_results)
    
    if not all_results:
        print("No benchmark results collected. Check your configuration and API keys.")
        sys.exit(1)
    
    # Save results
    output_file = save_results(all_results, args.output_dir)
    
    # Print summary
    print_summary(all_results)
    
    print(f"\n✓ Benchmarking complete! Results saved to {output_file}")
    print(f"\n📊 Generate compliance report with:")
    print(f"   python src/generate_report.py --metrics {output_file}")
    
    # Return exit code based on pass rate
    pass_rate = sum(1 for r in all_results if r.passed) / len(all_results)
    return 0 if pass_rate >= 0.5 else 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
