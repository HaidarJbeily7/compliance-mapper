# Compliance Mapper

A tool for benchmarking AI model trustworthiness and mapping results to legal compliance frameworks, starting with the EU AI Act.

## Overview

This project provides:

1. **Trust Framework Benchmarking** - Adapters for TrustEval and DecodingTrust frameworks
2. **Legal Compliance Mapping** - Maps benchmark metrics to EU AI Act requirements
3. **Evidence Report Generation** - Automated compliance evidence reports

## Quick Start

### 1. Installation

```bash
# Clone the repository with submodules
git clone --recurse-submodules <repository-url>
cd compliance-mapper

# Or if already cloned, initialize submodules
git submodule init
git submodule update

# Install the package in development mode
pip install -e .

# Set up API keys
cp .env.example .env
# Edit .env with your OpenAI and Anthropic API keys
```

### 2. Run Benchmarks

```bash
# Run all benchmarks on both models
python src/run_benchmarks.py

# Run on specific model
python src/run_benchmarks.py --model gpt-4o-mini

# Run specific framework
python src/run_benchmarks.py --framework trust_eval

# Or use the installed command (after pip install -e .)
run-benchmarks
```

### 3. Generate Compliance Report

```bash
# Generate report from latest benchmark results
python src/generate_report.py

# Specify metrics file
python src/generate_report.py --metrics metrics/results_20241115_120000.jsonl

# Generate JSON format
python src/generate_report.py --format json

# Or use the installed command
generate-report
```

## Project Structure

```tree
compliance-mapper/
├── src/                     # Source code
│   ├── benchmarks/          # Benchmark framework adapters
│   │   ├── trust_eval_runner.py
│   │   ├── decoding_trust_runner.py
│   │   ├── model_client.py
│   │   └── base.py
│   ├── compliance/          # Legal framework mappings
│   │   ├── eu_ai_act_matrix.csv
│   │   ├── mapper.py
│   │   └── report_builder.py
│   ├── run_benchmarks.py    # Main benchmark script
│   └── generate_report.py   # Report generation script
├── external/                # External repositories (git submodules)
│   └── TrustEval-toolkit/   # TrustEval framework (submodule)
├── metrics/                 # Benchmark results (JSONL)
├── reports/                 # Generated compliance reports
├── config.yaml              # Configuration
├── setup.py                 # Package setup
├── pyproject.toml           # Modern Python project config
├── requirements.txt         # Python dependencies
└── SUBMODULE_SETUP.md       # Git submodule instructions
```

## Benchmarking Coverage

### Trust Frameworks

| Framework | Tests | Focus Areas |
|-----------|-------|-------------|
| **TrustEval** | 5 | Safety, Fairness, Robustness, Privacy, Truthfulness |
| **DecodingTrust** | 7 | Toxicity, Bias, Robustness, OOD, Privacy, Ethics, Fairness |

### EU AI Act Coverage

- **16 Requirements Mapped** across Articles 9, 10, 11, 13-15, 52, 61
- **Coverage Status**: 37.5% fully covered, 43.75% partially covered, 18.75% gaps
- **Key Areas**: Data governance, risk management, transparency, robustness, privacy

## Configuration

Edit `config.yaml` to customize:

```yaml
models:
  gpt-4o-mini:
    temperature: 0.7
    max_tokens: 4096
    rate_limit: 50

benchmarks:
  trust_eval:
    enabled: true
    tests: [toxicity, stereotype_bias, ...]
```

## Requirements

- Python 3.10+
- OpenAI API key
- Anthropic API key

See `requirements.txt` for full dependency list.

## Contributing

This is a proof-of-concept implementation. Contributions welcome!

## License

MIT License

## References

- **TrustEval-toolkit**: <https://github.com/TrustGen/TrustEval-toolkit>
- **DecodingTrust**: <https://github.com/AI-secure/DecodingTrust>
- **EU AI Act**: <https://artificialintelligenceact.eu/>

---
