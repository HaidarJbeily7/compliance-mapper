"""
Report builder for generating compliance evidence reports.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from .mapper import ComplianceMapper


class ReportBuilder:
    """Builds structured compliance evidence reports."""
    
    def __init__(self, mapper: ComplianceMapper):
        """
        Initialize report builder.
        
        Args:
            mapper: ComplianceMapper instance
        """
        self.mapper = mapper
    
    def load_metrics(self, metrics_file: str) -> List[Dict[str, Any]]:
        """
        Load metrics from JSONL file.
        
        Args:
            metrics_file: Path to metrics JSONL file
            
        Returns:
            List of metric dictionaries
        """
        metrics = []
        with open(metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics
    
    def generate_report(
        self,
        metrics_file: str,
        output_dir: str = 'reports',
        format: str = 'markdown'
    ) -> Path:
        """
        Generate compliance evidence report.
        
        Args:
            metrics_file: Path to metrics JSONL file
            output_dir: Output directory for report
            format: Report format ('markdown' or 'json')
            
        Returns:
            Path to generated report
        """
        # Load metrics
        metrics = self.load_metrics(metrics_file)
        
        # Map metrics to requirements
        mapping = self.mapper.map_metric_to_requirements(metrics)
        
        # Get coverage summary
        coverage = self.mapper.get_coverage_summary()
        
        # Generate report
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if format == 'markdown':
            report_file = output_path / f'evidence_{timestamp}.md'
            self._generate_markdown_report(report_file, metrics, mapping, coverage)
        elif format == 'json':
            report_file = output_path / f'evidence_{timestamp}.json'
            self._generate_json_report(report_file, metrics, mapping, coverage)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return report_file
    
    def _generate_markdown_report(
        self,
        output_file: Path,
        metrics: List[Dict[str, Any]],
        mapping: Dict[str, Any],
        coverage: Dict[str, Any]
    ):
        """Generate markdown format report."""
        with open(output_file, 'w') as f:
            # Header
            f.write("# EU AI Act Compliance Evidence Report\n\n")
            f.write(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"This report provides evidence of compliance with the EU AI Act based on ")
            f.write(f"trust benchmarking of AI models.\n\n")
            
            # Models tested
            models = set(m['model_name'] for m in metrics)
            f.write(f"**Models Evaluated:** {', '.join(sorted(models))}\n\n")
            
            # Frameworks used
            frameworks = set(m['framework'] for m in metrics)
            f.write(f"**Frameworks Used:** {', '.join(sorted(frameworks))}\n\n")
            
            # Total tests
            f.write(f"**Total Tests Conducted:** {len(metrics)}\n\n")
            
            f.write("---\n\n")
            
            # Coverage Summary
            f.write("## Compliance Coverage Summary\n\n")
            f.write(f"- **Total Requirements:** {coverage['total_requirements']}\n")
            f.write(f"- **Fully Covered:** {coverage['covered']} ({coverage['coverage_percentage']:.1f}%)\n")
            f.write(f"- **Partially Covered:** {coverage['partial']}\n")
            f.write(f"- **Gaps Identified:** {coverage['gaps']}\n")
            f.write(f"- **Overall Coverage:** {coverage['partial_coverage_percentage']:.1f}%\n\n")
            
            # Visual indicator
            covered_pct = int(coverage['coverage_percentage'])
            partial_pct = int(coverage['partial_coverage_percentage']) - covered_pct
            gap_pct = 100 - int(coverage['partial_coverage_percentage'])
            
            f.write("**Coverage Breakdown:**\n\n")
            f.write(f"```\n")
            f.write(f"Fully Covered:    {'█' * (covered_pct // 2)}{' ' * (50 - covered_pct // 2)} {covered_pct}%\n")
            f.write(f"Partially Covered: {'▓' * (partial_pct // 2)}{' ' * (50 - partial_pct // 2)} {partial_pct}%\n")
            f.write(f"Gaps:             {'░' * (gap_pct // 2)}{' ' * (50 - gap_pct // 2)} {gap_pct}%\n")
            f.write(f"```\n\n")
            
            f.write("---\n\n")
            
            # Score Summary by Model
            f.write("## Model Performance Summary\n\n")
            for model in sorted(models):
                model_metrics = [m for m in metrics if m['model_name'] == model]
                avg_score = sum(m['score'] for m in model_metrics) / len(model_metrics)
                passed = sum(1 for m in model_metrics if m['passed'])
                
                f.write(f"### {model}\n\n")
                f.write(f"- **Average Score:** {avg_score:.2f}\n")
                f.write(f"- **Tests Passed:** {passed}/{len(model_metrics)} ({passed/len(model_metrics)*100:.1f}%)\n")
                f.write(f"- **Tests Conducted:** {len(model_metrics)}\n\n")
                
                # Breakdown by test
                f.write("**Test Results:**\n\n")
                f.write("| Test | Framework | Score | Status |\n")
                f.write("|------|-----------|-------|--------|\n")
                for m in sorted(model_metrics, key=lambda x: (x['framework'], x['test_name'])):
                    status = "✓ Pass" if m['passed'] else "✗ Fail"
                    f.write(f"| {m['test_name']} | {m['framework']} | {m['score']:.2f} | {status} |\n")
                f.write("\n")
            
            f.write("---\n\n")
            
            # Evidence by Requirement
            f.write("## Evidence by EU AI Act Requirement\n\n")
            
            # Group by status
            for status in ['COVERED', 'PARTIAL', 'GAP']:
                requirements = self.mapper.get_requirements_by_status(status)
                if not requirements:
                    continue
                
                f.write(f"### {status} Requirements\n\n")
                
                for req in requirements:
                    f.write(f"#### {req.article}: {req.requirement}\n\n")
                    f.write(f"**Description:** {req.description}\n\n")
                    f.write(f"**Status:** {req.gap_status}\n\n")
                    
                    # Find evidence for this requirement
                    key = f"{req.article}:{req.requirement}"
                    if key in mapping and mapping[key]['evidence']:
                        f.write("**Evidence:**\n\n")
                        f.write("| Model | Metric | Score | Status |\n")
                        f.write("|-------|--------|-------|--------|\n")
                        
                        for evidence in mapping[key]['evidence']:
                            status_icon = "✓" if evidence['passed'] else "✗"
                            f.write(f"| {evidence['model']} | {evidence['metric_id']} | ")
                            f.write(f"{evidence['score']:.2f} | {status_icon} |\n")
                        f.write("\n")
                    else:
                        f.write("**Evidence:** No direct benchmark evidence available.\n\n")
                    
                    if req.notes:
                        f.write(f"**Notes:** {req.notes}\n\n")
                    
                    f.write("---\n\n")
            
            # Gaps and Recommendations
            f.write("## Identified Gaps and Recommendations\n\n")
            gaps = self.mapper.get_requirements_by_status('GAP')
            partial = self.mapper.get_requirements_by_status('PARTIAL')
            
            if gaps:
                f.write("### Critical Gaps\n\n")
                f.write("The following requirements have no benchmark evidence:\n\n")
                for req in gaps:
                    f.write(f"- **{req.article}: {req.requirement}**\n")
                    f.write(f"  - {req.description}\n")
                    f.write(f"  - Recommendation: {req.notes}\n\n")
            
            if partial:
                f.write("### Partial Coverage\n\n")
                f.write("The following requirements have partial evidence:\n\n")
                for req in partial:
                    f.write(f"- **{req.article}: {req.requirement}**\n")
                    f.write(f"  - {req.description}\n")
                    f.write(f"  - Next steps: {req.notes}\n\n")
            
            # Recommended Actions
            f.write("---\n\n")
            f.write("## Recommended Actions\n\n")
            f.write("1. **Address Critical Gaps:** Implement mechanisms for requirements with no evidence\n")
            f.write("2. **Strengthen Partial Coverage:** Enhance documentation and testing for partial requirements\n")
            f.write("3. **Continuous Monitoring:** Establish ongoing benchmarking and compliance tracking\n")
            f.write("4. **Documentation:** Maintain comprehensive technical documentation per Art. 11\n")
            f.write("5. **Human Oversight:** Implement human oversight mechanisms per Art. 14\n\n")
            
            # Footer
            f.write("---\n\n")
            f.write("*This report was automatically generated by the Compliance Mapper tool.*\n")
            f.write(f"*Source metrics: {len(metrics)} benchmark results*\n")
            f.write(f"*Compliance framework: EU AI Act*\n")
    
    def _generate_json_report(
        self,
        output_file: Path,
        metrics: List[Dict[str, Any]],
        mapping: Dict[str, Any],
        coverage: Dict[str, Any]
    ):
        """Generate JSON format report."""
        report = {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat() + 'Z',
                'framework': 'EU AI Act',
                'models_evaluated': list(set(m['model_name'] for m in metrics)),
                'frameworks_used': list(set(m['framework'] for m in metrics)),
                'total_tests': len(metrics)
            },
            'coverage_summary': coverage,
            'requirements': {},
            'gaps': [],
            'recommendations': []
        }
        
        # Add requirements with evidence
        for key, value in mapping.items():
            req = value['requirement']
            report['requirements'][key] = {
                'article': req.article,
                'requirement': req.requirement,
                'description': req.description,
                'status': req.gap_status,
                'evidence': value['evidence'],
                'notes': req.notes
            }
        
        # Add gaps
        for req in self.mapper.get_requirements_by_status('GAP'):
            report['gaps'].append({
                'article': req.article,
                'requirement': req.requirement,
                'description': req.description,
                'notes': req.notes
            })
        
        # Add recommendations
        report['recommendations'] = [
            "Address critical gaps by implementing missing mechanisms",
            "Strengthen partial coverage with enhanced documentation",
            "Establish continuous monitoring and compliance tracking",
            "Maintain comprehensive technical documentation",
            "Implement human oversight mechanisms"
        ]
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

