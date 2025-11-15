"""
Compliance mapper for linking metrics to legal requirements.
"""

import csv
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass


@dataclass
class ComplianceRequirement:
    """Represents a single compliance requirement."""
    article: str
    requirement: str
    description: str
    metric_ids: List[str]
    evidence_path: str
    gap_status: str  # COVERED, PARTIAL, GAP
    notes: str
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> 'ComplianceRequirement':
        """Create from CSV row."""
        metric_ids = [m.strip() for m in row['Metric_IDs'].split(';') if m.strip()]
        return cls(
            article=row['Article'],
            requirement=row['Requirement'],
            description=row['Description'],
            metric_ids=metric_ids,
            evidence_path=row['Evidence_Path'],
            gap_status=row['Gap_Status'],
            notes=row['Notes']
        )


class ComplianceMapper:
    """Maps benchmark metrics to EU AI Act requirements."""
    
    def __init__(self, matrix_path: str = 'compliance/eu_ai_act_matrix.csv'):
        """
        Initialize compliance mapper.
        
        Args:
            matrix_path: Path to the compliance matrix CSV
        """
        self.matrix_path = Path(matrix_path)
        self.requirements: List[ComplianceRequirement] = []
        self._load_matrix()
    
    def _load_matrix(self):
        """Load compliance matrix from CSV."""
        with open(self.matrix_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.requirements.append(ComplianceRequirement.from_csv_row(row))
    
    def get_requirements_for_metric(self, metric_id: str) -> List[ComplianceRequirement]:
        """
        Get all requirements that reference a specific metric.
        
        Args:
            metric_id: Metric identifier (e.g., "trust_eval:toxicity")
            
        Returns:
            List of requirements that reference this metric
        """
        matching = []
        for req in self.requirements:
            if metric_id in req.metric_ids or 'ALL_METRICS' in req.metric_ids:
                matching.append(req)
        return matching
    
    def get_requirements_by_status(self, status: str) -> List[ComplianceRequirement]:
        """
        Get all requirements with a specific gap status.
        
        Args:
            status: Gap status (COVERED, PARTIAL, GAP)
            
        Returns:
            List of matching requirements
        """
        return [req for req in self.requirements if req.gap_status == status]
    
    def get_requirements_by_article(self, article: str) -> List[ComplianceRequirement]:
        """
        Get all requirements for a specific article.
        
        Args:
            article: Article identifier (e.g., "Art. 10")
            
        Returns:
            List of requirements for this article
        """
        return [req for req in self.requirements if req.article == article]
    
    def get_all_referenced_metrics(self) -> Set[str]:
        """
        Get set of all metric IDs referenced in the matrix.
        
        Returns:
            Set of metric identifiers
        """
        metrics = set()
        for req in self.requirements:
            for metric_id in req.metric_ids:
                if metric_id != 'ALL_METRICS' and metric_id != 'N/A':
                    metrics.add(metric_id)
        return metrics
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """
        Get summary of compliance coverage.
        
        Returns:
            Dictionary with coverage statistics
        """
        total = len(self.requirements)
        covered = len(self.get_requirements_by_status('COVERED'))
        partial = len(self.get_requirements_by_status('PARTIAL'))
        gaps = len(self.get_requirements_by_status('GAP'))
        
        return {
            'total_requirements': total,
            'covered': covered,
            'partial': partial,
            'gaps': gaps,
            'coverage_percentage': (covered / total * 100) if total > 0 else 0,
            'partial_coverage_percentage': ((covered + partial) / total * 100) if total > 0 else 0
        }
    
    def map_metric_to_requirements(self, metric_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Map benchmark results to compliance requirements.
        
        Args:
            metric_results: List of benchmark result dictionaries
            
        Returns:
            Dictionary mapping requirements to their evidence
        """
        mapping = {}
        
        for result in metric_results:
            metric_id = f"{result['framework'].lower()}:{result['test_name']}"
            requirements = self.get_requirements_for_metric(metric_id)
            
            for req in requirements:
                key = f"{req.article}:{req.requirement}"
                if key not in mapping:
                    mapping[key] = {
                        'requirement': req,
                        'evidence': []
                    }
                
                mapping[key]['evidence'].append({
                    'metric_id': metric_id,
                    'model': result['model_name'],
                    'score': result['score'],
                    'passed': result['passed'],
                    'timestamp': result['timestamp']
                })
        
        return mapping

