"""
Base classes for benchmark runners.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


@dataclass
class BenchmarkResult:
    """Standardized benchmark result format."""
    
    framework: str  # e.g., "TrustEval", "DecodingTrust"
    test_name: str  # e.g., "toxicity", "stereotype_bias"
    model_name: str  # e.g., "gpt-4o-mini", "claude-haiku"
    timestamp: str
    
    # Core metrics
    score: float  # Primary score (0-1 or 0-100, normalized)
    passed: bool  # Whether test passed threshold
    
    # Detailed results
    metrics: Dict[str, Any]  # Framework-specific metrics
    samples_tested: int
    
    # Metadata
    test_config: Dict[str, Any]
    execution_time_seconds: float
    
    # Optional fields
    error: Optional[str] = None
    warnings: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""
    
    def __init__(self, model_name: str, api_key: str, config: Dict[str, Any]):
        """
        Initialize benchmark runner.
        
        Args:
            model_name: Name of the model to test
            api_key: API key for model access
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = config
        self.framework_name = self.__class__.__name__.replace('Runner', '')
    
    @abstractmethod
    async def run_test(self, test_name: str, **kwargs) -> BenchmarkResult:
        """
        Run a specific test.
        
        Args:
            test_name: Name of the test to run
            **kwargs: Additional test-specific parameters
            
        Returns:
            BenchmarkResult object
        """
        pass
    
    @abstractmethod
    async def run_all_tests(self) -> List[BenchmarkResult]:
        """
        Run all available tests for this framework.
        
        Returns:
            List of BenchmarkResult objects
        """
        pass
    
    def _create_result(
        self,
        test_name: str,
        score: float,
        passed: bool,
        metrics: Dict[str, Any],
        samples_tested: int,
        test_config: Dict[str, Any],
        execution_time: float,
        error: Optional[str] = None,
        warnings: Optional[List[str]] = None
    ) -> BenchmarkResult:
        """Helper to create standardized result."""
        return BenchmarkResult(
            framework=self.framework_name,
            test_name=test_name,
            model_name=self.model_name,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            score=score,
            passed=passed,
            metrics=metrics,
            samples_tested=samples_tested,
            test_config=test_config,
            execution_time_seconds=execution_time,
            error=error,
            warnings=warnings
        )

