"""Report generation for model evaluation results.

This module provides functionality to generate formatted evaluation reports
from evaluation results, including metrics tables, confusion matrix breakdowns,
error analysis, and feature importance summaries.
"""

from src.model_evaluation.data_models import (
    EvaluationResult,
    EvaluationReport,
)


class ReportGenerator:
    """Generates formatted evaluation reports from evaluation results.
    
    This class formats evaluation metrics, confusion matrix data, error analysis,
    and feature importance into human-readable text for display and export.
    """
    
    def generate_report(self, eval_result: EvaluationResult) -> EvaluationReport:
        """Generate a comprehensive evaluation report.
        
        Args:
            eval_result: Complete evaluation result with all metrics and analysis
            
        Returns:
            EvaluationReport with formatted text for all sections
        """
        summary = self._format_summary(eval_result)
        metrics_table = self._format_metrics_table(eval_result)
        confusion_breakdown = self._format_confusion_breakdown(eval_result)
        error_analysis_text = self._format_error_analysis(eval_result)
        feature_importance_text = self._format_feature_importance(eval_result)
        
        return EvaluationReport(
            summary=summary,
            metrics_table=metrics_table,
            confusion_breakdown=confusion_breakdown,
            error_analysis_text=error_analysis_text,
            feature_importance_text=feature_importance_text,
        )
    
    def _format_summary(self, eval_result: EvaluationResult) -> str:
        """Format a brief summary of evaluation results."""
        metrics = eval_result.metrics
        return (
            f"Model Evaluation Report: {eval_result.model_name}\n"
            f"Evaluated at: {eval_result.timestamp}\n"
            f"Overall Accuracy: {metrics.accuracy:.2%}"
        )

    def _format_metrics_table(self, eval_result: EvaluationResult) -> str:
        """Format metrics as a table with accuracy, precision, recall, F1-score."""
        metrics = eval_result.metrics
        
        header = "=" * 40
        lines = [
            header,
            "Classification Metrics",
            header,
            f"{'Metric':<20} {'Value':>15}",
            "-" * 40,
            f"{'Accuracy':<20} {metrics.accuracy:>15.4f}",
            f"{'Precision':<20} {metrics.precision:>15.4f}",
            f"{'Recall':<20} {metrics.recall:>15.4f}",
            f"{'F1-Score':<20} {metrics.f1_score:>15.4f}",
            header,
            "",
            "Classification Report:",
            metrics.classification_report,
        ]
        
        return "\n".join(lines)
    
    def _format_confusion_breakdown(self, eval_result: EvaluationResult) -> str:
        """Format confusion matrix breakdown with TP, TN, FP, FN."""
        cm = eval_result.confusion_matrix
        total = cm.tp + cm.tn + cm.fp + cm.fn
        
        header = "=" * 40
        lines = [
            header,
            "Confusion Matrix Breakdown",
            header,
            f"{'Metric':<25} {'Count':>10}",
            "-" * 40,
            f"{'True Positives (TP)':<25} {cm.tp:>10}",
            f"{'True Negatives (TN)':<25} {cm.tn:>10}",
            f"{'False Positives (FP)':<25} {cm.fp:>10}",
            f"{'False Negatives (FN)':<25} {cm.fn:>10}",
            "-" * 40,
            f"{'Total Samples':<25} {total:>10}",
            header,
        ]
        
        return "\n".join(lines)
    
    def _format_error_analysis(self, eval_result: EvaluationResult) -> str:
        """Format error analysis with FP and FN rates."""
        ea = eval_result.error_analysis
        
        header = "=" * 40
        lines = [
            header,
            "Error Analysis",
            header,
            f"{'Metric':<25} {'Value':>10}",
            "-" * 40,
            f"{'False Positive Count':<25} {ea.fp_count:>10}",
            f"{'False Negative Count':<25} {ea.fn_count:>10}",
            f"{'False Positive Rate':<25} {ea.fp_rate:>10.4f}",
            f"{'False Negative Rate':<25} {ea.fn_rate:>10.4f}",
            header,
            "",
            "Interpretation:",
            ea.interpretation,
        ]
        
        return "\n".join(lines)
    
    def _format_feature_importance(self, eval_result: EvaluationResult) -> str:
        """Format feature importance lists."""
        fi = eval_result.feature_importance
        
        if fi is None:
            return "Feature importance not available for this model."
        
        header = "=" * 50
        lines = [
            header,
            f"Feature Importance ({fi.model_type} model)",
            header,
        ]
        
        # Top features pushing towards fake news
        lines.append("")
        lines.append("Top Features for Fake News Classification:")
        lines.append("-" * 50)
        lines.append(f"{'Rank':<6} {'Feature':<30} {'Importance':>12}")
        lines.append("-" * 50)
        
        for rank, (feature, importance) in enumerate(fi.top_fake_features, 1):
            lines.append(f"{rank:<6} {feature:<30} {importance:>12.4f}")
        
        # Top features pushing towards real news
        lines.append("")
        lines.append("Top Features for Real News Classification:")
        lines.append("-" * 50)
        lines.append(f"{'Rank':<6} {'Feature':<30} {'Importance':>12}")
        lines.append("-" * 50)
        
        for rank, (feature, importance) in enumerate(fi.top_real_features, 1):
            lines.append(f"{rank:<6} {feature:<30} {importance:>12.4f}")
        
        lines.append(header)
        
        return "\n".join(lines)
