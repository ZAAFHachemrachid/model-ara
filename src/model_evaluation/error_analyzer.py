"""Error analysis for model evaluation.

This module provides the ErrorAnalyzer class for analyzing prediction errors,
computing error rates, and generating interpretations for fake news detection.
"""

import logging

from src.model_evaluation.data_models import ConfusionMatrixData, ErrorAnalysis


logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Analyzes prediction errors for model evaluation.
    
    This class computes false positive and false negative rates,
    and provides interpretation of error criticality for fake news detection.
    """
    
    def analyze_errors(self, cm_data: ConfusionMatrixData) -> ErrorAnalysis:
        """Analyze prediction errors from confusion matrix data.
        
        Computes false positive rate (FP / (TN + FP)) and false negative rate
        (FN / (TP + FN)), and generates interpretation text about which error
        type is more critical for fake news detection.
        
        Args:
            cm_data: ConfusionMatrixData containing TP, TN, FP, FN values.
        
        Returns:
            ErrorAnalysis with counts, rates, and interpretation.
        """
        fp_count = cm_data.fp
        fn_count = cm_data.fn
        
        # Compute false positive rate: FP / (TN + FP)
        # Handle division by zero
        fp_denominator = cm_data.tn + cm_data.fp
        if fp_denominator == 0:
            logger.warning(
                "Cannot compute FP rate: TN + FP = 0. Setting FP rate to 0.0"
            )
            fp_rate = 0.0
        else:
            fp_rate = cm_data.fp / fp_denominator
        
        # Compute false negative rate: FN / (TP + FN)
        # Handle division by zero
        fn_denominator = cm_data.tp + cm_data.fn
        if fn_denominator == 0:
            logger.warning(
                "Cannot compute FN rate: TP + FN = 0. Setting FN rate to 0.0"
            )
            fn_rate = 0.0
        else:
            fn_rate = cm_data.fn / fn_denominator
        
        # Generate interpretation about error criticality
        interpretation = self._generate_interpretation(
            fp_count, fn_count, fp_rate, fn_rate
        )
        
        return ErrorAnalysis(
            fp_count=fp_count,
            fn_count=fn_count,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
            interpretation=interpretation,
        )
    
    def _generate_interpretation(
        self,
        fp_count: int,
        fn_count: int,
        fp_rate: float,
        fn_rate: float,
    ) -> str:
        """Generate interpretation text about error criticality.
        
        For fake news detection:
        - False Positives (FP): Real news classified as fake - damages credibility
        - False Negatives (FN): Fake news classified as real - allows misinformation
        
        Args:
            fp_count: Number of false positives.
            fn_count: Number of false negatives.
            fp_rate: False positive rate.
            fn_rate: False negative rate.
        
        Returns:
            Interpretation text explaining error criticality.
        """
        lines = []
        
        lines.append("Error Analysis for Fake News Detection:")
        lines.append("")
        lines.append(
            f"- False Positives (FP): {fp_count} cases "
            f"(rate: {fp_rate:.2%})"
        )
        lines.append(
            "  Real news incorrectly classified as fake. "
            "This damages credibility and may suppress legitimate information."
        )
        lines.append("")
        lines.append(
            f"- False Negatives (FN): {fn_count} cases "
            f"(rate: {fn_rate:.2%})"
        )
        lines.append(
            "  Fake news incorrectly classified as real. "
            "This allows misinformation to spread unchecked."
        )
        lines.append("")
        
        # Determine which error type is more critical
        if fn_rate > fp_rate:
            lines.append(
                "Critical Finding: False Negative rate is higher than False Positive rate. "
                "The model is more likely to miss fake news (classify it as real), "
                "which is particularly concerning for misinformation prevention. "
                "Consider adjusting the classification threshold to reduce false negatives."
            )
        elif fp_rate > fn_rate:
            lines.append(
                "Critical Finding: False Positive rate is higher than False Negative rate. "
                "The model is more likely to flag real news as fake, "
                "which could damage trust in legitimate sources. "
                "Consider adjusting the classification threshold to reduce false positives."
            )
        else:
            lines.append(
                "Finding: False Positive and False Negative rates are balanced. "
                "The model shows similar error patterns for both types of misclassification."
            )
        
        return "\n".join(lines)
