class DriftMonitor:
    def __init__(self, confidence_threshold=0.45, max_low_confidence=3):
        """
        confidence_threshold: below this = uncertain prediction
        max_low_confidence: how many times before drift is declared
        """
        self.confidence_threshold = confidence_threshold
        self.max_low_confidence = max_low_confidence
        self.low_confidence_count = 0

    def update(self, confidence):
        """
        Update drift monitor with latest prediction confidence.
        Returns True if drift is detected.
        """
        if confidence < self.confidence_threshold:
            self.low_confidence_count += 1
        else:
            self.low_confidence_count = 0

        if self.low_confidence_count >= self.max_low_confidence:
            return True  # Drift detected

        return False

    def reset(self):
        self.low_confidence_count = 0
