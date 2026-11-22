"""
Real-time Sliding Window Buffer and Debouncing Manager
Maintains temporal landmark queues and implements consensus-based debouncing logic
to ensure stable, non-flickering gesture translations.
"""

from collections import deque
from typing import Deque, List, Optional, Tuple
import numpy as np


class SlidingWindowBuffer:
    """
    FIFO queue of fixed capacity T (default 30 frames) for real-time temporal inference.
    """

    def __init__(self, window_size: int = 30, feature_dim: int = 201):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.buffer: Deque[np.ndarray] = deque(maxlen=window_size)

    def append(self, frame_features: np.ndarray) -> None:
        """Appends a 1D landmark feature vector of shape (feature_dim,) to buffer."""
        assert frame_features.shape[0] == self.feature_dim, (
            f"Expected feature dim {self.feature_dim}, got {frame_features.shape[0]}"
        )
        self.buffer.append(frame_features)

    def is_ready(self) -> bool:
        """Returns True when buffer has accumulated a full temporal window."""
        return len(self.buffer) == self.window_size

    def get_window(self) -> np.ndarray:
        """Returns stacked numpy array of shape (T, feature_dim)."""
        return np.array(self.buffer, dtype=np.float32)

    def clear(self) -> None:
        self.buffer.clear()


class TemporalDebouncer:
    """
    Debouncer that requires K consecutive identical predictions with confidence
    above a threshold before committing a new gesture to prevent flickers and false positives.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        consecutive_frames_required: int = 5,
        cooldown_frames: int = 15,
    ):
        self.confidence_threshold = confidence_threshold
        self.consecutive_frames_required = consecutive_frames_required
        self.cooldown_frames = cooldown_frames

        self.last_predicted_class: Optional[int] = None
        self.consecutive_count: int = 0
        self.committed_class: Optional[int] = None
        self.cooldown_counter: int = 0
        self.translation_history: List[str] = []

    def update(
        self,
        predicted_class: int,
        confidence: float,
        class_name: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Updates debouncer state with the current frame's prediction.

        Returns:
            committed: bool, True if a new gesture was officially committed
            committed_name: str or None, the committed class name
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, None

        if confidence < self.confidence_threshold:
            self.consecutive_count = 0
            self.last_predicted_class = None
            return False, None

        if predicted_class == self.last_predicted_class:
            self.consecutive_count += 1
        else:
            self.last_predicted_class = predicted_class
            self.consecutive_count = 1

        # Check if threshold reached
        if self.consecutive_count >= self.consecutive_frames_required:
            if self.committed_class != predicted_class:
                self.committed_class = predicted_class
                self.translation_history.append(class_name)
                self.cooldown_counter = self.cooldown_frames  # Enter cooldown
                self.consecutive_count = 0
                return True, class_name

        return False, None

    def get_sentence(self, max_tokens: int = 6) -> str:
        """Returns the recent committed tokens as a space-separated transcript."""
        tokens = self.translation_history[-max_tokens:]
        return " ".join(tokens) if tokens else ""

    def reset(self) -> None:
        self.last_predicted_class = None
        self.consecutive_count = 0
        self.committed_class = None
        self.cooldown_counter = 0
        self.translation_history.clear()
