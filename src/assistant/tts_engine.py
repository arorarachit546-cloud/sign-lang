"""
Non-blocking Text-to-Speech (TTS) Engine
Uses pyttsx3 with background thread queueing to synthesize speech without blocking video inference.
"""

import queue
import threading
from typing import Optional


class TTSEngine:
    """
    Thread-safe asynchronous speech synthesizer.
    Pushes text requests to a worker queue to avoid stutter in live 30 FPS video feeds.
    """

    def __init__(self, rate: int = 160, volume: float = 0.9, enabled: bool = True):
        self.rate = rate
        self.volume = volume
        self.enabled = enabled
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._tts_available = False

        if self.enabled:
            self._start_worker()

    def _start_worker(self):
        try:
            import pyttsx3
            # Test engine instantiation
            test_engine = pyttsx3.init()
            test_engine.stop()
            self._tts_available = True
        except Exception:
            self._tts_available = False

        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()

    def _run_loop(self):
        engine = None
        if self._tts_available:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
            except Exception:
                engine = None

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:  # Sentinel to terminate
                break

            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
            else:
                # Log to console if speech hardware unavailable
                print(f"[TTS Audio Output]: \"{text}\"")

            self._queue.task_done()

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def speak(self, text: str) -> None:
        """Enqueues text for spoken pronunciation."""
        if not self.enabled or not text.strip():
            return
        self._queue.put(text.strip())

    def stop(self) -> None:
        """Stops worker thread cleanly."""
        self._stop_event.set()
        self._queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
