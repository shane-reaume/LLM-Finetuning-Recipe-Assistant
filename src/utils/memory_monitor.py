import torch
import threading
import time
import logging

class GPUMemoryMonitor:
    """Monitor GPU memory usage during training."""
    
    def __init__(self, interval_seconds=60, log_level=logging.INFO):
        self.interval = interval_seconds
        self.keep_running = False
        self.thread = None
        self.logger = logging.getLogger("MemoryMonitor")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
    def _monitor(self):
        peak_allocated = 0
        while self.keep_running:
            if torch.cuda.is_available():
                # Get current memory stats
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                peak_allocated = max(peak_allocated, allocated)
                
                self.logger.info(f"GPU Memory: {allocated:.2f}GB allocated, "
                                f"{reserved:.2f}GB reserved, "
                                f"{peak_allocated:.2f}GB peak")
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring GPU memory in a background thread."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available. Memory monitoring disabled.")
            return
            
        if self.thread is not None and self.thread.is_alive():
            self.logger.warning("Monitor already running")
            return
            
        self.keep_running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        self.logger.info("Memory monitoring started")
        
    def stop(self):
        """Stop the monitoring thread."""
        self.keep_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.logger.info("Memory monitoring stopped")

# Usage example:
# monitor = GPUMemoryMonitor(interval_seconds=30)
# monitor.start()
# ... training loop ...
# monitor.stop()
