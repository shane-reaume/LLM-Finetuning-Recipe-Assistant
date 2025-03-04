#!/usr/bin/env python3
"""
GPU memory optimization utility for PyTorch training
"""

import os
import argparse
import subprocess
import torch
import sys
from pathlib import Path

# Add project root to path to import from config_optimizer.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config_optimizer import get_system_info
except ImportError:
    # Fallback if config_optimizer can't be imported
    def get_system_info():
        return {"cpu": {"cores": 0, "threads": 0, "memory_gb": 0}, 
                "gpu": {"available": False, "count": 0, "devices": []}}

def print_gpu_memory_stats():
    """Print current GPU memory usage statistics"""
    if torch.cuda.is_available():
        print("\nGPU Memory Statistics:")
        print("-" * 50)
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Get current memory usage
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            max_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            
            print(f"  - Allocated: {allocated:.2f} GB")
            print(f"  - Reserved:  {reserved:.2f} GB")
            print(f"  - Total:     {max_memory:.2f} GB")
            print(f"  - Utilization: {(allocated/max_memory)*100:.1f}%")
            
            # Additional memory stats if available
            try:
                stats = torch.cuda.memory_stats(i)
                if stats:
                    print(f"  - Fragmentation: {stats.get('fragmentation', 0):.1f}%")
                    print(f"  - Active blocks: {stats.get('active_blocks.all.current', 0)}")
            except:
                pass
    else:
        print("No CUDA-capable GPU detected")

def suggest_roundup_power2_divisions(total_memory_gb):
    """Suggest optimal roundup_power2_divisions setting based on GPU memory"""
    if total_memory_gb >= 24:
        return "[256:1,512:2,1024:4,>:8]"  # More divisions for large memory
    elif total_memory_gb >= 12:
        return "[256:1,512:2,>:4]"  # Balanced for mid-range GPUs
    else:
        return "[512:1,>:2]"  # Minimal for small GPUs

def optimize_memory_settings():
    """Return recommended PyTorch CUDA memory optimization settings"""
    if not torch.cuda.is_available():
        return "PYTORCH_NO_CUDA_MEMORY_CACHING=1"
    
    # Get GPU info
    gpu_props = torch.cuda.get_device_properties(0)
    total_memory_gb = gpu_props.total_memory / (1024**3)
    
    # Get roundup power2 divisions
    roundup = suggest_roundup_power2_divisions(total_memory_gb)
    
    # Optimize based on available memory
    if total_memory_gb >= 24:  # High-end GPU
        return f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6,roundup_power2_divisions:{roundup}"
    elif total_memory_gb >= 16:  # Mid-range GPU
        return f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.7,roundup_power2_divisions:{roundup}"
    elif total_memory_gb >= 8:  # Entry-level discrete GPU 
        return f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8,roundup_power2_divisions:{roundup}"
    else:  # Low memory GPU
        return f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32,garbage_collection_threshold:0.9,roundup_power2_divisions:{roundup}"

def suggest_batch_size():
    """Recommend batch size based on GPU memory"""
    # Use the function from config_optimizer.py if available
    try:
        from config_optimizer import benchmark_gpu
        result = benchmark_gpu("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        if result and "recommended_batch_size" in result:
            return result["recommended_batch_size"]
    except:
        pass
        
    # Fallback implementation
    if not torch.cuda.is_available():
        return 1
    
    # Get GPU info
    gpu_props = torch.cuda.get_device_properties(0)
    total_memory_gb = gpu_props.total_memory / (1024**3)
    
    # Very rough estimation for TinyLlama with LoRA
    if total_memory_gb >= 32:
        return 8
    elif total_memory_gb >= 24:
        return 6
    elif total_memory_gb >= 16:
        return 4
    elif total_memory_gb >= 12:
        return 3
    elif total_memory_gb >= 8:
        return 2
    else:
        return 1

def run_memory_test():
    """Run a simple test to measure max GPU memory usage"""
    if not torch.cuda.is_available():
        print("No CUDA-capable GPU detected")
        return
    
    print("Running memory allocation test...")
    
    # Start with a small tensor and gradually increase
    max_allocated = 0
    allocation_step_gb = 0.5
    tensors = []
    
    try:
        print("Press Ctrl+C to stop the test when you see memory errors approaching")
        while True:
            # Create a tensor of size allocation_step_gb
            tensor_size = int(allocation_step_gb * (1024**3) / 4)  # 4 bytes per float32
            tensors.append(torch.rand(tensor_size, device='cuda'))
            
            # Check memory usage
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            max_allocated = max(max_allocated, allocated)
            
            print(f"Allocated: {allocated:.2f} GB (max: {max_allocated:.2f} GB)")
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except RuntimeError as e:
        print(f"\nReached maximum memory allocation: {e}")
    
    # Clean up
    tensors = None
    torch.cuda.empty_cache()
    
    print(f"Maximum successful allocation: {max_allocated:.2f} GB")
    print("Memory test complete")

def main():
    parser = argparse.ArgumentParser(description="Optimize GPU memory usage for PyTorch training")
    parser.add_argument('--stats', action='store_true', help='Display GPU memory statistics')
    parser.add_argument('--test', action='store_true', help='Run memory allocation test')
    parser.add_argument('--env', action='store_true', help='Show recommended environment variables')
    parser.add_argument('--batch', action='store_true', help='Suggest batch size based on GPU memory')
    parser.add_argument('--system', action='store_true', help='Show full system information (using config_optimizer)')
    args = parser.parse_args()
    
    # Default behavior if no args specified: show all info
    if not any(vars(args).values()):
        args.stats = args.env = args.batch = True
    
    if args.stats:
        print_gpu_memory_stats()
    
    if args.env:
        print("\nRecommended environment variable:")
        print(optimize_memory_settings())
        print("\nUsage example:")
        print(f"{optimize_memory_settings()} python -m src.model.recipe_train --config config/text_generation_optimized.yaml")
    
    if args.batch:
        print("\nRecommended batch size for your GPU:", suggest_batch_size())
    
    if args.test:
        run_memory_test()
        
    if args.system:
        try:
            sys_info = get_system_info()
            print("\nSystem Information:")
            print(f"CPU: {sys_info['cpu']['cores']} cores / {sys_info['cpu']['threads']} threads")
            print(f"RAM: {sys_info['cpu']['memory_gb']} GB")
            
            if sys_info["gpu"]["available"]:
                print("\nGPU Information:")
                for i, device in enumerate(sys_info["gpu"]["devices"]):
                    print(f"  GPU {i}: {device['name']}" + 
                          (f" ({device['memory_gb']} GB)" if device['memory_gb'] else ""))
        except:
            print("Failed to get system information from config_optimizer.py")

if __name__ == '__main__':
    main()
