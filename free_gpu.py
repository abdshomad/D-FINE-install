#!/usr/bin/env python3
"""
Free up GPU memory by killing processes using GPUs.

This script identifies and optionally kills processes that are using GPU resources.
Useful for clearing GPU memory before training or when encountering OOM errors.
"""

import subprocess
import sys
import os
import argparse
from typing import List, Dict


def get_gpu_processes() -> List[Dict[str, str]]:
    """Get list of processes using GPUs."""
    try:
        # Use nvidia-smi to get GPU processes
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory,gpu_uuid', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 3:
                    processes.append({
                        'pid': parts[0].strip(),
                        'process_name': parts[1].strip(),
                        'used_memory': parts[2].strip() if len(parts) > 2 else 'N/A',
                        'gpu_uuid': parts[3].strip() if len(parts) > 3 else 'N/A'
                    })
        return processes
    except subprocess.CalledProcessError:
        print("Error: Could not query GPU processes. Is nvidia-smi available?")
        return []
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Are NVIDIA drivers installed?")
        return []


def kill_process(pid: str, force: bool = False) -> bool:
    """Kill a process by PID."""
    try:
        signal = 'SIGKILL' if force else 'SIGTERM'
        subprocess.run(['kill', f'-{9 if force else 15}', pid], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Free up GPU memory by killing processes using GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List processes using GPUs
  python free_gpu.py

  # Kill all processes using GPUs (graceful)
  python free_gpu.py --kill

  # Force kill all processes using GPUs
  python free_gpu.py --kill --force

  # Kill specific process
  python free_gpu.py --kill --pid 12345
        """
    )
    parser.add_argument('--kill', action='store_true', help='Kill processes (default: just list them)')
    parser.add_argument('--force', action='store_true', help='Force kill (SIGKILL) instead of graceful (SIGTERM)')
    parser.add_argument('--pid', type=str, help='Kill specific process by PID')
    parser.add_argument('--exclude-self', action='store_true', default=True, help='Exclude current Python process (default: True)')
    
    args = parser.parse_args()
    
    # Get current process PID if excluding self
    current_pid = None
    if args.exclude_self:
        current_pid = str(os.getpid())
    
    processes = get_gpu_processes()
    
    if not processes:
        print("No processes found using GPUs.")
        return 0
    
    print(f"\nFound {len(processes)} process(es) using GPU(s):\n")
    print(f"{'PID':<10} {'Process Name':<30} {'Memory':<15} {'GPU UUID':<40}")
    print("-" * 95)
    
    for proc in processes:
        pid = proc['pid']
        if args.exclude_self and pid == current_pid:
            continue
        print(f"{pid:<10} {proc['process_name']:<30} {proc['used_memory']:<15} {proc['gpu_uuid']:<40}")
    
    if args.pid:
        # Kill specific PID
        if args.kill:
            print(f"\nKilling process {args.pid}...")
            if kill_process(args.pid, args.force):
                print(f"Successfully killed process {args.pid}")
            else:
                print(f"Failed to kill process {args.pid}")
                return 1
        else:
            print(f"\nUse --kill to kill process {args.pid}")
    elif args.kill:
        # Kill all processes
        killed_count = 0
        failed_count = 0
        
        print(f"\nKilling {len(processes)} process(es)...")
        for proc in processes:
            pid = proc['pid']
            if args.exclude_self and pid == current_pid:
                print(f"Skipping current process (PID {pid})")
                continue
            
            print(f"Killing PID {pid} ({proc['process_name']})...")
            if kill_process(pid, args.force):
                killed_count += 1
                print(f"  ✓ Killed PID {pid}")
            else:
                failed_count += 1
                print(f"  ✗ Failed to kill PID {pid}")
        
        print(f"\nSummary: {killed_count} killed, {failed_count} failed")
        
        if failed_count > 0:
            return 1
    else:
        print("\nUse --kill to kill these processes, or --kill --force for force kill")
        print("Use --pid <PID> to kill a specific process")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
