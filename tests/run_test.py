"""
Simple run script to test network stability without unicode issues
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from test_network_stability import main

if __name__ == "__main__":
    success = main()
    print(f"\nOverall result: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
