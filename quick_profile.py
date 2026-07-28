"""Quick 5-step profile of the Gigatoken trainer."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import train_gigatoken as tg
tg.STEPS = 5
import os
os.chdir(os.path.dirname(__file__) or ".")
sys.argv = ['quick_profile.py', '--text', 'shakespeare.txt']
tg.main()
