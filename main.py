import sys
from agent import run_agent

if __name__ == "__main__":
    #mode = sys.argv[1] # initialize or find
    mode = "find"
    run_agent(mode)