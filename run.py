import os
from agent import Agent
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    logger = Logger()
    agent = Agent(logger)
    agent.run()

if __name__ == '__main__':
    main()
