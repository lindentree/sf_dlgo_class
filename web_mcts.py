import argparse
import uvicorn
from dlgo import web
from dlgo import mcts

BOARD_SIZE = 5


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--bind-address', default='127.0.0.1')
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--pg-agent')
    parser.add_argument('--predict-agent')
    parser.add_argument('--q-agent')
    parser.add_argument('--ac-agent')

    args = parser.parse_args()
    bot = mcts.MCTSAgent(700, temperature=1.4)
    web_app = web.web_app(bot)
    uvicorn.run(web_app, host=args.bind_address, port=args.port)



if __name__ == '__main__':
    main()
