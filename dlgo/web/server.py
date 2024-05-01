import os
from os import path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder    # New import
from fastapi.staticfiles import StaticFiles

from dlgo import agent
from dlgo import goboard_fast as goboard
from dlgo.utils import coords_from_point
from dlgo.utils import point_from_coords
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5000",
    "http://localhost",
    "http://localhost:8080",
]


def web_app(bot_map):

    here = os.path.dirname(__file__)
    static_path = os.path.join(here, 'static')
    #print(static_path)

    app = FastAPI(title="web_app")

    app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"message": "Hello World"}
    
    @app.post("/select-move/{bot_name}")
    async def select_move(bot_name, request: Request):
        content = await request.json()
        print(content)
        board_size = content['board_size']
        game_state = goboard.GameState.new_game(board_size)

        print(content, "TESTING")
        # Replay the game up to this point.
        for move in content['moves']:
            if move == 'pass':
                next_move = goboard.Move.pass_turn()
            elif move == 'resign':
                next_move = goboard.Move.resign()
            else:
                next_move = goboard.Move.play(point_from_coords(move))
            game_state = game_state.apply_move(next_move)
        bot_agent = bot_map[bot_name]
        bot_move = bot_agent.select_move(game_state)
        if bot_move.is_pass:
            bot_move_str = 'pass'
        elif bot_move.is_resign:
            bot_move_str = 'resign'
        else:
            bot_move_str = coords_from_point(bot_move.point)
        return jsonable_encoder({
            'bot_move': bot_move_str,
            'diagnostics': bot_agent.diagnostics()
        })
    

    app.mount("/static", StaticFiles(directory=static_path, html=True), name="static")

    return app
