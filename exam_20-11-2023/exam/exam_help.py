import random

import ipywidgets as widgets
from IPython.display import display, HTML


def generate_empty_table(n_rows, n_cols):
    return [[0 for _ in range(n_cols)] for _ in range(n_rows)]


def generate_random_table(n_rows, n_cols, n_agents=1, n_boxes=1, n_goal_fields=1):
    table = generate_empty_table(n_rows, n_cols)
    random_coords = [(row, col) for row in range(n_rows) for col in range(n_cols)]
    n = n_agents + n_boxes + n_goal_fields
    assert n <= len(random_coords), 'Нема слободни места за сите агенти и кутии кои ги планираш'
    fields = random.sample(random_coords, n)
    symbols = [1] * n_agents + [2] * n_boxes + [3] * n_goal_fields
    for (i, j), symbol in zip(fields, symbols):
        table[i][j] = symbol
    return table


class Board(widgets.Output):
    def __init__(self, M, N):
        super().__init__()
        
        self.M, self.N = M, N
        self.transform = lambda x: '🔷' if x == 1 else '🟥' if x == 2 else '🟢' if x == 3 else '·'
        self.update(self.new())
        
    def new(self):
        return [[0 for _ in range(self.N)] for _ in range(self.M)]
    
    def html(self, board):
        board_html = '<div style="font-family: monospace; display: inline-block; line-height: 1.2em;">'
        for row in board:
            row = list(map(self.transform, row))
            for piece in row:
                board_html += f'<span style="width: 2em; height: 2em; display: inline-block;">{piece}</span>'
            board_html += '<br>'
        board_html += '</div>'
        return board_html

    def update(self, new_board):
        with self:
            self.clear_output(wait=True)
            display(HTML(self.html(new_board)), display_id='board_output')
