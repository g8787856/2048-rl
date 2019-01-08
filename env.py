from tkinter import *
from env_logic import *
import time
# from random import *

SIZE = 500
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
BACKGROUND_COLOR_DICT = {   2:"#eee4da", 4:"#ede0c8", 8:"#f2b179", 16:"#f59563", \
                            32:"#f67c5f", 64:"#f65e3b", 128:"#edcf72", 256:"#edcc61", \
                            512:"#edc850", 1024:"#edc53f", 2048:"#edc22e" }
CELL_COLOR_DICT = { 2:"#776e65", 4:"#776e65", 8:"#f9f6f2", 16:"#f9f6f2", \
                    32:"#f9f6f2", 64:"#f9f6f2", 128:"#f9f6f2", 256:"#f9f6f2", \
                    512:"#f9f6f2", 1024:"#f9f6f2", 2048:"#f9f6f2" }
FONT = ("Verdana", 40, "bold")

KEY_UP = 0
KEY_DOWN = 1
KEY_LEFT = 2
KEY_RIGHT = 3

class Game(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')

        self.commands = { KEY_UP: up, KEY_DOWN: down, KEY_LEFT: left, KEY_RIGHT: right }
        self.n_actions = len(self.commands)
        self.n_features = 16

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()
        
        # self.mainloop()
        self.update_idletasks()
        self.update()
        self.matrix = []
        
    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        background.grid()
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE/GRID_LEN, height=SIZE/GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def init_matrix(self):
        self.matrix = new_game(4)

        self.matrix=add_two(self.matrix)
        self.matrix=add_two(self.matrix)

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number], fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()
        self.update()

    def step(self, action):
        reward = 0
        over = False
        now_big, new_big = 0, 0
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                if now_big <= self.matrix[i][j]:
                    now_big = self.matrix[i][j]

        self.matrix, done = self.commands[action](self.matrix)

        if done:
            self.matrix = add_two(self.matrix)
            self.update_grid_cells()
            over, new_big = game_state(self.matrix)

        if new_big == 2048:
            reward = 1
        elif (new_big - now_big == 16):
            reward = 0.1
        elif (new_big - now_big == 32):
            reward = 0.2
        elif (new_big - now_big == 64):
            reward = 0.3
        elif (new_big - now_big == 128):
            reward = 0.4
        elif (new_big - now_big == 256):
            reward = 0.6
        elif (new_big - now_big == 512):
            reward = 0.8

        if new_big == 0:
            reward = -0.5

        if over and new_big != 2048:
            reward = -1

        return self.matrix, reward, over

    def reset(self):
        self.init_matrix()
        self.update_grid_cells()
        return self.matrix

    def render(self):
        time.sleep(0.2)
        self.update_grid_cells()
