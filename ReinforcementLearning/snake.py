'''Snake game using a numpy array for visualization'''

import numpy as np

class Game():

    def __init__(self,grid,snake_size, snake_pad=2):
        self.board = np.ones((grid,grid,3))
        self.snake = Snake(self.board,snake_size)
        self.pad = snake_pad
        self.apple = None
        self.score = 0
        self.total = 0
        self.deaths = 0

    def reset(self):
        self.snake.reset(self.pad)
        self.spawn_apple()

    def draw(self):
        # white background
        self.board[:,:,:] = 1
        # black snake
        self.board[self.snake.pos[0],self.snake.pos[1],:] = 0
        for s in self.snake.segments:
            self.board[s[0],s[1],:] = 0
        # red apple
        self.board[self.apple[0],self.apple[1],1:] = 0

    def spawn_apple(self):
        # dummy x,y for while loop
        x,y = self.snake.pos
        # keep generating new apples until a free cell is rolled
        while ((x,y) in self.snake.segments+[self.snake.pos]):
            x = np.random.randint(self.board.shape[0])
            y = np.random.randint(self.board.shape[1])
        self.apple = np.array([x,y],dtype=int)

    def iterate(self):
        reward = 0
        episode = 0
        # move snake
        self.snake.move()
        # check death condition
        grid = len(self.board)
        p = self.snake.pos
        a = self.apple
        if ((p[0]<0 or p[0]>=grid or p[1]<0 or p[1]>=grid) or (p in self.snake.segments)):
            self.lose()
            episode = -1
            reward = -1
        elif (p == a).all():
            # eat apple
            self.eat()
            episode = 1
            reward = 1
        return reward, episode

    def lose(self):
        self.reset()
        self.score=0
        self.deaths+=1

    def eat(self):
        self.snake.eat()
        self.score+=1
        self.total+=1
        old = self.apple
        self.spawn_apple()

    def get_state(self):
        grid = len(self.board)
        state = np.zeros(6)
        x,y = self.snake.pos
        'S,E,N,W'
        if ((x+1,y) in self.snake.segments or x+1>=grid):
            state[0] = 1
        if ((x-1,y) in self.snake.segments or x-1<0):
            state[2] = 1
        if ((x,y+1) in self.snake.segments or y+1>=grid):
            state[1] = 1
        if ((x,y-1) in self.snake.segments or y-1<0):
            state[3] = 1
        state[4] = np.sign(x-self.apple[0])
        state[5] = np.sign(y-self.apple[1])
        return state

class Snake():
    
    movement = ([[1,0],[0,1],[-1,0],[0,-1]])
    
    def __init__(self,board,starting_size):
        # the board on which the snake is moving
        self.grid = board.shape
        # position of head
        self.pos = None
        # direction facing
        self.d = 0
        # number of tail segments
        self.size = starting_size
        self.s_size = starting_size
        # position of tail segments
        self.segments = []
    
    def reset(self,pad=2):
        # pad starting position
        x = np.random.randint(pad,self.grid[0]-pad)
        y = np.random.randint(pad,self.grid[1]-pad)
        self.pos = (x,y)
        self.d = np.random.randint(4)
        self.size = self.s_size
        self.segments = []
        
    def move(self):
        # add head's position to segments
        self.segments.append(self.pos)
        # move head
        x,y = self.pos
        dx,dy = self.movement[self.d]
        self.pos = (x+dx,y+dy)
        # remove one segment if needed
        seg = None
        if (len(self.segments)>self.size):
            seg = self.segments.pop(0)
        # return new position and position that has been cleared
        return self.pos, seg
    
    def eat(self):
        self.size+=1

if (__name__ == '__main__'):
    print('This module is not intended to run by iself')
