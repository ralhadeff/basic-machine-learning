import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, NumericProperty, ListProperty, OptionProperty, ReferenceListProperty
from kivy.graphics import Rectangle, Ellipse, Line, Color
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window

from random import randint
import numpy as np

from qlearn import Qlearn

# snake starting size
snake_size = 4
# game speed
fps = 1
# default paint color
color = (0,0,0)
neurons = 14

class Game(Widget):
    
    # game objects
    snake = ObjectProperty(None)
    apple = ObjectProperty(None)
    # background color
    Window.clearcolor = (1, 1, 1, 1)
    # fixed window size
    w,h = (640,640)
    Window.size = (w, h)
    kivy.config.Config.set('graphics','resizable', False)
    # grid size (number of grid points in x and y)
    grid = 8
    # score (apples eaten)
    score = NumericProperty(0)
    # iteration (game number played)
    iteration = NumericProperty(0)
    average = NumericProperty(0)
    total = 0
    b = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # keyboard binding for testing purposes
        self._keyboard = Window.request_keyboard(self._on_keyboard_down, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        # trick to overcome initial sizing problem
        self.turn_count = 2
        self.brain = Qlearn([neurons,64,4],100000,0.9,5,batch_size=100)
        self.fps = fps

    def start(self):
        # draw border
        with self.canvas.before:
            Color(*color)
            Line(width=3.,
                 rectangle=(0, 0, self.w, self.h))   
        # add objects
        self.new_snake()
        self.new_apple()        
        # start update loop
        self.update()
    
    def reset(self):
        # reset score, increase game countx
        self.score = 0
        if (self.iteration>0):
            self.average = round(self.total/self.iteration,2)
        self.iteration+=1
        # delete existing objects
        self.snake.remove()
        self.apple.remove()
        # create new ones
        self.new_snake()
        self.new_apple()

    def new_snake(self):
        # starting position and direction
        x = randint(snake_size-1,self.grid-(snake_size))
        y = randint(snake_size-1,self.grid-(snake_size))
        d = randint(0,3)
        # assign to snake
        self.snake.head.pos = (x,y)
        self.snake.head.direction

    def new_apple(self, *args):
        x = randint(1, self.grid)
        y = randint(1, self.grid)
        # make sure apple is not generated inside the snake
        occupancy = self.snake.get_occupancy()
        while [x,y] in occupancy:
            x = randint(1, self.grid)
            y = randint(1, self.grid)
        self.apple.add(x,y)

    def check_defeat(self):
        pos = self.snake.head.position
        # snake bites itself
        if pos in self.snake.tail.positions:
            return True
        # snake hit walls
        if not (1<=pos[0]<=self.grid and 1<=pos[1]<=self.grid):
            return True
        # default
        return False

    def update(self, *args):
        # initial sizing problem solution
        if (self.turn_count==2):
            self.turn_count-=1
            return
        elif (self.turn_count==1):
            self.reset()
            self.turn_count-=1
            self.reward = 0
            self.last_action=0
            self.dead=False
            return
        else:
            state = self.get_state()
            # reward from previous action, after the action
            if (self.b):
                action = self.brain.update(self.reward,state)
                self.last_action = action
                self.play_action(action)
            # snake is alive
            if (self.right_direction()):
                self.reward=0.1
            else:
                self.reward=-0.1
            self.snake.move()
            # check defeat
            if self.check_defeat():
                self.reset()
                self.reward=-10
            # check eat
            if (self.snake.head.position == self.apple.pos):
                self.score+= 1
                self.reward=self.score*10
                self.total+= 1
                self.snake.eat()
                self.apple.remove()
                self.new_apple()
    
    def right_direction(self):
        h = self.snake.head.pos
        d = self.snake.head.direction
        a = self.apple.pos
        
        x = np.sign(h[0]-a[0])
        y = np.sign(h[1]-a[1])
        
        # d: up, left, down, right
        # x: +1 apple is to the left
        # y: +1 apple is down
        
        if x==1 and d==1:
            return True
        if x==-1 and d==3:
            return True
        if y==1 and d==2:
            return True
        if y==-1 and d==0:
            return True
        
    def get_state(self):
        # head, direction, list of tail positions, apple
        h = self.snake.head.pos
        d = self.snake.head.direction
        t = self.snake.tail.positions
        a = self.apple.pos

        state = np.zeros(neurons,dtype=int)
        
        grid = self.grid
        
        idx = 0
        for x in range(-1,2):
            for y in range(-1,2):
                i = [h[0]+x,h[1]+y]
                if (i in t or i[0]==1 or i[0]==grid or i[1]==1 or i[1]==grid):
                    state[idx] = 1
                idx+=1
        
        # is this a new game (meaning the snake has been killed)?
        if (self.reward<-0.2):
            state[idx] = 1
            
        # did the snake eat any apples recently?
        if (self.snake.tail.size>len(self.snake.tail.positions)):
            state[idx+1] = 1
        else:
            state[4] = 0
        
        state[idx+2] = d
        state[idx+3] = h[0]-a[0]
        state[idx+4] = h[1]-a[1] 
        
        # 4 is the head itself, can reuse
        state[4] = self.score
        
        return state
    
    def show_state(self):
        s = self.get_state()
        return s

    def play_action(self,a):
        self.snake.head.direction = a
    
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        # rotate left or right
        if keycode[1] == 'a':
            self.snake.rotate(0)
        if keycode[1] == 'd':
            self.snake.rotate(1)
        # testing
        if keycode[1] == 'q':
            # print out current state vector
            print(self.show_state())
        if keycode[1] == 'e':
            # manually run one game frame
            self.update()
        if keycode[1] == 'z':
            # enable/disable AI
            self.b = not self.b
        if keycode[1] == 'x':
            # change game speed
            Clock.unschedule(self.update)
            self.fps = {1:6,6:50,50:1}[self.fps]
            Clock.schedule_interval(self.update, 1.0/self.fps)
        if keycode[1] == 'c':
            # dump memory of brain
            print(self.brain.memory)
            
        return True

class Apple(Widget): 
    
    g = ObjectProperty(None)

    def add(self,x,y):
        # grid coordinates
        self.pos = (x,y)
        with self.canvas:
            # window coordinates
            gx = (x-1)*self.size[0]
            gy = (y-1)*self.size[1]
            # make apple red
            Color(1,0,0)
            self.g = Ellipse(pos=(gx,gy), size=self.size)
            # restore default color for rest
            Color(*color)

    def remove(self, *args):
        self.canvas.remove(self.g)
        self.g = ObjectProperty(None)        

class Snake(Widget):
    
    head = ObjectProperty(None)
    tail = ObjectProperty(None)

    def move(self):
        # save head position
        pos = list(self.head.position)
        # move head
        self.head.move()
        # update tail
        self.tail.update(pos)
        
    def rotate(self,left):
        d = self.head.direction
        if left:
            d-=1
        else:
            d+=1
        # to easily wrap at the edges
        wrap={-1:3,0:0,1:1,2:2,3:3,4:0}
        self.head.direction = wrap[d]

    def eat(self):
        # eat apple, increase tail size
        self.tail.size += 1  
        
    def remove(self):
        self.head.remove()
        self.tail.remove()

    def get_occupancy(self):
        return self.head.position + self.tail.positions

class SnakeHead(Widget):
    
    direction = OptionProperty(0, options=[0,1,2,3])
    x = NumericProperty(0)
    y = NumericProperty(0)
    position = ReferenceListProperty(x, y)

    g = ObjectProperty(None)

    def remove(self):
        if (self.g is not None and isinstance(self.g,ObjectProperty)==False):
            self.canvas.remove(self.g)
            self.g = ObjectProperty(None)

    def draw(self):
        with self.canvas:
            if (self.g is not None and isinstance(self.g,ObjectProperty)==False):
                self.canvas.remove(self.g)
            # draw half circle (the other half is overlapping the body
            fix = ((1,1.5),(0.5,1),(1,0.5),(1.5,1))[self.direction]
            x = (self.x-fix[0])*self.size[0]
            y = (self.y-fix[1])*self.size[1]
            self.g = Ellipse(pos=(x,y), size=self.size)
            
    def move(self):
        if self.direction == 0:
            self.position[1]+= 1
        elif self.direction == 1:
            self.position[0]-= 1
        elif self.direction == 2:
            self.position[1]-= 1
        elif self.direction == 3:
            self.position[0]+= 1
        self.draw()

class SnakeTail(Widget):
    
    size = NumericProperty(snake_size)
    positions = ListProperty()
    graphics = ListProperty()

    def remove(self):
        self.size = snake_size
        for g in self.graphics:
            self.canvas.remove(g)
        self.positions = []
        self.graphics = []
       
    def update(self, pos):
        # add new segment
        self.positions.append(pos)
        # delete the final segment (unless an apple has been eaten recently)
        if len(self.positions) > self.size:
            del(self.positions[0])
            r = True
        # same for graphics
        with self.canvas:
            x = (pos[0]-1)*self.width
            y = (pos[1]-1)*self.height
            self.graphics.append(Rectangle(pos=(x,y), size=(self.width, self.height)))
            if len(self.graphics) > self.size:
                self.canvas.remove(self.graphics.pop(0))

class SnakeApp(App):

    def build(self):
        game = Game()
        game.start()
        Clock.schedule_interval(game.update, 1.0/fps)
        return game

if __name__ == '__main__':
    SnakeApp().run()
