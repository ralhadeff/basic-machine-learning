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
fps = 80
# default paint color
color = (0,0,0)

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
    grid = 12
    # score (apples eaten)
    score = NumericProperty(0)
    # iteration (game number played)
    iteration = NumericProperty(0)
    average = NumericProperty(0)
    total = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # keyboard binding for testing purposes
        self._keyboard = Window.request_keyboard(self._on_keyboard_down, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        # trick to overcome initial sizing problem
        self.turn_count = 2
        self.brain = Qlearn([54,64,3],10000,0.9,3)

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
        # reset score, increase game count
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
            return
        else:
            state = self.get_state()
            # reward from previous action, after the action
            action = self.brain.update(self.reward,state)
            self.play_action(action)
            if (self.reward<0):
                self.reward=0
            # snake is facing apple
            if self.right_direction():
                self.reward+= 2
            else:
                self.reward+= 1
            # move snake
            self.snake.move()
            # check defeat
            if self.check_defeat():
                self.reset()
                self.reward=-250
            # check eat
            if (self.snake.head.position == self.apple.pos):
                self.reward+=25
                self.score+= 1
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

        state = np.zeros(54,dtype=int)
        
        # head position
        x,y = h
        # radius
        r = 2
        gap = (r+1)**2
        # counter
        c = 0
        for i in range(-2,2+1):
            for j in range(-2,2+1):
                if [x+i,y+j] in t:
                    state[c]=1
                    c+=1
                if ([x+i,y+j]==a):
                    state[c+gap]=1          
        
        # borders
        if x<=2:
            state[[0,5,10,15,20]]=1
            if (x==1):
                state[[1,6,11,16,21]]=1
        if x>=self.grid-1:
            state[[4,9,14,19,24]]=1 
            if x==self.grid:
                state[[3,8,13,18,23]]=1 
        if y<=2:
            state[[0,1,2,3,4]]=1
            if y==1:
                state[[5,6,7,8,9]]=1
        if y>=self.grid-1:
            state[[20,21,22,23,24]]=1 
            if y==self.grid:
                state[[15,16,17,18,19]]=1
        
        state[2*gap] = np.sign(h[0]-a[0])
        state[2*gap+1] = np.sign(h[1]-a[1])
        
        if (d==0):
            state[2*gap+2] = 1
        elif (d==2):
            state[2*gap+2] = -1
        if (d==1):
            state[2*gap+3] = 1
        elif (d==3):
            state[2*gap+3] = -1            
       
        return state
    
    def show_state(self):
        s = self.get_state()
        print('snake:')
        print(s[[3,2,1]])
        print([s[4],0,s[0]])
        print(s[[5,6,7]])
        i = 8
        print('apple:')
        print(s[[i+3,i+2,i+1]])
        print([s[i+4],0,s[i+0]])
        print(s[[i+5,i+6,i+7]])              

    def play_action(self,a):
        # 0 left, 1 right, 2 nothing
        if (a<2):
            self.snake.rotate(a)
    
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        # rotate left or right
        if keycode[1] == 'a':
            self.snake.rotate(0)
        if keycode[1] == 'd':
            self.snake.rotate(1)
        # testing
        if keycode[1] == 'q':
            print(self.show_state())
        if keycode[1] == 'e':
            self.update()
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