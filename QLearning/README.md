# Q-learning

_work in progress_

### Reinforcement learning through a Kivy snake game

The game is played on a 2D grid with a discretely-moving snake (in black) that can face N/S/W/E; picking an apple (red circle) gives the snake points but increases its length by one tail segment.  
If the snake hits the edges or one of its own tail segments it dies and the game is lost (and a new game restarts).  

The AI is learning to play using a simple DQN, being rewarded ...<*to be updated*>.  

**In progress** I will add a youtube movie of the training process

For testing purposes, the key bindings are:
`a` and `d` - rotate left and right
`q` - print state on screen
`e` - run one frame
`z` - disable AI, to play manually. Otherwise, the AI override the game completely.
`x` - change game speed (1 fps, 10 fps, 50 fps)
`c` - print entire memory of the network to the terminal
