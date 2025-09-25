from game import PongGame
from dqn_agent import DQNAgent
import torch
import pygame


env = PongGame()
agent = DQNAgent()
agent.model.load_state_dict(torch.load("pong_ai.pt"))
agent.model.eval()
agent.epsilonActive = False

state = env.reset()
done = False

while True:
    playerAction = 0 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        playerAction = 1
    elif keys[pygame.K_DOWN]:
        playerAction = 2

    
    aiAction = agent.act(state)
    state, rw, done = env.step(aiAction, playerAction)
    env.render()



    if done:
        state = env.reset() 
    

