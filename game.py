import pygame
import random
import numpy as np

HIT_BALL_REWARD = 1.0
PROX_REWARD_MULTIPLIER = 0.1
MOVEMENT_PENALTY = 0.001
LOSING_PENALTY = 1.0
WINNING_REWARD = 2.0

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PADDLE_SPEED = 10

class PongGame:
    def __init__(
        self,
        hit_ball_reward: float = HIT_BALL_REWARD,
        prox_reward_multiplier: float = PROX_REWARD_MULTIPLIER,
        movement_penalty: float = MOVEMENT_PENALTY,
        losing_penalty: float = LOSING_PENALTY,
        winning_reward: float = WINNING_REWARD,
        ai_paddle_speed: int = PADDLE_SPEED,
        player_paddle_speed: int = PADDLE_SPEED,
    ):
        pygame.init()
        self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("VEXX PONG")
        self.clock = pygame.time.Clock()

        # Configurable dynamics and rewards
        self.hit_ball_reward = hit_ball_reward
        self.prox_reward_multiplier = prox_reward_multiplier
        self.movement_penalty = movement_penalty
        self.losing_penalty = losing_penalty
        self.winning_reward = winning_reward
        self.ai_paddle_speed = ai_paddle_speed
        self.player_paddle_speed = player_paddle_speed

        self.reset()

    def reset(self):
        self.ball = pygame.Rect(SCREEN_WIDTH//2, SCREEN_HEIGHT//2, 12, 12)
        self.player = pygame.Rect(SCREEN_WIDTH - 20, SCREEN_HEIGHT//2, 12, 60)
        self.ai = pygame.Rect(20, SCREEN_HEIGHT//2, 12, 60)
        self.ballVel = [4, random.choice([-6, 6])]
        self.playerVel = 0
        self.score = {"player": 0, "ai": 0}
        return self.get_state()
    
    def get_state(self):
        ballCenterX = self.ball.x + self.ball.width / 2
        ballCenterY = self.ball.y + self.ball.height / 2
        aiCenterY = self.ai.y + self.ai.height / 2

        return np.array([
            ballCenterX / SCREEN_WIDTH,
            ballCenterY / SCREEN_HEIGHT,
            self.ballVel[0] / 10,
            self.ballVel[1] / 10,
            aiCenterY / SCREEN_HEIGHT,
            (ballCenterY - aiCenterY) / SCREEN_HEIGHT,  # Vertical distance to ball
            (ballCenterX - self.ai.x) / SCREEN_WIDTH
        ], dtype=np.float32)
    

    def step(self, actionAi, action_player): # action = 0 (stay), 1 (up), 2 (down)
        # Temp
        if (self.ballVel[0] >= 0):
            self.ballVel[0] = 6
        else:
            self.ballVel[0] = -6
        # Store previous positions for reward calculation
        prevBallX = self.ball.x
        prevAiCenter = self.ai.y + self.ai.height // 2
        prevBallCenter = self.ball.y + self.ball.height // 2
        # Movement + Boundaries
        if actionAi == 1:
            self.ai.y -= self.ai_paddle_speed
        elif actionAi == 2:
            self.ai.y += self.ai_paddle_speed
        self.ai.y = max(0, min(SCREEN_HEIGHT - 60, self.ai.y))

        if action_player == 1:
            self.player.y -= self.player_paddle_speed
        elif action_player == 2:
            self.player.y += self.player_paddle_speed
        self.player.y = max(0, min(SCREEN_HEIGHT - 60, self.player.y))

        # Ball movement
        self.ball.x += self.ballVel[0]
        self.ball.y += self.ballVel[1]
        if self.ball.top <= 0 or self.ball.bottom >= SCREEN_HEIGHT:
            self.ballVel[1] *= -1

        reward = 0.0
        if self.ball.colliderect(self.ai):
            # Classic Pong physics: X velocity stays constant, only Y changes based on hit position
            self.ballVel[0] = -6 if self.ballVel[0] > 0 else 6  # Keep X speed constant at 4
            # Where on the paddle the ball hit (-1 to 1)
            relativeIntersectY = (self.ai.y + self.ai.height / 2) - (self.ball.y + self.ball.height / 2)
            normalizedIntersectY = relativeIntersectY / (self.ai.height / 2)
            # Prevent extreme angles -1 -> 1
            normalizedIntersectY = max(-1, min(1, normalizedIntersectY))
            self.ballVel[1] = normalizedIntersectY * 6 # Max y velocity
            self.ball.x = self.ai.x + self.ai.width # prevent "sticking"
            reward += self.hit_ball_reward
        elif self.ball.colliderect(self.player):
            # Classic Pong physics: X velocity stays constant, only Y changes based on hit position
            self.ballVel[0] = -6 if self.ballVel[0] > 0 else 6  # Keep X speed constant at 4
            relativeIntersectY = (self.player.y + self.player.height / 2) - (self.ball.y + self.ball.height / 2)
            normalizedIntersectY = relativeIntersectY / (self.player.height / 2)
            normalizedIntersectY = max(-1, min(1, normalizedIntersectY))
            self.ballVel[1] = normalizedIntersectY * 6
            self.ball.x = self.player.x - self.player.width # prevent "sticking"
        
        # Reward for reducing vertical distance to the ball ONLY when it's moving toward the AI
        afterAiCenter = self.ai.y + self.ai.height // 2
        ballCenter = self.ball.y + self.ball.height // 2
        prevDistance = abs(prevBallCenter - prevAiCenter)
        distanceToBall = abs(ballCenter - afterAiCenter)
        maxDistance = SCREEN_HEIGHT
        if self.ballVel[0] < 0:  # Ball traveling toward AI (to the left)
            distanceImprovement = max(0.0, (prevDistance - distanceToBall)) / maxDistance
            reward += self.prox_reward_multiplier * distanceImprovement

        # Small Penalty for unnecessary movement (to encourage efficiency)
        if actionAi != 0:
            reward -= self.movement_penalty

        # Scoring
        done = False
        if self.ball.left <= 0:
            self.score["player"] += 1
            reward -= self.losing_penalty
            done = True
        elif self.ball.right >= SCREEN_WIDTH:
            self.score["ai"] += 1
            reward += self.winning_reward
            done = True

        return self.get_state(), reward, done

    def render(self):
        self.display.fill(BLACK)
        pygame.draw.rect(self.display, WHITE, self.player)
        pygame.draw.rect(self.display, WHITE, self.ai)
        pygame.draw.ellipse(self.display, WHITE, self.ball)
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
