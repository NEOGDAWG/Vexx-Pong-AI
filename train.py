from game import PongGame
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import torch
import random
import time
from statistics import mean

# Configure environment and training curriculum
env = PongGame(
    hit_ball_reward=0.2,
    prox_reward_multiplier=0.05,
    movement_penalty=0.001,
    losing_penalty=1.0,
    winning_reward=1.0,
    ai_paddle_speed=10,
    player_paddle_speed=8,
)
agent = DQNAgent()

# Opponent curriculum (smooth, per-step)
TRAINING_BOT_RANDOMNESS_START = 0.30
TRAINING_BOT_RANDOMNESS_END = 0.05
TRAINING_BOT_SPEED_START = 8.0
TRAINING_BOT_SPEED_END = 10.0
CURRICULUM_STEPS = 300_000  # steps to reach end values

MIN_DISTANCE_FROM_BALL_TO_MOVE = 20
TARGET_UPDATE_FREQ_STEPS = 2000
TRAIN_FREQ = 4
MAX_EPISODE_STEPS = 1000

# Evaluation configuration
EVAL_EPISODES = 200
EVAL_BOT_RANDOMNESS = 0.10
EVAL_BOT_SPEED = 10.0

# Keep epsilon decay reasonably slow to ensure exploration (per-step decay)
agent.epsilonDecay = 0.9995

episode_rewards = []
episodes = 3000
stepCount = 0

train_start_time = time.time()

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0.0
    episodeSteps = 0

    while not done:
        stepCount += 1
        episodeSteps += 1

        aiAction = agent.act(state)

        # Smooth curriculum: interpolate randomness and speed based on total steps
        progress = min(1.0, stepCount / CURRICULUM_STEPS)
        TRAINING_BOT_RANDOMNESS = (
            TRAINING_BOT_RANDOMNESS_START * (1.0 - progress)
            + TRAINING_BOT_RANDOMNESS_END * progress
        )
        TRAINING_BOT_SPEED = (
            TRAINING_BOT_SPEED_START
            + (TRAINING_BOT_SPEED_END - TRAINING_BOT_SPEED_START) * progress
        )
        # Apply opponent speed to environment
        env.player_paddle_speed = TRAINING_BOT_SPEED

        # Training Bot with some randomness
        trainingBotAction = 0
        if random.random() < TRAINING_BOT_RANDOMNESS:
            trainingBotAction = random.randint(0, 2)
        else:
            # Imperfect following with delay/error
            ballCenter = env.ball.y + env.ball.height // 2
            trainingBotCenter = env.player.y + env.player.height // 2
            diff = ballCenter - trainingBotCenter

            if (
                abs(diff) > MIN_DISTANCE_FROM_BALL_TO_MOVE
            ):  # Only move if ball is far from paddle center
                if diff > 0:
                    trainingBotAction = 2
                elif diff < 0:
                    trainingBotAction = 1

        nextState, reward, done = env.step(aiAction, trainingBotAction)
        agent.memorize(state, aiAction, reward, nextState, done)

        # Train every TRAIN_FREQ steps
        if stepCount % TRAIN_FREQ == 0:
            agent.trainStep()

        # Update target network on a step schedule for smoother updates
        if stepCount % TARGET_UPDATE_FREQ_STEPS == 0:
            agent.updateTarget()
            print(f"Target network updated at step {stepCount}")

        # Per-step epsilon decay
        agent.epsilon = max(agent.epsilonMin, agent.epsilon * agent.epsilonDecay)

        state = nextState
        score += reward
        # env.render() # Uncomment to render game during training (slow)

        # Prevent episodes from running too long
        if episodeSteps > MAX_EPISODE_STEPS:
            done = True

    # (moved target update and epsilon decay into step loop)

    episode_rewards.append(score)
    print(
        f"Episode {episode+1}/{episodes} | Score: {score:.3f} | "
        f"Epsilon: {agent.epsilon:.3f} | BotRnd: {TRAINING_BOT_RANDOMNESS:.3f} | "
        f"BotSpd: {TRAINING_BOT_SPEED:.3f}"
    )

train_end_time = time.time()
train_minutes = (train_end_time - train_start_time) / 60.0

torch.save(agent.model.state_dict(), "pong_ai.pt")

# Compute early vs late reward improvements
window = min(100, len(episode_rewards))
early_avg = mean(episode_rewards[:window])
late_avg = mean(episode_rewards[-window:])
improvement_factor = late_avg / early_avg if early_avg != 0 else float("inf")

print("\n=== Training Summary ===")
print(f"Total environment steps: {stepCount}")
print(f"Training time: {train_minutes:.2f} minutes")
print(
    f"Average episode reward (first {window} episodes): {early_avg:.3f}\n"
    f"Average episode reward (last {window} episodes): {late_avg:.3f}\n"
    f"Improvement factor: {improvement_factor:.2f}x"
)

# Plot training curve
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Shaped Reward")
plt.title("AI Training Progress")
plt.show()

# Evaluation: greedy policy vs fixed opponent
print("\n=== Evaluation (Greedy Policy vs Scripted Opponent) ===")
agent.epsilonActive = False  # disable exploration during evaluation

wins = 0
losses = 0
total_margin = 0

for eval_ep in range(EVAL_EPISODES):
    state = env.reset()
    done = False

    # Fixed evaluation opponent parameters
    env.player_paddle_speed = EVAL_BOT_SPEED

    while not done:
        aiAction = agent.act(state)

        # Scripted opponent: same tracking logic but with fixed randomness/speed
        trainingBotAction = 0
        if random.random() < EVAL_BOT_RANDOMNESS:
            trainingBotAction = random.randint(0, 2)
        else:
            ballCenter = env.ball.y + env.ball.height // 2
            trainingBotCenter = env.player.y + env.player.height // 2
            diff = ballCenter - trainingBotCenter

            if abs(diff) > MIN_DISTANCE_FROM_BALL_TO_MOVE:
                if diff > 0:
                    trainingBotAction = 2
                elif diff < 0:
                    trainingBotAction = 1

        state, _, done = env.step(aiAction, trainingBotAction)

    # After each evaluation episode, env.score reflects who scored the point
    margin = env.score["ai"] - env.score["player"]
    total_margin += margin
    if margin > 0:
        wins += 1
    else:
        losses += 1

win_rate = wins / EVAL_EPISODES
avg_margin = total_margin / EVAL_EPISODES

print(f"Evaluation episodes: {EVAL_EPISODES}")
print(f"Win rate vs scripted opponent: {win_rate*100:.1f}%")
print(f"Average score margin (ai - opponent): {avg_margin:.3f}")

env.close()