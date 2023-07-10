import numpy as np          
import wandb                 
from vizdoom import *
import torch
from IPython.display import clear_output
from collections import namedtuple, deque
from a2c_agent import A2CAgent
from stack_frame import preprocess_frame, stack_frame

def create_environment():
    game = vizdoom.DoomGame()
    
    # Load the correct configuration
    game.load_config("../doom files/deadly_corridor.cfg")
    game.set_doom_scenario_path("../doom files/deadly_corridor.wad")
        
    possible_actions  = np.identity(3,dtype=int).tolist()
    
    return game, possible_actions

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (60, -12, -80, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

# Define sweep config
sweep_configuration = {
    'program': 'corridor_sweep.py',
    'method': 'bayes',
    'name': 'sweep-corridor',
    'metric': {'goal': 'maximize', 'name': 'score'},
    'parameters': 
    {
        'episodes': {'values': [1000, 2000, 5000]},
        'skiprate': {'values': [4, 8, 12]},
        'update_every': {'values': [50, 100, 200, 500]},
        'gamma': {'max': 0.99, 'min': 0.9},
        'alpha': {'value': 0.1},
        'beta': {'value': 0.1},
        'epsilon': {'max': 1.0, 'min': 0.9},
        'epsilon_decay': {'max': 0.9995, 'min': 0.9},
        'epsilon_min': {'max': 0.1, 'min': 0.0001},
     }
}

def train():
    game, possible_actions = create_environment()
    
    INPUT_SHAPE = (4, 84, 84)
    ACTION_SIZE = len(possible_actions)
    SEED = 0
    actor_savefile = 'models/sweeps/corridor/actor.pth'
    critic_savefile = 'models/sweeps/corridor/critic.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    clear_output(wait=True)
    print("Starting training")
    run = wandb.init()
    
    n_episodes  =  wandb.config.episodes    # Number of episodes to train for
    skiprate    =  wandb.config.skiprate    # How many frames to skip
    update      =  wandb.config.update_every    # How often to update the network
    gamma       =  wandb.config.gamma   # Discount factor
    alpha       =  wandb.config.alpha   # Actor learning rate
    beta        =  wandb.config.beta    # Critic learning rate
    epsilon     =  wandb.config.epsilon # Exploration rate
    epsilon_decay = wandb.config.epsilon_decay  # Decay rate for exploration
    epsilon_min = wandb.config.epsilon_min      # Minimum exploration probability
    
    print("Creating agent")
    agent = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, gamma, alpha, beta, update,
                 epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    
    
    scores = []
    game.set_window_visible(False)
    game.init()
    print("Starting episodes")
    for i_episode in range(1, n_episodes+1):
        game.new_episode()
        state = stack_frames(None, game.get_state().screen_buffer.transpose(1, 2, 0), True) 
        score = 0
        while True:
            action, log_prob, entropy = agent.act(state)
            reward = game.make_action(possible_actions[action], skiprate)
            done = game.is_episode_finished()
            score += reward
            if done:
                break
            else:
                next_state = stack_frames(state, game.get_state().screen_buffer.transpose(1, 2, 0), False)
                agent.step(state, log_prob, entropy, reward, done, next_state)
                state = next_state
        scores.append(score)              # save most recent score
            
        wandb.log({"epsilon": agent.epsilon, "loss": agent.loss, "score": score})

        print('\rEpisode {}\tScore: {},\tAverage Score: {:.2f},\tEpsilon: {:.4f},\tLoss: {:.4f}'.format(i_episode, score, np.mean(scores), agent.epsilon, agent.loss), end="")
            
    game.close()
    wandb.finish()
        
    return scores

def main():
    #sweep_id = wandb.sweep(sweep=sweep_configuration, project='vizdoom-a2c-corridor')
    sweep_id = "igq7wcuy"
    wandb.agent(sweep_id, function=train, count=100)

if __name__ == "__main__":
    main()