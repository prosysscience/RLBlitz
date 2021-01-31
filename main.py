import gym

from models.ActorCritic import ActorCritic

if __name__ == '__main__':
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    ActorCritic(state_dim, action_dim)
