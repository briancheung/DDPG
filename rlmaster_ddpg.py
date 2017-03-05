import filter_env
from ddpg import *
import gc
from envs.mujoco_envs import move_single_env as msenv
gc.enable()

ENV_NAME = 'rlmaster'
EPISODES = 100000
TEST = 10

def main():
    #env = filter_env.makeFilteredEnv(msenv.get_environment())
    env = msenv.get_environment()
    
    agent = DDPG(env)
    env = gym.wrappers.Monitor(env, 'experiments/' + ENV_NAME, force=True)

    for episode in xrange(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(env.spec.timestep_limit):
					#env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
    env.monitor.close()

if __name__ == '__main__':
    main()
