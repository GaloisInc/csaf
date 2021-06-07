import random
import numpy as np
import time
import matplotlib.pyplot as pl 

from PIL import Image


# Compute a single rollout.
#
# env: Environment
# policy: Policy
# render: bool
def simulate(env, policy, init_state, n_steps, render):
    # Step 1: Initialization
    env.reset()
    state = init_state
    done = False
    policy.reset(init_state)
    total_goal_error = 0.0
    total_safe_error = 0.0

    # Step 2: Compute rollout
    for i in range(n_steps):
        # Step 2a: Render environment
        if render:
            env.render(state)
            time.sleep(0.01)

        # Step 2b: Action
        action = policy.get_action(state)
        #if render:
        #    print("Action: ", action)

        if len(action) == 0:
            done = True
            break
 
        # Step 2c: Transition environment
        next_state, safe_error, goal_error, done = env.step(state, action)
        total_safe_error += safe_error
        #print("State: ", next_state)
        #if render:
        #    print("Safe error: ", safe_error, " Goal error: ", goal_error)
 
        # Step 2d: Update state
        state = next_state

        if done:
            break

    # Step 3: Render final state
    if render:
        env.render(state)
        time.sleep(2)

    total_goal_error = env.check_goal(state)
    return total_safe_error, total_goal_error


def simulate_from_states(env, states, render):
    # Step 1: Initialization
    env.reset()
    
    for state,_ in states:
        # Step 2a: Render environment
        if render:
            #env.set_act_for_render(action)
            env.render(state)
            time.sleep(0.01)

        safe_error = env.check_safe(state)
        goal_error = env.check_goal(state)
        
        print("Safe error: ", safe_error, " Goal error: ", goal_error)
        

    time.sleep(2)

def simulate_from_states1(env, states, render):
    # Step 1: Initialization
    env.reset()
    
    for state in states:
        # Step 2a: Render environment
        if render:
            #env.set_act_for_render(action)
            env.render(state)
            time.sleep(0.01)

        safe_error = env.check_safe(state)
        goal_error = env.check_goal(state)
        
        print("Safe error: ", safe_error, " Goal error: ", goal_error)
        

    time.sleep(2)

def simulate_from_states_gif(env, states, render, outfile):
    # Step 1: Initialization
    env.reset()
    frames = []
    for state,action in states:
        # Step 2a: Render environment
        if render:
            env.set_act_for_render(action)
            frames.append(Image.fromarray(env.render(state, mode='rgb_array')))
            #time.sleep(0.01)

        safe_error = env.check_safe(state)
        goal_error = env.check_goal(state)
        
        print("Safe error: ", safe_error, " Goal error: ", goal_error)
        

    with open(outfile, 'wb') as f:
        im = Image.new('RGB', frames[0].size)
        im.save(f, save_all = True, append_images=frames, duration = 20, loop=0)

def simulate_from_file(env, filename):
    file = open(filename)
    states = []
    for l in file.readlines():
        states.append(np.array(eval(l)))
    print(len(states))

    simulate_from_states1(env, states, True)

def plot_traj(env, policy, init_state):
    states, safe_err, goal_err = get_traj_from_policy(env, policy, init_state)
    print("safe_err: ", safe_err)
    print("goal_err: ", goal_err)
    env.plot_init(states[-1])
    env.plot_states(states)
    pl.tight_layout()
    pl.show()
    pl.close()

def get_traj_from_policy(env, policy, init_state, n_steps = 1000):
    states = []

    env.reset()
    state = init_state
    done = False
    policy.reset(init_state)
    total_goal_error = 0.0
    total_safe_error = 0.0
    

    for i in range(n_steps):
        
        
        action = policy.get_action(state)
        states.append((state, action))

        if len(action) == 0:
            done = True
            break
 
        next_state, safe_error, goal_error, done = env.step(state, action)
        total_safe_error += safe_error
       
        state = next_state
        if done:
            break

    
    states.append((state, []))
    total_goal_error = env.check_goal(state)

    return states, total_safe_error, total_goal_error
