""" This provides a convenient way of running the different implementations in the command line """

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script runs a few basic algorithms demonstrating the RL"
                                                 " Policy Gradient approach")
    parser.add_argument("method", choices=["cross_entropy", "hill_climbing", "ddpg"], help="The Policy Gradient method "
                                                                                           "supported")
    parser.add_argument("--episodes", default=1000, help="Maximum number of episodes to train an agent")
    parser.add_argument("--timesteps", default=700, help="Maximum number of time steps to take per episode")
    parser.add_argument("--plot", choices=["true", "false"], help="Flag indicating whether to plot the scores or not")
    args = parser.parse_args()

    num_episodes = args.episodes
    max_time_steps = args.timesteps
    method = args.method
    plot = args.plot == "true"
    print("Running the {} method. Max number of episodes {}, max number of time steps {}".format(method, num_episodes,
          max_time_steps))

    if method == "cross_entropy":
        from crossentropy import run_cross_entropy_cmd
        run_cross_entropy_cmd(num_episodes=num_episodes, max_time_steps=max_time_steps, plot_when_done=plot)
    elif method == "hill_climbing":
        from hillclimbing import run_hill_climbing_cmd
        run_hill_climbing_cmd(num_episodes=num_episodes, max_time_steps=max_time_steps, plot_when_done=plot)
    elif method == "ddpg":
        from ddpg_agent import run_ddpg_cmd
        run_ddpg_cmd(num_episodes=num_episodes, max_time_steps=max_time_steps, plot_when_done=plot)
    else:
        # this should not happen as it should be taken care above when adding the argument - just in case
        raise RuntimeError("Unsupported method {} provided", method)
