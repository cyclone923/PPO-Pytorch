env_setting = {"LunarLander-v2":
                   {
                    "solved_reward": 230,
                    "max_timesteps": 3000,
                    "update_timestep": 2000,
                    "ppo_k_updates": 4
                    },
                "BipedalWalker-v2":
                   {
                    "solved_reward": 285,
                    "max_timesteps": 3000,
                    "update_timestep": 4000,
                    "ppo_k_updates": 20
                    }
               }

def get_env_setting(env_name):
    return env_setting[env_name]