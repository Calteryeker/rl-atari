import gymnasium as gym
import pandas
import optuna
from mycnn import MyCnn

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DQN

TOTAL_TIMESTEPS=100000
TOTAL_TIMESTEPS_FINAL=1500000
EVAL_FREQ = 500
ENV_NAME='ALE/Galaxian-v5'
GAME='galaxian'

policy_kwargs = dict(
  features_extractor_class=MyCnn,
  features_extractor_kwargs=dict(features_dim=512),
)

def train_ppo(trial : optuna.Trial, log_path, clip_rew=False):
    # chama os métodos do "trial" (tentativa) para sugerir valores para os parâmetros
    lr = trial.suggest_categorical('learning_rate', [0.0003, 0.003, 0.03, 0.3])
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    clip_range = trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3])
    
    print(f"\nTRIAL #{trial.number}: lr={lr}, n_steps={n_steps}, clip_range={clip_range}")
    
    new_logger = configure(f"{log_path}/train_ppo_{str(trial.number).zfill(3)}", ["csv", "tensorboard"])

    env_ppo = gym.make(ENV_NAME)
    env_ppo_wrapped = AtariWrapper(env=env_ppo, clip_reward=clip_rew, action_repeat_probability=0.25)
    
    """ env_eval_ppo = gym.make('ALE/DonkeyKong-v5')
    env_eval_ppo_wrapped = AtariWrapper(env=env_eval_ppo)
    env_eval_ppo_monitored = Monitor(env=env_eval_ppo_wrapped)

    eval_callback = EvalCallback(env_eval_ppo_monitored, best_model_save_path="./",
                             log_path="./logs/", eval_freq=EVAL_FREQ,
                             deterministic=True, render=False) """

    model_ppo = PPO("CnnPolicy", env_ppo_wrapped, policy_kwargs=policy_kwargs, verbose=0,
                     learning_rate=lr,
                       n_steps=n_steps,
                           clip_range=clip_range, device='cuda')
    model_ppo.set_logger(new_logger)
    model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)

    df = pandas.read_csv(f"{log_path}/train_ppo_{str(trial.number).zfill(3)}/progress.csv")
    x = sum(df["rollout/ep_rew_mean"])/len(df["rollout/ep_rew_mean"])
    return x

def final_exec(best_params, log_path, timesteps, clip_rew=False, train_dqn=True):
    lr, n_steps, clip_range = best_params.values()
    print(lr, n_steps, clip_range)
    
    if train_dqn:
      print("DQN")
      env_dqn = gym.make(ENV_NAME)
      env_dqn_wrapper = AtariWrapper(env=env_dqn, clip_reward=clip_rew, action_repeat_probability=0.25)

      new_logger_dqn = configure(log_path+"/dqn", ["csv", "tensorboard"])

      model_dqn = DQN("CnnPolicy", env_dqn_wrapper, policy_kwargs=policy_kwargs, verbose=0,
                      learning_rate=lr, device='cuda', buffer_size=300000, tensorboard_log=log_path)
      
      model_dqn.set_logger(new_logger_dqn)
      model_dqn.learn(total_timesteps=timesteps, progress_bar=True, tb_log_name="dqn_run")
      model_dqn.save(log_path+"/dqn_"+GAME)

    print("PPO")
    env_ppo = gym.make(ENV_NAME)
    env_ppo_wrapper = AtariWrapper(env=env_ppo, clip_reward=clip_rew, action_repeat_probability=0.25)

    new_logger_ppo = configure(log_path+"/ppo", ["csv", "tensorboard"])

    model_ppo = PPO("CnnPolicy", env_ppo_wrapper, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log=log_path,
                     learning_rate=lr,
                       n_steps=n_steps,
                           clip_range=clip_range, device='cuda')
    
    model_ppo.set_logger(new_logger_ppo)
    model_ppo.learn(total_timesteps=timesteps, log_interval=4, progress_bar=True, tb_log_name="ppo_run")
    model_ppo.save(log_path+"/ppo_"+GAME)

if __name__== '__main__':
  study_noclip = optuna.create_study(direction='maximize', 
                            storage='sqlite:///ppo_galaxian_noclip_100k.db', 
                            study_name= 'ppo_optimization', 
                            load_if_exists=True)
      
  #study_noclip.optimize(lambda trial : train_ppo(trial, log_path='./logs_noclip_galaxian_100k'+GAME, clip_rew=False), n_trials=24, n_jobs=6, show_progress_bar=True) 

  print("MELHORES PARÂMETROS:")
  print(study_noclip.best_params)
      
  #final_exec(best_params=study_noclip.best_params, log_path='./logs_noclip_100k_'+GAME, timesteps=TOTAL_TIMESTEPS_FINAL)

  study = optuna.create_study(direction='maximize', 
                            storage='sqlite:///ppo_galaxian_clipped_100k.db', 
                            study_name= 'ppo_optimization', 
                            load_if_exists=True)
      
  study.optimize(lambda trial : train_ppo(trial, log_path='./logs_clipped_100k_'+GAME, clip_rew=True), n_trials=24, n_jobs=6, show_progress_bar=True)

  print("MELHORES PARÂMETROS:")
  print(study.best_params)

  final_exec(best_params=study.best_params, log_path="./logs_clipped_galaxian_100k_"+GAME, timesteps=TOTAL_TIMESTEPS_FINAL, clip_rew=True)