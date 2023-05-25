"""
Class: Logger
created by: Daniel Mauricio Fernandez Gonzalez
e-mail: dafer21@student.sdu.dk
date: 11 April 2023

Logger
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import os
from os import path
import pickle
import matplotlib as mpl
import json
from .videoSaver import VideoSaver

PLOT_IN_OTHER_FILE = False

if not PLOT_IN_OTHER_FILE:
    mpl.use('qt5agg')
root_run_folder_name = "runs"


class VideoSettings:
    height = 480
    width = 640
    fps = 30
    n_frames = 100
    filename = "video"


class DataStore:
    def __init__(self):
        self.record = -99999
        self.weights = None
        self.weights_post = None
        self.noise = None
        self.iteration = 0
        self.index = 0
        self.time = 0.

    def store_weights(self, new_record, weights, noise, iteration, time, index):
        refreshed = False

        if new_record > self.record:
            self.weights = weights
            self.noise = noise.detach().clone()
            self.iteration = iteration
            refreshed = True
            self.record = new_record
            self.index = index
            self.time = time

        return refreshed

    def store_weights_post(self, weights_post):
        self.weights_post = weights_post

        return False


class Logger:
    video_settings: VideoSettings

    def __init__(self, save=False, frequency=20, frequency_plot=5, robot="", nn_config=None, PIBB_param=None,
                 test_value=False, size_figure=4, video_frequency=0, video_settings=None, show_PPO_graph=False):

        if video_frequency != 0 and video_settings is None:
            raise Exception("Set up the settings for recording the video")

        self.save_data = save
        self.nn_config = nn_config
        self.PIBB_param = PIBB_param
        self.test_value = test_value
        self.size_figure = size_figure
        self.filename = None

        self.video_settings = video_settings
        self.video_frequency = video_frequency
        self.video_saver = None if video_frequency == 0 else [VideoSaver(self.video_settings.fps,
                                                                         self.video_settings.n_frames,
                                                                         self.video_settings.height,
                                                                         self.video_settings.width,
                                                                         self.video_settings.filename)]
        self.record_in_progress = False
        self.n_video_saver = 0 if self.video_saver is None else 1
        self.all_buffer_video_full = False

        if not path.exists(root_run_folder_name):
            os.mkdir(root_run_folder_name)

        now = datetime.now()
        self.folder_time = now.strftime("%Y_%m_%d.%H_%M_%S")
        self.folder = os.path.join(root_run_folder_name, self.folder_time)

        self.frequency = frequency
        self.frequency_plot = frequency_plot
        self.curriculum = None
        self.algorithm_parameters = None

        self.stored_info = {
            "max_reward": DataStore(),
            "min_reward": DataStore(),
            "mean_reward": DataStore(),

            "max_distance": DataStore(),
            "min_distance": DataStore(),
            "mean_distance": DataStore(),
        }
        self.renew_data = {
            "max_reward": False,
            "min_reward": False,
            "mean_reward": False,

            "max_distance": False,
            "min_distance": False,
            "mean_distance": False,
        }

        self.soft_reward = []
        self.x_axis = []
        self.mean_distances = []
        self.mean_reward = []
        self.mean_std_height = []
        self.falling_robots = []

        self.x_axis_plot = []
        self.mean_distances_plot = []
        self.mean_reward_plot = []
        self.soft_reward_plot = []
        self.falling_robots_plot = []
        self.iteration = 0

        if not PLOT_IN_OTHER_FILE:
            self.figure, self.ax = plt.subplots(2, 2)
        else:
            self.figure = None
            self.ax = None

        self.figure_PPO_param = None
        self.ax_PPO_param = None

        self.gamma = []
        self.learning_rates = []
        self.surrogate_loss = []
        self.value_loss = []
        self.entropy = []
        self.student_loss = []
        self.terrain_curriculum = []
        self.loss_mean = []
        self.time_steps_PPO = []
        self.show_PPO_graph = not self.test_value and show_PPO_graph
        
        if self.show_PPO_graph:
            self.figure_PPO_param, self.ax_PPO_param = plt.subplots(4, 2)

        if not PLOT_IN_OTHER_FILE:
            plt.ion()

        self.distance = []
        self.time = []

        if robot != "":
            self.set_robot_name(robot)

    def is_recording_in_progress(self):
        return self.record_in_progress

    def load_multiple_video_recoder(self, video_settings, video_frequency):
        self.video_settings = video_settings
        self.video_frequency = video_frequency

        self.n_video_saver = len(video_settings)

        self.video_saver = []

        for setting in video_settings:
            setting: VideoSettings

            self.video_saver.append(
                VideoSaver(
                    setting.fps,
                    setting.n_frames,
                    setting.height,
                    setting.width,
                    os.path.join(self.folder, setting.filename)
                )
            )

            if self.save_data:
                print(f"Videos will be saved as: {os.path.join(self.folder, setting.filename)}_x.mp4")

    def _start_video_record_(self):
        self.record_in_progress = True
        self.all_buffer_video_full = False

        for video_saver in self.video_saver:
            video_saver.start_record()

    def store_and_save_video(self, frames):
        if self.record_in_progress:
            self.store_frames(frames)

            if self.all_buffer_video_full:
                self.save_videos()

    def store_frames(self, frames):
        self.all_buffer_video_full = True

        max_n_frames = self.n_video_saver if len(frames) > self.n_video_saver else len(frames)

        for n_item in range(max_n_frames):
            self.video_saver[n_item].store_frame(frames[n_item])
            self.all_buffer_video_full = self.all_buffer_video_full and self.video_saver[n_item].buffer_full

    def save_videos(self):
        print("Saving videos ...")

        for video_saver in self.video_saver:
            filename = video_saver.filename + "_" + str(self.iteration)
            video_saver.save_video(filename, self.save_data)

        self.all_buffer_video_full = False
        self.record_in_progress = False

    def recover_nn_information(self, filename=None):
        filename = self.filename if filename is None else filename
        self.folder = os.path.dirname(filename)
        nn_info_file = os.path.join(self.folder, "nn_config.json")

        with open(nn_info_file, "r") as f:
            nn_info = json.load(f)

        return nn_info

    def recover_curriculum(self, filename):
        self.folder = os.path.dirname(filename)

        with open(os.path.join(self.folder, "curriculum.pickle"), 'rb') as f:
            curriculum = pickle.load(f)

        return curriculum

    def recover_algorithm_parameters(self, filename):
        self.folder = os.path.dirname(filename)

        with open(os.path.join(self.folder, "algorithm_parameters.pickle"), 'rb') as f:
            algorithm_param = pickle.load(f)

        return algorithm_param

    def recover_data_class(self, filename):
        self.filename = filename

        with open(filename, "rb") as f:
            data = pickle.load(f)

        return data

    def recover_data(self, filename, post=True):
        if self.filename is None:
            self.filename = filename

        with open(filename, "rb") as f:
            data = pickle.load(f)

        return data.weights_post if post else data.weights, data.noise, data.iteration, data.record, data.index

    def set_robot_name(self, robot):
        if robot is not None:
            path_to_check = os.path.join(root_run_folder_name, robot)

            if self.save_data:
                if not path.exists(path_to_check):
                    os.mkdir(path_to_check)

                self.folder = os.path.join(path_to_check, self.folder_time)
                
                if self.video_saver is not None:
                    self._change_path_videos(self.folder)
                
    def _change_path_videos(self, new_path):
        if type(self.video_saver) is list:

            for videSaver in self.video_saver:
                videSaver.filename = os.path.join(new_path, os.path.basename(videSaver.filename))

    def save_points_testing(self, distance, time):
        self.distance.append(float(torch.mean(distance)))
        self.time.append(time)

    def create_folder(self):

        if self.save_data:
            os.mkdir(self.folder)

    def plot_in_grap(self, plot_full_data=False):
        if self.test_value:

            xpoints = np.array(self.time)
            ypoints = np.array(self.distance)

            self.ax.set_title("Distance vs time")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Distance (m)")

            self.ax.plot(xpoints, ypoints)
        else:

            plot_x_value = self.x_axis if plot_full_data else self.x_axis_plot
            plot_soft_reward_value = self.soft_reward if plot_full_data else self.soft_reward_plot
            plot_mean_distances_value = self.mean_distances if plot_full_data else self.mean_distances_plot
            plot_mean_reward_value = self.mean_reward if plot_full_data else self.mean_reward_plot
            plot_fallen_value = self.falling_robots if plot_full_data else self.falling_robots_plot

            xpoints = np.array(plot_x_value)
            ypoints = np.array([r * 10 for r in plot_mean_reward_value])

            self.ax[0][0].clear()

            self.ax[0][0].set_title("Avg. reward vs iteration")
            # self.ax[0].set_xlabel("Iteration")
            self.ax[0][0].set_ylabel("Reward")

            self.ax[0][0].plot(xpoints, ypoints, 'b', label='instant')
            self.ax[0][0].plot(xpoints, np.array([r * 10 for r in plot_soft_reward_value]), 'r', label='mean')
            self.ax[0][0].legend()

            ypoints = np.array(plot_mean_distances_value)
            self.ax[1][0].clear()

            self.ax[1][0].set_title("Avg. distance (m) vs iteration")
            self.ax[1][0].set_xlabel("Iteration")
            self.ax[1][0].set_ylabel("Distance (m)")
            self.ax[1][0].plot(xpoints, ypoints)

            ypoints = np.array(plot_fallen_value)
            self.ax[0][1].clear()

            self.ax[0][1].set_title("N fallen robots vs iteration")
            self.ax[0][1].set_xlabel("Iteration")
            self.ax[0][1].set_ylabel("n robots")
            self.ax[0][1].plot(xpoints, ypoints)

            ypoints = np.array(self.dof_pos)
            self.ax[1][1].clear()

            self.ax[1][1].set_xlabel("timestep")
            self.ax[1][1].set_ylabel("position (rad)")
            self.ax[1][1].plot(ypoints)
            ypoints = np.array(self.dof_desired)
            self.ax[1][1].plot(ypoints, linestyle='dashed', alpha=0.5)

    def plot_PPO_progres(self):
        if not(self.ax_PPO_param is None):
            def plot_info(info, row, column, x_points, y_label, x_label, title):
                ypoints = np.array(info)
                self.ax_PPO_param[row][column].clear()
    
                # self.ax_PPO_param[row][column].set_title(title)
                self.ax_PPO_param[row][column].set_xlabel(x_label)
                self.ax_PPO_param[row][column].set_ylabel(y_label)

                self.ax_PPO_param[row][column].plot(x_points, ypoints)

            xpoints = np.array(self.time_steps_PPO)

            plot_info(self.learning_rates, 0, 0, xpoints, "Learning rate", "Iteration", "Learning rate vs iteration")
            plot_info(self.surrogate_loss, 0, 1, xpoints, "Surrogate loss", "Iteration", "Avg. surrogate loss vs iteration")
            
            plot_info(self.value_loss, 1, 0, xpoints, "value loss", "Iteration", "Avg. value loss vs iteration")
            plot_info(self.entropy, 1, 1, xpoints, "Entropy", "Iteration", "Entropy vs iteration")

            plot_info(self.student_loss, 2, 0, xpoints, "Student loss", "Iteration", "Student loss vs iteration")
            plot_info(self.gamma, 2, 1, xpoints, "gamma", "Iteration", "Gamma vs iteration")

            plot_info(self.terrain_curriculum, 3, 0, xpoints, "Terrain difficulty", "Iteration", "Terrain difficulty vs iteration")
            plot_info(self.loss_mean, 3, 1, xpoints, "loss_AC", "Iteration", "Loss AC vs iteration")

    def __save_datapoints__(self):
        if self.test_value:
            dic = {
                "time": self.time,
                "distance": self.distance
            }

            filename = "testing_graph_data.json"
        else:
            dic = {
                "iteration": self.x_axis,
                "mean_distance": self.mean_distances,
                "mean_reward": self.mean_reward,
            }

            filename = "learning_graph_data.json"

            if self.show_PPO_graph:
                ppo_graph_data = {
                    "gamma": self.gamma,
                    "learning_rates": self.learning_rates,
                    "surrogate_loss": self.surrogate_loss,
                    "value_loss": self.value_loss,
                    "entropy": self.entropy,
                    "student_loss": self.student_loss,
                    "terrain_curriculum": self.terrain_curriculum,
                    "loss_mean": self.loss_mean,
                    "time_steps_PPO": self.time_steps_PPO,
                }

                filename_ppo = "learning_graph_data_ppo.json"
                filename_ppo = os.path.join(self.folder, filename_ppo)

                with open(filename_ppo, "w") as f:
                    json.dump(ppo_graph_data, f, indent=2)

        file_graph_data = os.path.join(self.folder, filename)

        with open(file_graph_data, "w") as f:
            json.dump(dic, f, indent=2)

    def plot_learning(self, iteration):
        if (iteration % self.frequency_plot) == 0:
            self.plot_log(False, False)

    def plot_log(self, save=True, block=True, plot_file_name="", save_datapoint=False):

        self.plot_in_grap(save)
        self.figure.set_size_inches(18.5, 10.5)
        ppo_graph_on = self.show_PPO_graph and len(self.time_steps_PPO) != 0

        if ppo_graph_on:
            self.plot_PPO_progres()
            self.figure_PPO_param.set_size_inches(18.5, 10.5)

        if save:
            self.figure.savefig(os.path.join(self.folder, plot_file_name + ".png"), dpi=100)

            if ppo_graph_on:
                self.figure_PPO_param.savefig(os.path.join(self.folder, "PPO_param" + ".png"), dpi=100)

        if save_datapoint:
            self.__save_datapoints__()

        plt.pause(0.001)

        if not ppo_graph_on:
            plt.figure(self.figure.number)
        
        plt.show(block=block)
        plt.pause(0.001)

    def new_interval_plot(self, remove_indices):
        self.x_axis_plot = [i for j, i in enumerate(self.x_axis_plot) if j not in remove_indices]
        self.mean_reward_plot = [i for j, i in enumerate(self.mean_reward_plot) if j not in remove_indices]
        self.soft_reward_plot = [i for j, i in enumerate(self.soft_reward_plot) if j not in remove_indices]
        self.mean_distances_plot = [i for j, i in enumerate(self.mean_distances_plot) if j not in remove_indices]
        self.falling_robots_plot = [i for j, i in enumerate(self.falling_robots_plot) if j not in remove_indices]


    def store_PPO_run_info(self, PPO_info, iteration):
        if not(PPO_info is None) and self.show_PPO_graph and (iteration % self.frequency_plot) == 0:
            self.gamma.append(PPO_info["gamma"])
            self.learning_rates.append(PPO_info["lr"])
            self.surrogate_loss.append(PPO_info["surrogate_loss"])
            self.value_loss.append(PPO_info["value_loss"])
            self.entropy.append(PPO_info["entropy"])
            self.student_loss.append(PPO_info["student_loss"])
            self.terrain_curriculum.append(PPO_info["terrain_curriculum"])
            self.loss_mean.append(PPO_info["loss_mean"])
            self.time_steps_PPO.append(PPO_info["time_steps_PPO"])

    def store_data(self, distance, reward, weight, noise, iteration, total_time, falling_robots, dof_pos, dof_desired, std_height=None, show_plot=False,
                   pause=False, PPO_info=None):

        mean_reward = float(torch.mean(reward))
        mean_distance = float(torch.mean(distance))
        std_height_mean = float(torch.mean(std_height)) if not (std_height is None) else None
        self.iteration = iteration

        if self.video_frequency != 0 and (iteration % self.video_frequency) == 0:
            self._start_video_record_()

        if (iteration % self.frequency_plot) == 0:
            self.x_axis.append(iteration)
            self.mean_distances.append(mean_distance)
            self.mean_reward.append(mean_reward)
            self.mean_std_height.append(std_height_mean)
            self.soft_reward.append(float(np.mean(self.mean_reward[-5:])))
            self.falling_robots.append(falling_robots)

            self.x_axis_plot.append(iteration)
            self.mean_distances_plot.append(mean_distance)
            self.mean_reward_plot.append(mean_reward)
            self.soft_reward_plot.append(float(np.mean(self.mean_reward_plot[-5:])))
            self.falling_robots_plot.append(falling_robots)

            self.dof_pos = dof_pos
            self.dof_desired = dof_desired

            self.store_PPO_run_info(PPO_info, iteration)

            if show_plot:
                self.plot_log(False, pause)

        if self.save_data:
            new_data = {
                "max_reward": float(torch.max(reward)),
                "min_reward": float(torch.min(reward)),
                "mean_reward": mean_reward,

                "max_distance": float(torch.max(distance)),
                "min_distance": float(torch.min(distance)),
                "mean_distance": mean_distance,
            }

            for k in new_data:
                self.renew_data[k] = self.stored_info[k].store_weights(new_data[k], weight, noise, iteration,
                                                                       total_time, torch.argmax(distance))

    def store_data_post(self, weights):

        for k in self.renew_data:
            if self.renew_data[k]:
                self.renew_data[k] = self.stored_info[k].store_weights_post(weights)

    def store_curriculum(self, curriculum):
        self.curriculum = curriculum

    def store_algorithm_parameters(self, algorithm_parameters):
        self.algorithm_parameters = algorithm_parameters

    def store_reward_param(self, reward_params):
        self.rewards_weights = reward_params

    def __save_nn_config(self):
        with open(os.path.join(self.folder, "nn_config.json"), "w") as f:
            json.dump(self.nn_config, f, indent=2)

        self.nn_config = None

    def __save_PIBB_param(self):
        with open(os.path.join(self.folder, "pibb_param.json"), "w") as f:
            json.dump(self.PIBB_param, f, indent=2)

        self.PIBB_param = None

    def __save_curriculum(self):
        import pickle

        with open(os.path.join(self.folder, "curriculum.pickle"), "wb") as f:
            pickle.dump(self.curriculum, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.curriculum = None

    def __save_algorithm_parameters(self):
        import pickle

        with open(os.path.join(self.folder, "algorithm_parameters.pickle"), "wb") as f:
            pickle.dump(self.algorithm_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.algorithm_parameters = None

    def save_rewards(self, rewards, filename):
        if self.save_data:
            self.__save_rewards_param(filename=filename, rewards=rewards)

    def __save_rewards_param(self, filename=None, rewards=None):
        if filename is None:
            filename = "rewards_param.json"

        if rewards is None:
            rewards = self.rewards_weights

        with open(os.path.join(self.folder, filename), "w") as f:
            json.dump(rewards, f, indent=2)

        self.rewards_weights = None

    def save_stored_data(self, force=False, actual_weight=None, actual_reward=None, iteration=None, total_time=None,
                         noise=None, index=0):

        if not self.save_data:
            return False

        if not (force or 0 == (iteration % self.frequency)):
            return False

        if not path.exists(self.folder):
            self.create_folder()

        if not (self.nn_config is None):
            self.__save_nn_config()

        if not (self.PIBB_param is None):
            self.__save_PIBB_param()

        if not (self.curriculum is None):
            self.__save_curriculum()

        if not (self.algorithm_parameters is None):
            self.__save_algorithm_parameters()

        if not (self.rewards_weights is None):
            self.__save_rewards_param()

        for k in self.stored_info:
            filename = k + "_data.pickle"

            full_name = os.path.join(self.folder, filename)
            with open(full_name, "wb") as f:
                pickle.dump(self.stored_info[k], f)

        if not (actual_weight is None):
            temp = DataStore()

            temp.iteration = iteration
            temp.weights = actual_weight
            temp.weights_post = actual_weight
            temp.noise = noise
            temp.index = index
            temp.record = float(torch.mean(actual_reward))
            temp.time = total_time

            full_name = os.path.join(self.folder, f"{iteration}.pickle")

            with open(full_name, "wb") as f:
                pickle.dump(temp, f)

        return True
