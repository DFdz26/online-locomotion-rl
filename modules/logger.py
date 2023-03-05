import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import os
from os import path
import pickle
import matplotlib as mpl
import json

mpl.use('qt5agg')
root_run_folder_name = "runs"


class data_store:
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
            self.weights = weights.detach().clone()
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
    def __init__(self, save=False, frequency=20, frequency_plot=5, robot="", nn_config=None, PIBB_param=None,
                 test_value=False, size_figure=2):
        self.save_data = save
        self.nn_config = nn_config
        self.PIBB_param = PIBB_param
        self.test_value = test_value
        self.size_figure = size_figure
        self.filename = None

        if not path.exists(root_run_folder_name):
            os.mkdir(root_run_folder_name)

        now = datetime.now()
        self.folder_time = now.strftime("%d_%m_%Y__%H_%M_%S")
        self.folder = os.path.join(root_run_folder_name, self.folder_time)

        self.frequency = frequency
        self.frequency_plot = frequency_plot

        self.stored_info = {
            "max_reward": data_store(),
            "min_reward": data_store(),
            "mean_reward": data_store(),

            "max_distance": data_store(),
            "min_distance": data_store(),
            "mean_distance": data_store(),
        }
        self.renew_data = {
            "max_reward": False,
            "min_reward": False,
            "mean_reward": False,

            "max_distance": False,
            "min_distance": False,
            "mean_distance": False,
        }

        self.x_axis = []
        self.mean_distances = []
        self.mean_reward = []
        self.mean_std_heigh = []
        self.figure, self.ax = plt.subplots(size_figure)
        plt.ion()
        self.distance = []
        self.time = []

        if robot != "":
            self.set_robot_name(robot)

    def recover_nn_information(self):
        self.folder = os.path.dirname(self.filename)
        nn_info_file = os.path.join(self.folder, "nn_config.json")

        with open(nn_info_file, "r") as f:
            nn_info = json.load(f)

        return nn_info

    def recover_data_class(self, filename):
        self.filename = filename

        with open(filename, "rb") as f:
            data = pickle.load(f)

        return data

    def recover_data(self, filename, post=True):
        self.filename = filename

        with open(filename, "rb") as f:
            data = pickle.load(f)

        return data.weights_post if post else data.weights, data.noise, data.iteration, data.record, data.index

    def set_robot_name(self, robot):
        path_to_check = os.path.join(root_run_folder_name, robot)

        if self.save_data:
            if not path.exists(path_to_check):
                os.mkdir(path_to_check)

            self.folder = os.path.join(path_to_check, self.folder_time)

    def save_points_testing(self, distance, time):
        self.distance.append(float(torch.mean(distance)))
        self.time.append(time)

    def create_folder(self):

        if self.save_data:
            os.mkdir(self.folder)

    def plot_in_grap(self):
        if self.test_value:

            xpoints = np.array(self.time)
            ypoints = np.array(self.distance)

            self.ax.set_title("Distance vs time")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Distance (m)")

            self.ax.plot(xpoints, ypoints)
        else:
            xpoints = np.array(self.x_axis)
            ypoints = np.array(self.mean_reward)

            self.ax[0].set_title("Avg. reward vs iteration")
            self.ax[0].set_xlabel("Iteration")
            self.ax[0].set_ylabel("Reward")

            self.ax[0].plot(xpoints, ypoints)
            ypoints = np.array(self.mean_distances)

            self.ax[1].set_title("Avg. distance (m) vs iteration")
            self.ax[1].set_xlabel("Iteration")
            self.ax[1].set_ylabel("Distance (m)")
            self.ax[1].plot(xpoints, ypoints)

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

        file_graph_data = os.path.join(self.folder, filename)

        with open(file_graph_data, "w") as f:
            json.dump(dic, f, indent=2)

    def log(self, save=True, block=True, plot_file_name="", save_datapoint=False):

        self.plot_in_grap()

        fig = mpl.pyplot.gcf()

        fig.set_size_inches(18.5, 10.5)
        if save:
            plt.savefig(os.path.join(self.folder, plot_file_name + ".png"), dpi=100)

        if save_datapoint:
            self.__save_datapoints__()

        plt.pause(0.001)
        plt.show(block=block)
        plt.pause(0.001)

    def store_data(self, distance, reward, weight, noise, iteration, total_time, std_height, show_plot=False,
                   pause=False):

        mean_reward = float(torch.mean(reward))
        mean_distance = float(torch.mean(distance))
        std_height_mean = float(torch.mean(std_height)) if not (std_height is None) else None

        if (iteration % self.frequency_plot) == 0:
            self.x_axis.append(iteration)
            self.mean_distances.append(mean_distance)
            self.mean_reward.append(mean_reward)
            self.mean_std_heigh.append(std_height_mean)

            if show_plot:
                self.log(False, pause)

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

    def __save_rewards_param(self):
        with open(os.path.join(self.folder, "rewards_param.json"), "w") as f:
            json.dump(self.rewards_weights, f, indent=2)

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

        if not (self.rewards_weights is None):
            self.__save_rewards_param()

        for k in self.stored_info:
            filename = k + "_data.pickle"

            full_name = os.path.join(self.folder, filename)
            with open(full_name, "wb") as f:
                pickle.dump(self.stored_info[k], f)

        if not (actual_weight is None):
            temp = data_store()

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
