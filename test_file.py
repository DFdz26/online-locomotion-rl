import torch


def init_height_points(num_envs):
    # 1mx1.6m rectangle (without center line)
    y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])  # 10-50cm on each side
    x = 0.1 * torch.tensor([-8, -6, -4, -2, 2, 4, 6, 8])  # 20-80cm on each side
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    num_height_points = grid_x.numel()
    points = torch.zeros(num_envs, num_height_points, 3)
    points[:, :, 0] = grid_x.flatten()
    points[:, :, 1] = grid_y.flatten()
    return points, num_height_points


if __name__ == "__main__":
    points_, num_heights = init_height_points(1)

    print(points_)
    print(num_heights)
