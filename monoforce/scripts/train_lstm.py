import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import matplotlib.pyplot as plt
from monoforce.models.traj_predictor.lstm import TrajectoryLSTM
from monoforce.datasets.rough import ROUGH, rough_seq_paths


class Data(ROUGH):
    def __init__(self, path, is_train=False):
        super(Data, self).__init__(path, is_train=is_train)

    def get_sample(self, i):
        control_ts, controls = self.get_controls(i)

        traj = self.get_traj(i)
        traj_ts = traj['stamps']
        traj_ts = torch.as_tensor(traj_ts - traj_ts[0], dtype=torch.float32)

        poses = traj['poses']
        xyz = torch.as_tensor(poses[:, :3, 3], dtype=torch.float32)
        Rs = poses[:, :3, :3]
        rpy = torch.as_tensor(Rotation.from_matrix(Rs).as_euler('xyz'), dtype=torch.float32)
        states = torch.cat((xyz, rpy), dim=-1)

        hm = self.get_geom_height_map(i)[0:1]
        return (hm,
                control_ts, controls,
                traj_ts, states)


class Trainer:
    def __init__(self,
                 state_features=6,
                 control_features=2,
                 heightmap_shape=(128, 128),
                 batch_size=1,
                 lr=1e-4,
                 n_epochs=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate the model
        self.lstm = TrajectoryLSTM(state_features, control_features, heightmap_shape)
        self.lstm.train()
        self.lstm.to(self.device)

        # Dataset
        ds = Data(rough_seq_paths[0])
        self.batch_size = batch_size
        self.loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        # Loss function and optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr)

        self.n_epochs = n_epochs

    def inference(self, batch):
        hm, control_ts, controls, traj_ts, xyz_rpy = batch
        xyz_rpy0 = xyz_rpy[:, 0]
        xyz_rpy_pred = self.lstm(xyz_rpy0, controls, hm)
        return xyz_rpy_pred

    def compute_loss(self, batch, xyz_rpy_pred):
        hm, control_ts, controls, traj_ts, xyz_rpy = batch
        # find the closest timesteps in the trajectory to the ground truth timesteps
        ts_ids = torch.argmin(torch.abs(control_ts.unsqueeze(1) - traj_ts.unsqueeze(2)), dim=2)
        # compute the loss as the mean squared error between the predicted and ground truth poses
        batch_size = xyz_rpy_pred.size(0)
        loss = self.criterion(xyz_rpy_pred[torch.arange(batch_size).unsqueeze(1), ts_ids], xyz_rpy)
        return loss

    def vis(self, batch, xyz_rpy_pred):
        with torch.no_grad():
            hm, control_ts, controls, traj_ts, xyz_rpy = batch

            plt.figure(figsize=(21, 7))
            plt.subplot(1, 3, 1)
            plt.plot(xyz_rpy[0, :, 0].cpu().numpy(), xyz_rpy[0, :, 1].cpu().numpy(),
                     '-r', label='GT poses')
            plt.plot(xyz_rpy_pred[0, :, 0].cpu().numpy(), xyz_rpy_pred[0, :, 1].cpu().numpy(),
                     '--b', label='Pred poses')
            plt.grid()
            plt.axis('equal')
            plt.legend()
            plt.title('Trajectories XY')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')

            plt.subplot(1, 3, 2)
            plt.plot(traj_ts[0].cpu().numpy(), xyz_rpy[0, :, 2].cpu().numpy(),
                     '-r', label='GT poses')
            plt.plot(control_ts[0].cpu().numpy(), xyz_rpy_pred[0, :, 2].cpu().numpy(),
                     '--b', label='Pred poses')
            plt.grid()
            plt.ylim(-1, 1)
            plt.legend()
            plt.title('Trajectories Z')
            plt.xlabel('t [s]')
            plt.ylabel('Z [m]')

            plt.subplot(1, 3, 3)
            plt.title('Euler angles')
            plt.plot(traj_ts[0].cpu().numpy(), xyz_rpy[0, :, 3].cpu().numpy(),
                     '-r', label='GT roll')
            plt.plot(control_ts[0].cpu().numpy(), xyz_rpy_pred[0, :, 3].cpu().numpy(),
                     '--r', label='Pred roll')
            plt.plot(traj_ts[0].cpu().numpy(), xyz_rpy[0, :, 4].cpu().numpy(),
                     '-g', label='GT pitch')
            plt.plot(control_ts[0].cpu().numpy(), xyz_rpy_pred[0, :, 4].cpu().numpy(),
                     '--g', label='Pred pitch')
            plt.plot(traj_ts[0].cpu().numpy(), xyz_rpy[0, :, 5].cpu().numpy(),
                     '-b', label='GT yaw')
            plt.plot(control_ts[0].cpu().numpy(), xyz_rpy_pred[0, :, 5].cpu().numpy(),
                     '--b', label='Pred yaw')
            plt.ylim(-1, 1)
            plt.legend()
            plt.grid()
            plt.xlabel('t [s]')
            plt.ylabel('angle [rad]')

            plt.show()

    def train(self):
        for epoch in range(self.n_epochs):
            # Training loop
            epoch_loss = 0.
            for batch in tqdm(self.loader, desc=f'Training epoch {epoch}:', leave=False, total=len(self.loader)):
                self.optimizer.zero_grad()

                # model inference
                batch = [b.to(self.device) for b in batch]
                xyz_rpy_pred = self.inference(batch)

                # compute loss
                loss = self.compute_loss(batch, xyz_rpy_pred)
                epoch_loss += loss.item()

                # backpropagation and optimization
                loss.backward()
                self.optimizer.step()
            epoch_loss /= len(self.loader)
            print(f'Epoch loss: {epoch_loss}')
            self.vis(batch, xyz_rpy_pred)


def main():
    trainer = Trainer(batch_size=16)
    trainer.train()


if __name__ == '__main__':
    main()
