import numpy as np
import importlib
from typing import Optional, Dict
from ctf4science.eval_module import evaluate_custom
import torch
import torch.nn as nn
from torchdiffeq import odeint

class Normalizer:
    def __init__(self, y,normalize=True):
        self.norm=normalize
        self.mean = y.mean(dim=0, keepdim=True)
        self.std = y.std(dim=0, keepdim=True)

    def normalize(self, y):
        if not self.norm:
            return y
        return (y - self.mean) / self.std
    def denormalize(self, y):
        if not self.norm:
            return y       
        return y * self.std + self.mean
    
class ODEF(nn.Module):
    def __init__(self, hidden_dim=512,input_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.Tanh(),
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, y):
        return self.net(y)


class NeuralOde:
    """
    A class representing a NeuralODE baseline
    """

    def __init__(self, config: Dict, train_data: Optional[np.ndarray] = None, init_data: Optional[np.ndarray] = None, prediction_timesteps=None,training_timesteps=None, pair_id: Optional[int] = None):
        """
        Initialize the NeuralODE model with the provided configuration.

        Args:
            config (Dict): Configuration dictionary containing method and parameters.
            train_data (Optional[np.ndarray]): Training data for 'average' method.
        """
        self.method = config['model']['method']
        self.dataset_name = config['dataset']['name']
        self.lr = float( config['params']['lr'])
        self.hidden_size = int(config['params']['hidden_dim'])
        self.batch_size =  int(config['params']['batch_size'])
        self.seq_len =  int(config['params']['seq_len'])
        self.pair_id = pair_id
        self.train_data = train_data
        self.init_data=init_data
        self.prediction_timesteps=torch.tensor(prediction_timesteps)
        self.training_timesteps=torch.tensor(training_timesteps)
        self.prediction_horizon_steps = len(prediction_timesteps)

    def predict(self) -> np.ndarray:
        """
        Generate predictions based on the specified model method.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t_train=[]
        y_train=[]
        t_eval=[]
        y_eval=[]
        for traj_idx,traj in enumerate(self.train_data):
 
            traj = torch.tensor(traj, dtype=torch.float32).to(device)  # shape: [3, T]
            T = traj.shape[0]

            split_idx = int(0.8 * T)

            t_total = self.training_timesteps[traj_idx]

            t_train.append(t_total[:split_idx])
            y_train.append( traj[:split_idx])
            t_eval.append( t_total[split_idx:])
            y_eval.append( traj[split_idx:])
            

        def train_neural_ode(t_train_list, y_train_list, t_eval_list, y_eval_list, niters=100, lr=1e-3, batch_size=40, seq_len=40):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # === Concatenate all training data for normalization ===
            y_train_all = torch.cat(y_train_list, dim=0)
            normalizer = Normalizer(y_train_all, normalize=self.method == "normalize")

            # === Normalize all trajectories ===
            y_train_list = [normalizer.normalize(y).to(device) for y in y_train_list]
            y_eval_list = [normalizer.normalize(y).to(device) for y in y_eval_list]
            t_train_list = [t.to(device) for t in t_train_list]
            t_eval_list = [t.to(device) for t in t_eval_list]

            dim = y_train_list[0].shape[1]
            ode_func = ODEF(input_dim=dim,hidden_dim=self.hidden_size).to(device)
            optimizer = torch.optim.Adam(ode_func.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            # === Build dataset from all subsequences in all trajectories ===
            X_all, T_all = [], []
            for t, y in zip(t_train_list, y_train_list):
                num_sequences = y.shape[0] - seq_len
                for i in range(num_sequences):
                    X_all.append(y[i:i + seq_len])
                    T_all.append(t[i:i + seq_len])
            X = torch.stack(X_all)
            T = torch.stack(T_all)

            dataset = torch.utils.data.TensorDataset(T, X)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            best_eval_loss = float('inf')
            best_model_state = None
            iters = 0

            for epoch in range(niters):
                if iters > niters:
                    break
                for t_batch, y_batch in loader:
                    epoch_loss = 0.0
                    iters += 1
                    if iters > niters:
                        break

                    t_batch = t_batch.to(device)  # [B, L]
                    y_batch = y_batch.to(device)  # [B, L, D]
                    y0 = y_batch[:, 0, :]         # [B, D]

                    pred_y = torch.stack([
                        odeint(ode_func, y0[i], t_batch[i]) for i in range(y0.shape[0])
                    ])  # [B, L, D]

                    loss = loss_fn(pred_y, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    # === Evaluate on all eval trajectories ===
                    with torch.no_grad():
                        eval_losses = []
                        for t_eval, y_eval in zip(t_eval_list, y_eval_list):
                            pred_eval = odeint(ode_func, y_eval[0], t_eval)
                            loss_eval = loss_fn(
                                normalizer.denormalize(pred_eval),
                                normalizer.denormalize(y_eval)
                            )
                            eval_losses.append(loss_eval.item())
                        avg_eval_loss = sum(eval_losses) / len(eval_losses)

                        if avg_eval_loss < best_eval_loss:
                            best_eval_loss = avg_eval_loss
                            best_model_state = ode_func.state_dict()

                    print(f"Iteration {iters:4d} | Train Loss: {epoch_loss:.6f} | Eval Loss: {avg_eval_loss:.6f}")

            # === Load best model ===
            if best_model_state is not None:
                ode_func.load_state_dict(best_model_state)

            # === Final prediction ===
            if self.init_data is None:
                self.init_data = traj
            else:
                self.init_data = torch.tensor(self.init_data, dtype=torch.float32).to(device)

            norm_init = normalizer.normalize(self.init_data)
            pred_traj = odeint(ode_func, norm_init[-1], self.prediction_timesteps)
            pred_traj = normalizer.denormalize(pred_traj)
            return np.array(pred_traj.detach().cpu())

        pred= train_neural_ode(t_train, y_train, t_eval, y_eval,batch_size=self.batch_size,seq_len=self.seq_len,lr=self.lr)
        return pred