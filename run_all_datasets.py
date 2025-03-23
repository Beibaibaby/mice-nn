#!/usr/bin/env python3

import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data çimport Dataset, DataLoader, random_split

################################################################################
# 1) Neural Dataset Definition
################################################################################

class NeuralDataset(Dataset):
    def __init__(self, eigenface_evoked, dff_evoked,
                 eigenface_isi, dff_isi,
                 apply_norm=True):
        super().__init__()
        face_dim  = eigenface_evoked.shape[0]   # e.g. 500
        n_stim    = eigenface_evoked.shape[2]   # 4
        self.n_neurons = dff_evoked.shape[0]    # e.g. 628

        self.samples_x = []
        self.samples_y = []

        # ============ Evoked Data ============
        for c in range(n_stim):
            face_block   = eigenface_evoked[:, :, c]  # (face_dim, time)
            neural_block = dff_evoked[:, :, c]        # (n_neurons, time)
            stim_onehot = np.zeros(n_stim, dtype=np.float32)
            stim_onehot[c] = 1.0

            # Loop over time columns
            for col in range(face_block.shape[1]):
                face_col   = face_block[:, col]
                neural_col = neural_block[:, col]
                
                # Skip if any NaN in face_col or neural_col
                # (face_col might not have NaNs here, but let's be safe)
                if np.isnan(face_col).any() or np.isnan(neural_col).any():
                    continue

                # Convert to float32, combine
                face_col   = face_col.astype(np.float32)
                neural_col = neural_col.astype(np.float32)
                x_in = np.concatenate([face_col, stim_onehot], axis=0)
                self.samples_x.append(x_in)
                self.samples_y.append(neural_col)

        # ============ ISI Data ============
        face_isi_block   = eigenface_isi   # (face_dim, time)
        neural_isi_block = dff_isi         # (n_neurons, time)
        zero_stim = np.zeros(n_stim, dtype=np.float32)

        for col in range(face_isi_block.shape[1]):
            face_col   = face_isi_block[:, col]
            neural_col = neural_isi_block[:, col]

            # Skip any column that has NaNs
            if np.isnan(face_col).any() or np.isnan(neural_col).any():
                continue

            face_col   = face_col.astype(np.float32)
            neural_col = neural_col.astype(np.float32)
            x_in = np.concatenate([face_col, zero_stim], axis=0)
            self.samples_x.append(x_in)
            self.samples_y.append(neural_col)

        # Now convert to torch
        self.samples_x = torch.tensor(np.array(self.samples_x))  # (N, face_dim+4)
        self.samples_y = torch.tensor(np.array(self.samples_y))  # (N, n_neurons)

        print(f"After filtering NaN columns: {self.samples_x.shape[0]} samples remain.")

        # Per-neuron normalization
        if apply_norm and len(self.samples_x) > 0:
            self.means = self.samples_y.mean(dim=0)
            self.stds  = self.samples_y.std(dim=0)
            self.stds  = torch.where(self.stds < 1e-9, torch.ones_like(self.stds), self.stds)
            self.samples_y = (self.samples_y - self.means) / self.stds
        else:
            self.means = torch.zeros((self.n_neurons,), dtype=torch.float32)
            self.stds  = torch.ones((self.n_neurons,), dtype=torch.float32)

    def __len__(self):
        return len(self.samples_x)

    def __getitem__(self, idx):
        return self.samples_x[idx], self.samples_y[idx]


################################################################################
# 2) Multi-Branch MLP
################################################################################

class MultiBranchMLP(nn.Module):
    def __init__(self,
                 face_dim=500,
                 stim_dim=4,
                 hidden_face=[1024, 512, 256],
                 hidden_stim=[128, 64],
                 hidden_fuse=[256, 128],
                 output_dim=229):
        """
        output_dim can be different for each dataset (e.g. 229, 628, etc.).
        """
        super().__init__()

        # Face branch
        face_layers = []
        in_dim = face_dim
        for hdim in hidden_face:
            face_layers.append(nn.Linear(in_dim, hdim))
            face_layers.append(nn.ReLU())
            in_dim = hdim
        self.face_net = nn.Sequential(*face_layers)

        # Stim branch
        stim_layers = []
        in_dim_s = stim_dim
        for hdim_s in hidden_stim:
            stim_layers.append(nn.Linear(in_dim_s, hdim_s))
            stim_layers.append(nn.ReLU())
            in_dim_s = hdim_s
        self.stim_net = nn.Sequential(*stim_layers)

        # Fusion
        fuse_in = hidden_face[-1] + hidden_stim[-1]
        fuse_seq = []
        prev_dim = fuse_in
        for fdim in hidden_fuse:
            fuse_seq.append(nn.Linear(prev_dim, fdim))
            fuse_seq.append(nn.ReLU())
            prev_dim = fdim
        fuse_seq.append(nn.Linear(prev_dim, output_dim))  # final layer => n_neurons
        self.fuse = nn.Sequential(*fuse_seq)

        self.face_dim = face_dim
        self.stim_dim = stim_dim

    def forward(self, x):
        # x: shape (B, face_dim + stim_dim)
        face_part = x[:, :self.face_dim]
        stim_part = x[:, self.face_dim:(self.face_dim + self.stim_dim)]

        face_out = self.face_net(face_part)
        stim_out = self.stim_net(stim_part)
        combined = torch.cat([face_out, stim_out], dim=-1)
        out = self.fuse(combined)  # shape (B, output_dim)
        return out


################################################################################
# 3) Loss Functions
################################################################################

def correlation_loss(y_pred, y_true, eps=1e-8):
    """
    1 - Pearson correlation across all outputs (flattened).
    """
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)
    pm = y_pred_f.mean()
    tm = y_true_f.mean()
    cov = ((y_pred_f - pm) * (y_true_f - tm)).sum()
    var1 = ((y_pred_f - pm) ** 2).sum() + eps
    var2 = ((y_true_f - tm) ** 2).sum() + eps
    corr = cov / (var1.sqrt() * var2.sqrt())
    return 1.0 - corr


def combined_loss(y_pred, y_true, alpha=0.5):
    """
    A blend of MSE and (1 - correlation).
    """
    mse_v = ((y_pred - y_true) ** 2).mean()
    corr_v = correlation_loss(y_pred, y_true)
    return alpha * mse_v + (1 - alpha) * corr_v


################################################################################
# 4) R^2 Helper
################################################################################

def compute_r2(model, loader, ds, device='cuda'):
    """
    We'll pass the entire loader to the model, invert normalization,
    and compute average R^2 across all neurons.
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for x_batch, y_batch_norm in loader:
            x_batch = x_batch.to(device)
            y_batch_norm = y_batch_norm.to(device)

            preds_norm = model(x_batch)  # shape (B, n_neurons)

            # Invert normalization
            means = ds.means.to(device)   # (n_neurons,)
            stds  = ds.stds.to(device)    # (n_neurons,)
            preds_raw = preds_norm * stds + means
            true_raw  = y_batch_norm * stds + means

            all_preds.append(preds_raw.cpu().numpy())
            all_true.append(true_raw.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N, n_neurons)
    all_true  = np.concatenate(all_true,  axis=0)  # (N, n_neurons)

    # Compute R^2 per neuron
    n_neurons = all_preds.shape[1]
    r2_arr = []
    for n in range(n_neurons):
        y_t = all_true[:, n]
        y_p = all_preds[:, n]
        ss_res = np.sum((y_t - y_p)**2)
        ss_tot = np.sum((y_t - np.mean(y_t))**2)
        if ss_tot > 1e-12:
            r2_arr.append(1 - ss_res / ss_tot)
        else:
            r2_arr.append(0.0)

    return np.mean(r2_arr)


################################################################################
# 5) Train Function (Handles variable n_neurons)
################################################################################

def train_model(eigenface_evoked, dff_evoked,
                eigenface_isi, dff_isi,
                epochs=100, batch_size=64, lr=1e-3,
                alpha=0.5,
                device='cuda'):
    """
    Creates the dataset, splits train/val, trains the multi-branch MLP,
    and returns (model, ds_full, final_val_loss).
    If the dataset has n_neurons != 229, the final layer is sized accordingly.
    """

    ds_full = NeuralDataset(eigenface_evoked, dff_evoked,
                            eigenface_isi, dff_isi,
                            apply_norm=True)
    N = len(ds_full)
    val_size = int(0.2 * N)
    train_size = N - val_size

    if train_size <= 0:
        print("  ERROR: Not enough samples to form a training set. Skipping.")
        return None, None, float('inf')

    train_ds, val_ds = random_split(ds_full, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # n_neurons can vary
    n_neurons = ds_full.n_neurons
    face_dim  = eigenface_evoked.shape[0]
    n_stim    = eigenface_evoked.shape[2]

    # Build model
    model = MultiBranchMLP(
        face_dim=face_dim,
        stim_dim=n_stim,
        hidden_face=[1024, 512, 256],
        hidden_stim=[128, 64],
        hidden_fuse=[256, 128],
        output_dim=n_neurons
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    max_patience = 50

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)  # shape (B, n_neurons)
            loss = combined_loss(pred, y_batch, alpha=alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                preds_val = model(x_val)  # shape (B, n_neurons)
                lv = combined_loss(preds_val, y_val, alpha=alpha)
                val_loss_sum += lv.item()

        val_loss = val_loss_sum / len(val_loader)
        scheduler.step(val_loss)

        # Print progress every 20 epochs or final
        if (epoch % 20 == 0) or (epoch == epochs - 1):
            # Might produce NaN if everything is failing
            train_r2 = compute_r2(model, train_loader, ds_full, device=device)
            val_r2   = compute_r2(model, val_loader,   ds_full, device=device)

            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train R^2={train_r2:.4f}, Val R^2={val_r2:.4f}")

        # Early Stopping
        if val_loss < best_val_loss - 0.3:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print("Early stopping triggered.")
                break

    # If we never updated best_val_loss, it remains inf => we had all NaNs
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, ds_full, best_val_loss


################################################################################
# 6) Prediction Helpers
################################################################################

def invert_normalization(pred_norm, ds):
    """
    Convert predicted normalized (N, n_neurons) back to raw domain
    using ds.means, ds.stds from the training dataset.
    """
    device = pred_norm.device
    means = ds.means.to(device)
    stds  = ds.stds.to(device)
    return pred_norm * stds + means


def compute_average_face(ds_full):
    """
    Compute the average face vector (dim=face_dim) across ds_full.
    ds_full[i] => (x_in, y_in) with x_in of size (face_dim+4,).
    """
    all_faces = []
    face_dim = ds_full[0][0].shape[0] - 4  # e.g. 500
    for i in range(len(ds_full)):
        x_in, _ = ds_full[i]
        face_part = x_in[:face_dim]
        all_faces.append(face_part)
    all_faces = torch.stack(all_faces, dim=0)  # (N, face_dim)
    avg_face = all_faces.mean(dim=0)           # (face_dim,)
    return avg_face

def predict_stim_no_face_motion(model, ds_full, device='cuda'):
    """
    f(stimulus, no-face-motion) => shape (n_neurons, 4)
    """
    model.eval()
    face_dim = ds_full[0][0].shape[0] - 4
    n_stim   = 4
    avg_face = compute_average_face(ds_full).to(device)

    # build 4 one-hot input vectors
    stim_inputs = []
    for c in range(n_stim):
        x_in = torch.zeros(face_dim + n_stim, dtype=torch.float32, device=device)
        x_in[:face_dim] = avg_face
        x_in[face_dim + c] = 1.0
        stim_inputs.append(x_in)

    stim_inputs = torch.stack(stim_inputs, dim=0)  # (4, face_dim+4)

    with torch.no_grad():
        pred_norm = model(stim_inputs)   # (4, n_neurons)
        pred_raw  = invert_normalization(pred_norm, ds_full)  # (4, n_neurons)

    return pred_raw.cpu().numpy().T  # => (n_neurons, 4)


def predict_no_stim_face_motion(model, eigenface_evoked, ds_full, device='cuda'):
    """
    f(no-stimulus, face-motion) => shape (n_neurons, 4000)
    """
    model.eval()

    face_dim = eigenface_evoked.shape[0]
    n_stim   = eigenface_evoked.shape[2]

    face_motion_list = []
    for c in range(n_stim):
        face_block = eigenface_evoked[:, :, c]  # (face_dim, time)
        for col in range(face_block.shape[1]):
            face_col = face_block[:, col]
            x_in = torch.zeros(face_dim + n_stim, dtype=torch.float32)
            x_in[:face_dim] = torch.from_numpy(face_col)
            # last 4 dims = 0 => "no stimulus"
            face_motion_list.append(x_in)

    face_motion_tensor = torch.stack(face_motion_list, dim=0).to(device)  # (4000, face_dim+4)

    with torch.no_grad():
        pred_norm = model(face_motion_tensor)   # (4000, n_neurons)
        pred_raw  = invert_normalization(pred_norm, ds_full)  # (4000, n_neurons)

    return pred_raw.cpu().numpy().T  # => (n_neurons, 4000)


################################################################################
# 7) Main: Loop Over .mat Files, Train & Save If Valid
################################################################################

def process_single_file(mat_path,
                        epochs=150,
                        batch_size=64,
                        lr=1e-3,
                        alpha=0.5,
                        device='cuda'):
    """
    - Loads data from a .mat
    - Trains MLP (if feasible)
    - If training yields a valid (non-NaN) best_val_loss, we save predictions
      in a separate subfolder called 'Predictions_output'.
    - Otherwise, we print a "flag" message and skip saving.
    """

    print(f"\n=== Processing {mat_path} ===")
    try:
        data = scipy.io.loadmat(mat_path)
    except (OSError, NotImplementedError) as e:
        print(f"  ERROR loading {mat_path}: {e}")
        print("  Skipping this file.")
        return

    # 1) Check required variables
    required_vars = ["Eigenface_0_trials_evoked", "Eigenface_0_trials_isi",
                     "dFF0_trials_evoked",       "dFF0_trials_isi"]
    for var in required_vars:
        if var not in data:
            print(f"  WARNING: missing '{var}' in {mat_path}. Skipping.")
            return

    # 2) Convert
    eigenface_evoked = np.array(data["Eigenface_0_trials_evoked"])  # (face_dim, time, 4)
    eigenface_isi    = np.array(data["Eigenface_0_trials_isi"])     # (face_dim, time)
    dff_evoked       = np.array(data["dFF0_trials_evoked"])         # (n_neurons, time, 4)
    dff_isi          = np.array(data["dFF0_trials_isi"])            # (n_neurons, time)

    # 3) Train
    model, ds_full, best_val_loss = train_model(
        eigenface_evoked, dff_evoked,
        eigenface_isi,    dff_isi,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        alpha=alpha,
        device=device
    )

    # If train_model returned None => skip
    if model is None or ds_full is None:
        print("  => Not saving anything (invalid or zero training set).")
        return

    # 4) Check if best_val_loss is still inf (means all losses were NaN)
    if best_val_loss == float('inf') or np.isnan(best_val_loss):
        print(f"  => Training collapsed (val_loss=NaN). Flagging this dataset, skipping save.")
        return

    # 5) If valid, do predictions
    stim_no_face = predict_stim_no_face_motion(model, ds_full, device=device)
    no_stim_face = predict_no_stim_face_motion(model, eigenface_evoked, ds_full, device=device)

    # 6) Save in a new subfolder "Predictions_output" inside the same directory
    mat_dir    = os.path.dirname(mat_path)
    base_name  = os.path.splitext(os.path.basename(mat_path))[0]

    # create subfolder: e.g. ./Data_trial_binned_mat/Predictions_output
    save_folder = os.path.join(mat_dir, "Predictions_output")
    os.makedirs(save_folder, exist_ok=True)

    # form the filenames
    npy_stim   = os.path.join(save_folder, f"pred_stim_no_face_motion_{base_name}.npy")
    npy_nostim = os.path.join(save_folder, f"pred_no_stim_face_motion_{base_name}.npy")
    mat_output = os.path.join(save_folder, f"predictions_combined_{base_name}.mat")

    print(f"  Saving predictions into {save_folder}")
    print(f"    => {os.path.basename(npy_stim)}")
    print(f"    => {os.path.basename(npy_nostim)}")
    print(f"    => {os.path.basename(mat_output)}")

    # Save as .npy
    np.save(npy_stim,   stim_no_face)
    np.save(npy_nostim, no_stim_face)

    # Save also as .mat
    data_dict = {
        'stim_no_face': stim_no_face,   # shape (n_neurons, 4)
        'no_stim_face': no_stim_face    # shape (n_neurons, 4000)
    }
    scipy.io.savemat(mat_output, data_dict)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    mat_dir = "./Data_trial_binned_mat"
    if not os.path.isdir(mat_dir):
        print(f"Directory '{mat_dir}' does not exist. Exiting.")
        return

    all_files = sorted(os.listdir(mat_dir))
    mat_files = [f for f in all_files if f.endswith(".mat")]

    print(f"Found {len(mat_files)} .mat files in '{mat_dir}'. Will process each.")

    for mat_file in mat_files:
        mat_path = os.path.join(mat_dir, mat_file)
        process_single_file(mat_path,
                            epochs=150,
                            batch_size=64,
                            lr=1e-3,
                            alpha=0.5,
                            device=device)


if __name__ == "__main__":
    main()
