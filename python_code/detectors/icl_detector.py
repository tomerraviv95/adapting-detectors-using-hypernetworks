import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2Model, GPT2Config

from python_code import DEVICE
from python_code import conf
from python_code.utils.constants import DetectorUtil


def build_model(embedding_dim, n_positions, num_heads, num_layers, data_dim):
    model = TransformerModel(
        n_dims=data_dim,
        n_positions=2 * n_positions,
        n_embd=embedding_dim,
        n_layer=num_layers,
        n_head=num_heads,
    )
    return model


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd, n_layer, n_head):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, self.n_dims)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim_xs = xs_b.shape
        _, _, dim_ys = ys_b.shape
        # Check if xs_b or ys_b needs padding and pad the smaller one
        if dim_xs < dim_ys:
            padding = (0, dim_ys - dim_xs)  # Pad xs_b along the last dimension
            xs_b = F.pad(xs_b, padding)
        elif dim_ys < dim_xs:
            padding = (0, dim_xs - dim_ys)  # Pad ys_b along the last dimension
            ys_b = F.pad(ys_b, padding)
        # Stack and interleave the tensors
        zs = torch.stack((xs_b, ys_b), dim=2)
        zs = zs.view(bsize, 2 * points, max(dim_xs, dim_ys))  # Use max to get the correct new dimension
        return zs

    def forward(self, ys_batch, xs_batch, inds=None):
        if inds is None:
            inds = torch.arange(xs_batch.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= xs_batch.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(ys_batch, xs_batch)
        zs = zs.to(torch.float32)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        '''Mapping to Constallation Symbol'''
        prediction = torch.sigmoid(prediction)  # (torch.sigmoid(prediction) - 0.5) * np.sqrt(2)
        return prediction[:, ::2, :]


def mean_squared_error(xs_pred, xs):
    loss_state = xs - xs_pred
    loss_state_post = torch.norm(loss_state, dim=2)
    return (loss_state_post).square().mean()


def val_step(model, ys_batch, xs_batch, xs_real, optimizer, loss_func):
    model.eval()
    output = model(ys_batch, xs_batch)
    loss = mean_squared_error(output[:, :, :xs_real.shape[2]], xs_real)
    return loss.detach().item()


def predict_step(model, ys_batch, xs_batch):
    model.eval()
    output = model(ys_batch, xs_batch)
    return output[:, :, :xs_batch.shape[2]]


def train_step(model, ys_batch, xs_batch, xs_real, optimizer, loss_func):
    model.train()
    output = model(ys_batch, xs_batch)
    optimizer.zero_grad()
    loss = mean_squared_error(output[:, :, :xs_real.shape[2]], xs_real)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ICLDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 64
        self.num_head = 4
        self.num_layer = 2
        self.dropout = 0
        self.prompt_seq_length = 20
        self.data_dim = conf.n_ant
        self.learning_rate = 1e-4
        self.batch_size = 128
        self.epochs = 10000
        self.model_type = 'GPT2'
        self.model = build_model(embedding_dim=self.embedding_dim,
                                 n_positions=self.prompt_seq_length,
                                 num_heads=self.num_head,
                                 num_layers=self.num_layer,
                                 data_dim=self.data_dim).to(DEVICE)

    def __str__(self):
        return 'ICL Detector'

    def forward(self, rx: torch.Tensor, detector_util: DetectorUtil) -> torch.Tensor:
        reshaped_rx = rx.reshape([-1, self.prompt_seq_length, self.data_dim])
        total_batches = reshaped_rx.shape[0]
        sequence_length = reshaped_rx.shape[1]
        n_it = total_batches // self.batch_size + 1
        n_seq = sequence_length // self.prompt_seq_length + 1
        x_total = torch.zeros([reshaped_rx.shape[0], reshaped_rx.shape[1], detector_util.n_users]).int().to(DEVICE)
        with torch.no_grad():
            for i in range(n_it):
                batch_id = torch.arange(i * self.batch_size, min((i + 1) * self.batch_size, total_batches)).long()
                for j in range(n_seq):
                    seq_start = j * self.prompt_seq_length
                    seq_end = min((j + 1) * self.prompt_seq_length, sequence_length)

                    if seq_start >= seq_end:
                        continue  # Skip invalid ranges
                    # Extracting the sub-sequences for the batch
                    ys_batch = reshaped_rx[batch_id, seq_start:seq_end, :]
                    xs_batch = x_total[batch_id, seq_start:seq_end, :]
                    # Predict step (using > 0.5 threshold for binary decision)
                    output = (predict_step(self.model, ys_batch, xs_batch) > 0.5).int()
                    # Assign the output back into x_total for the respective batch and sequence
                    x_total[batch_id, seq_start:seq_end, :] = output
        return x_total.reshape([-1, detector_util.n_users])

    def train(self, x_tr: torch.Tensor, y_tr: torch.Tensor, detector_util: DetectorUtil = None, validation_split=0.2):
        x_tr = torch.stack(x_tr)
        total_batches = x_tr.shape[0]
        val_batches = int(total_batches * validation_split)
        train_batches = total_batches - val_batches

        # Split data into training and validation
        x_val, y_val = x_tr[-val_batches:], y_tr[-val_batches:]  # Validation set (20% of batches)
        x_tr, y_tr = x_tr[:train_batches], y_tr[:train_batches]  # Training set (80% of batches)
        total_batches = x_tr.shape[0]
        sequence_length = x_tr.shape[1]
        loss_function = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        n_it_per_epoch = train_batches // self.batch_size + 1
        early_stopping = EarlyStopping(patience=5)  # Set patience for early stopping
        VALIDATION_EPOCHS = 25

        for epoch in range(self.epochs):
            running_loss = 0

            # Training loop
            for i in range(n_it_per_epoch):
                # Select random sequence indices for this batch along the second dimension
                random_seq_indices = torch.tensor(
                    np.random.choice(sequence_length, self.prompt_seq_length, replace=False)).long()

                # Get the batch of data (first dim: batch size, second dim: sequence length)
                batch_id = torch.arange(i * self.batch_size, min((i + 1) * self.batch_size, total_batches)).long()

                # Select the random sequences for both x_tr and y_tr
                xs_batch = x_tr[batch_id, :, :]
                xs_batch = xs_batch[:, random_seq_indices, :]
                ys_batch = y_tr[batch_id, :, :]
                ys_batch = ys_batch[:, random_seq_indices, :]

                loss, output = train_step(self.model, ys_batch=ys_batch, xs_batch=xs_batch, xs_real=xs_batch,
                                          optimizer=optimizer, loss_func=loss_function)
                running_loss += loss / n_it_per_epoch

            # Validation step
            if epoch % VALIDATION_EPOCHS == 0:
                val_loss = self.validation_step(x_val, y_val, loss_function, optimizer)
                print(f"Validation loss for epoch {epoch}: {val_loss}")
                # Check early stopping condition
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
                self.model.train()

    def validation_step(self, x_val, y_val, loss_function, optimizer):
        val_loss = 0
        total_batches = x_val.shape[0]
        sequence_length = x_val.shape[1]

        with torch.no_grad():
            for i in range(int(total_batches)):
                # Select random sequence indices for this batch along the second dimension
                random_seq_indices = torch.tensor(
                    np.random.choice(sequence_length, self.prompt_seq_length, replace=False)).long()

                # Get the batch of data (first dim: batch size, second dim: sequence length)
                start_ind = i * self.batch_size
                end_ind = min((i + 1) * self.batch_size, total_batches)
                if start_ind > end_ind:
                    break
                batch_id = torch.arange(start_ind, end_ind).long()

                # Select the random sequences for both x_tr and y_tr
                xs_batch = x_val[batch_id, :, :]
                xs_batch = xs_batch[:, random_seq_indices, :]
                ys_batch = y_val[batch_id, :, :]
                ys_batch = ys_batch[:, random_seq_indices, :]

                val_loss += val_step(self.model, ys_batch=ys_batch, xs_batch=xs_batch, xs_real=xs_batch,
                                     optimizer=optimizer, loss_func=loss_function) / total_batches

        return val_loss

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
