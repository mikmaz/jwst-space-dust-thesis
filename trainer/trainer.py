from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from tqdm import tqdm
from abc import ABC, abstractmethod
import torch


class Trainer(ABC):
    def __init__(
            self, train_dl, val_dl, model, device, lr, lr_decay=0,
            weight_decay=0
    ):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if lr_decay > 0:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 'min', factor=lr_decay, threshold=1e-3,
                patience=10
            )
        else:
            self.scheduler = None
        print("Optimizer:", self.optimizer)
        print("Scheduler:", self.scheduler)
        print("arch list:", torch.cuda.get_arch_list())
        print("cuda version:", torch.version.cuda)

    @abstractmethod
    def loss_f(self, *args, **kwargs):
        pass

    @abstractmethod
    def mae_f(self, *args):
        pass

    @abstractmethod
    def eval_batch(self, x, y, z, x_exist):
        pass

    def evaluate(self):
        val_losses = []
        val_maes = []
        self.model.eval()
        with torch.no_grad():
            i = 0
            for x, y, z, x_exist in self.val_dl:
                i += 1
                # if i % 100 == 0:
                # print(i, 'eval', loss)
                # break
                loss, mae, skip_update = self.eval_batch(x, y, z, x_exist)
                if not skip_update:
                    val_maes.append(mae.item())
                    val_losses.append(loss.item())
        mean_loss = sum(val_losses[:-1]) / len(val_losses[:-1])
        mean_mae = sum(val_maes[:-1]) / len(val_maes[:-1])
        return mean_loss, mean_mae

    def train(self, n_epochs, stats_path, grad_clip=0):
        with open(f'{stats_path}/losses.txt', 'w+') as f:
            f.write(
                "training loss,training mae,validation loss,validation mae\n"
            )
        best_val_loss = None
        with tqdm(range(n_epochs)) as pbar:
            for epoch in pbar:
                epoch_losses = []
                epoch_maes = []
                i = 0
                for x, y, z, x_exist in self.train_dl:
                    i += 1
                    # if i % 10 == 0:
                    #     print(i, loss)
                    # break
                    self.model.train()
                    self.optimizer.zero_grad()
                    loss, mae, skip_update = self.eval_batch(x, y, z, x_exist)
                    if not skip_update:
                        epoch_maes.append(mae.item())
                        epoch_losses.append(loss.item())
                        loss.backward()
                        # torch.nn.utils.clip_grad_value_(
                        # model.parameters(), 1e5
                        # )
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                grad_clip
                            )
                        self.optimizer.step()

                if (epoch + 1) % max(1, (n_epochs // 5)) == 0:
                    torch.save(
                        self.model.state_dict(),
                        f"{stats_path}/checkpoint/model_{epoch}.pt"
                    )
                    torch.save(
                        self.optimizer.state_dict(),
                        f"{stats_path}/checkpoint/optim_{epoch}.pt"
                    )
                epoch_loss = sum(epoch_losses[:-1]) / len(epoch_losses[:-1])
                print(f'Epoch: {epoch}, train loss: {epoch_loss}')
                epoch_stats = [
                    epoch_loss, sum(epoch_maes[:-1]) / len(epoch_maes[:-1])
                ]

                val_loss, val_mse = self.evaluate()
                if self.scheduler is not None:
                    self.scheduler.step(val_mse)
                if best_val_loss is None or best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        self.model.state_dict(),
                        f"{stats_path}/checkpoint/best_model.pt"
                    )
                stats_str = ','.join(
                    [str(stat) for stat in epoch_stats + [val_loss, val_mse]]
                )
                with open(f'{stats_path}/losses.txt', 'a+') as f:
                    f.write(stats_str + '\n')
