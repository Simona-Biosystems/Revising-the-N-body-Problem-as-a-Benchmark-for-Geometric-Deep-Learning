import datetime
import json
import os
import time

import torch
import torchmetrics
from schedulefree import AdamWScheduleFree
from torch.optim.lr_scheduler import LambdaLR

import wandb
from inferencer import Inferencer
from utils.nbody_utils import get_device
from utils.utils_data import calculate_energies


class Trainer:
    def __init__(self, model, dataloader, args) -> None:
        self.args = args
        self.device = get_device(self.args.gpu_id)
        self.model = model.to(self.device)
        if self.args.double_precision:
            self.model = self.model.double()
        else:
            self.model = self.model.float()
        self.dataloader = dataloader
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler()
        self.criterion_pos = torch.nn.MSELoss()
        self.criterion_vel = torch.nn.MSELoss()
        self.criterion_energy = torch.nn.MSELoss()
        self.criterion_macros = torch.nn.MSELoss()
        self.criterion_com = torch.nn.MSELoss()

        self.loss_metric = torchmetrics.MeanMetric().to(self.device)
        self.mae_pos_metric = torchmetrics.MeanMetric().to(self.device)
        self.mae_vel_metric = torchmetrics.MeanMetric().to(self.device)
        self.mae_energy_metric = torchmetrics.MeanMetric().to(self.device)
        self.pos_percentage_error_metric = torchmetrics.MeanMetric().to(self.device)
        self.vel_percentage_error_metric = torchmetrics.MeanMetric().to(self.device)
        self.energy_percentage_error_metric = torchmetrics.MeanMetric().to(self.device)
        self.com_percentage_error_metric = torchmetrics.MeanMetric().to(self.device)

        self.training_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self.args.model_path:
            self.save_dir_path = (
                f"{os.path.dirname(self.args.model_path)}/{self.training_start_time}"
            )
            checkpoint = torch.load(self.args.model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Loaded model and optimizer state from {self.args.model_path}")

        else:
            self.save_dir_path = f"runs/ponita/{self.training_start_time}"

        self.num_neighbors = (
            self.args.num_neighbors
            if self.args.num_neighbors is not None
            else self.args.num_atoms - 1
        )
        self.inferencer = Inferencer(self.model, self.dataloader, self.args)

    def create_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=1e-8,
            lr=0.5,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    def create_lr_scheduler(self):
        return LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: self._rate(
                step,
                factor=self.args.lr_factor,
                warmup=100,
            ),
        )

    def _rate(self, step, factor, warmup):
        if step == 0:
            step = 1
        return factor * (
            self.model.get_model_size() ** (-0.5)
            * min(step ** (-0.5), step * warmup ** (-1.5))
        )

    def train_one_epoch(self, step_count):
        self.model.train()

        data, batch = self.dataloader.get_batch()
        data = self.dataloader.preprocess_batch(data)
        t0 = time.time()
        pred, loss = self.forward_pass(data, batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO: This may actually slow down training. Consider removing.
        # (Although the measurements will be less precise)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000

        self.update_metrics(pred, data, loss)
        self.log_results(step_count, dt)

        return self.model

    def centre_of_mass_loss(self, graph, pred, weight=1):
        """
        Calculate centre of mass loss for both predicted and gt step,
        then calculate loss.
        """
        batch_size = self.dataloader.dataset.batch_size
        num_atoms = self.dataloader.dataset.num_nodes
        pos_pred = graph.pos + pred[..., :3]
        pos_true = graph.pos + graph.y[..., :3]
        com_pred = []
        com_true = []
        for sim_idx in range(batch_size):
            com_pred.append(
                pos_pred[sim_idx * num_atoms : (sim_idx + 1) * num_atoms].mean(axis=0)
            )
            com_true.append(
                pos_true[sim_idx * num_atoms : (sim_idx + 1) * num_atoms].mean(axis=0)
            )
        com_pred = torch.stack(com_pred)
        com_true = torch.stack(com_true)
        loss_com = self.criterion_com(com_pred, com_true)
        err_com = com_pred - com_true
        percentage_error = (err_com / com_true).mean() * 100
        self.com_percentage_error_metric.update(torch.mean(torch.abs(percentage_error)))

        return loss_com * weight

    def energy_loss(self, predicted_pos, target_pos, predicted_vel, target_vel, mass):
        pred_energy = calculate_energies(
            predicted_pos,
            predicted_vel,
            mass,
            self.dataloader.dataset.num_nodes,
            self.dataloader.dataset.simulation.interaction_strength,
            self.dataloader.dataset.simulation.softening,
        )
        target_energy = calculate_energies(
            target_pos,
            target_vel,
            mass,
            self.dataloader.dataset.num_nodes,
            self.dataloader.dataset.simulation.interaction_strength,
            self.dataloader.dataset.simulation.softening,
        )
        loss_energy = self.criterion_energy(
            torch.tensor(pred_energy), torch.tensor(target_energy)
        )
        return loss_energy

    def forward_pass(
        self,
        graph,
        batch,
    ):
        pred = self.model(graph)
        loss_pos = self.criterion_pos(pred[..., :3], graph.y[..., :3])
        loss_vel = self.criterion_vel(pred[..., 3:], graph.y[..., 3:])
        loss = loss_pos + loss_vel
        if self.args.com_loss:
            loss_com = self.centre_of_mass_loss(graph, pred)
            loss += loss_com

        if self.args.energy_loss:
            loss_energy = self.energy_loss(
                predicted_pos=(graph.pos + pred[..., :3]).detach().cpu().numpy(),
                target_pos=(graph.pos + graph.y[..., :3]).detach().cpu().numpy(),
                predicted_vel=pred[..., 3:].detach().cpu().numpy(),
                target_vel=graph.y[..., 3:].detach().cpu().numpy(),
                mass=graph.mass.detach().cpu().numpy(),
            )
            loss += loss_energy
        return pred, loss

    def update_metrics(self, pred, graph, loss):
        self.loss_metric.update(loss)
        pred_pos = pred[..., :3]
        pred_vel = pred[..., 3:]

        target_pos = graph.y[..., :3]
        target_vel = graph.y[..., 3:]
        if self.args.energy_loss:
            pred_energy = calculate_energies(
                pred[..., :3].detach().cpu().numpy(),
                pred[..., 3:].detach().cpu().numpy(),
                graph.mass.detach().cpu().numpy(),
                self.dataloader.dataset.num_nodes,
                self.dataloader.dataset.simulation.interaction_strength,
                self.dataloader.dataset.simulation.softening,
            )
            target_energy = calculate_energies(
                target_pos.detach().cpu().numpy(),
                target_vel.detach().cpu().numpy(),
                graph.mass.detach().cpu().numpy(),
                self.dataloader.dataset.num_nodes,
                self.dataloader.dataset.simulation.interaction_strength,
                self.dataloader.dataset.simulation.softening,
            )
            err_energy = pred_energy - target_energy
            percentage_error = (err_energy / target_energy).mean() * 100
            self.energy_percentage_error_metric.update(
                torch.mean(torch.abs(torch.tensor(percentage_error)))
            )
            self.mae_energy_metric.update(
                torch.mean(torch.abs(torch.tensor(err_energy)))
            )

        err_pos = pred_pos.detach() - target_pos
        percentage_error = (err_pos / target_pos).mean() * 100
        self.pos_percentage_error_metric.update(torch.mean(torch.abs(percentage_error)))
        self.mae_pos_metric.update(torch.mean(torch.abs(err_pos)))

        err_vel = pred_vel.detach() - target_vel
        percentage_error = (err_vel / target_vel).mean() * 100
        self.vel_percentage_error_metric.update(torch.mean(torch.abs(percentage_error)))
        self.mae_vel_metric.update(torch.mean(torch.abs(err_vel)))

    def create_run_folder(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir_path = f"runs/ponita/{current_time}"
        os.makedirs(save_dir_path, exist_ok=True)
        return save_dir_path

    def save_training_args(self, save_dir_path):
        model_save_path = f"{save_dir_path}/{self.args.dataset_name}_best_model.pth"
        with open(
            os.path.join(os.path.dirname(model_save_path), "training_args.json"), "w"
        ) as f:
            args_dict = (
                vars(self.args) if not isinstance(self.args, dict) else self.args
            )
            json.dump({"args": args_dict}, f, indent=4)

    def save_model_params(self, save_dir_path):
        with open(f"{save_dir_path}/model_params.json", "w") as f:
            json.dump(self.model.get_serializable_attributes(), f, indent=4)

    def save_dataset_attributes(self, save_dir_path):
        dataset_save_path = f"{save_dir_path}/{self.args.dataset_name}_dataset"
        os.makedirs(dataset_save_path, exist_ok=True)
        attrs = self.dataloader.dataset.get_serializable_attributes()
        print(f"Training with attrs: {attrs}")
        # save dataset attributes to json
        with open(f"{dataset_save_path}/metadata.json", "w") as f:
            json.dump(attrs, f, indent=4)

    def create_wandb_run(self, save_dir_path):
        run = wandb.init(
            project=f"PONITA",
            config=self.args,
            name=self.args.run_name or save_dir_path,
        )
        return run

    def save_model(self, save_path=None):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if save_path is None:
            save_path = self.save_dir_path

        os.makedirs(save_path, exist_ok=True)
        torch.save(checkpoint, f"{save_path}/model.pth")
        print(f"Model and optimizer state saved to {save_path}")

    def log_results(self, count, dt):
        log_dict = {
            f"loss": self.loss_metric.compute().item(),
            f"MAE pos": self.mae_pos_metric.compute().item(),
            f"pos percentage error": self.pos_percentage_error_metric.compute().item(),
        }

        if "vel" in self.args.target:
            log_dict[f"MAE vel"] = self.mae_vel_metric.compute().item()
            log_dict[f"vel percentage error"] = (
                self.vel_percentage_error_metric.compute().item()
            )

        if self.args.energy_loss:
            log_dict[f"MAE energy"] = self.mae_energy_metric.compute().item()
            log_dict[f"energy percentage error"] = (
                self.energy_percentage_error_metric.compute().item()
            )

        if self.args.com_loss:
            log_dict[f"percent com"] = self.com_percentage_error_metric.compute().item()

        log_dict = self.modify_log_dict(log_dict)

        wandb.log(log_dict)

        print_str = (
            f"Step: {count + 1}/∞ | Loss: {self.loss_metric.compute().item():.5f} | "
            f"percent pos: {self.pos_percentage_error_metric.compute().item():.2f} | "
        )

        if "vel" in self.args.target:
            print_str += f"percent vel: {self.vel_percentage_error_metric.compute().item():.2f} | "

        if self.args.energy_loss:
            print_str += f"percent energy: {self.energy_percentage_error_metric.compute().item():.2f} | "

        if self.args.com_loss:
            print_str += f"percent com: {self.com_percentage_error_metric.compute().item():.5f} | "

        print_str += (
            f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
            f"{dt:.2f} ms @ {((self.args.batch_size * self.args.num_atoms) / (dt/1000)):.2f} ex/s"
        )

        print_str = self.modify_print_str(print_str)

        print(print_str)

    def modify_log_dict(self, log_dict):
        return log_dict

    def modify_print_str(self, print_str):
        return print_str

    def train(self):
        save_dir_path = self.create_run_folder()
        self.save_training_args(save_dir_path)
        self.save_model_params(save_dir_path)
        self.save_dataset_attributes(save_dir_path)
        run = self.create_wandb_run(save_dir_path)

        step_count = 0
        train_steps = self.args.train_steps
        while train_steps is None or step_count < train_steps:
            try:
                self.train_one_epoch(step_count)
                step_count += 1

                if step_count % self.args.save_model_every == 0:
                    self.save_model()

                if step_count % self.args.test_macros_every == 0:
                    self.inferencer.run_inference(
                        f"{self.save_dir_path}/checkpoints/{step_count}"
                    )

            except KeyboardInterrupt:
                print("Training interrupted. Saving model...")
                self.save_model()
                break

            except Exception as e:
                print(e)
                self.save_model()
                run.alert(
                    title="Training crashed",
                    text=f"Exception: {e}",
                )
                raise (e)
