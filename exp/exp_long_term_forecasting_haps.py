import os
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

from models import haps

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_HAPS(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_HAPS, self).__init__(args)

    def _build_model(self):
        model = haps.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    @staticmethod
    def _safe_item(x, default=0.0):
        if x is None:
            return float(default)
        if isinstance(x, (int, float)):
            return float(x)
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        return float(default)

    def _print_status(self):
        raw_model = (
            self.model.module
            if isinstance(
                self.model,
                nn.DataParallel,
            )
            else
            self.model
        )

        core = getattr(
            raw_model,
            "core",
            None,
        )

        variant = str(
            getattr(
                raw_model,
                "variant",
                getattr(
                    self.args,
                    "haps_variant",
                    "full",
                ),
            )
        )

        print(
            "========== HAPS A0-A6 Status =========="
        )

        print(
            "haps_variant               = {}"
            .format(
                variant
            )
        )

        print(
            "haps_rank                  = {}"
            .format(
                int(
                    getattr(
                        self.args,
                        "haps_rank",
                        16,
                    )
                )
            )
        )

        print(
            "haps_n_bands               = {}"
            .format(
                int(
                    getattr(
                        self.args,
                        "haps_n_bands",
                        4,
                    )
                )
            )
        )

        print(
            "A1 no_subspace             = {}"
            .format(
                int(
                    variant
                    ==
                    "no_subspace"
                )
            )
        )

        print(
            "A2 static_spectral         = {}"
            .format(
                int(
                    variant
                    ==
                    "static_spectral"
                )
            )
        )

        print(
            "A3 no_spectral             = {}"
            .format(
                int(
                    variant
                    ==
                    "no_spectral"
                )
            )
        )

        print(
            "A4 no_innovation           = {}"
            .format(
                int(
                    variant
                    ==
                    "no_innovation"
                )
            )
        )

        print(
            "A5 fixed_rank              = {}"
            .format(
                int(
                    variant
                    ==
                    "fixed_rank"
                )
            )
        )

        print(
            "A6 generic_decoder         = {}"
            .format(
                int(
                    variant
                    ==
                    "generic_decoder"
                )
            )
        )

        if (
            variant
            ==
            "generic_decoder"
            and
            core is not None
        ):
            print(
                "generic_hidden_dim          = {}"
                .format(
                    int(
                        getattr(
                            core,
                            "generic_hidden_dim",
                            0,
                        )
                    )
                )
            )

            print(
                "generic_target_head_params  = {}"
                .format(
                    int(
                        getattr(
                            core,
                            "generic_target_head_params",
                            0,
                        )
                    )
                )
            )

            print(
                "generic_actual_head_params  = {}"
                .format(
                    int(
                        getattr(
                            core,
                            "generic_actual_head_params",
                            0,
                        )
                    )
                )
            )

            print(
                "generic_parameter_gap       = {:.4%}"
                .format(
                    float(
                        getattr(
                            core,
                            "generic_parameter_gap",
                            0.0,
                        )
                    )
                )
            )

        print(
            "======================================="
        )

    def _forecast_one_batch(
        self,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        with_loss=False,
        epoch_idx_1based=None
    ):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = None if batch_x_mark is None else batch_x_mark.float().to(self.device)
        batch_y_mark = None if batch_y_mark is None else batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if with_loss:
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                y_true=batch_y,
                epoch_idx_1based=epoch_idx_1based
            )
            if isinstance(outputs, tuple) and len(outputs) == 2:
                forecast, loss_dict = outputs
            else:
                forecast, loss_dict = outputs, None
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(outputs, tuple):
                forecast = outputs[0]
            else:
                forecast = outputs
            loss_dict = None

        f_dim = -1 if self.args.features == 'MS' else 0
        forecast = forecast[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:]
        return forecast, target, loss_dict

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                forecast, target, _ = self._forecast_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, with_loss=False
                )
                loss = criterion(forecast.detach().cpu(), target.detach().cpu())
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if getattr(self.args, 'use_amp', False):
            scaler = torch.cuda.amp.GradScaler()

        self._print_status()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            meter_total = []
            meter_sup = []
            meter_pred = []
            meter_dyn = []
            meter_rank = []
            meter_gate = []
            meter_innov = []
            meter_rank_first = []
            meter_rank_last = []
            meter_band = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if getattr(self.args, 'use_amp', False):
                    with torch.cuda.amp.autocast():
                        forecast, target, loss_dict = self._forecast_one_batch(
                            batch_x, batch_y, batch_x_mark, batch_y_mark,
                            with_loss=True,
                            epoch_idx_1based=epoch + 1
                        )
                        if loss_dict is not None and "L_total" in loss_dict:
                            loss = loss_dict["L_total"]
                        else:
                            loss = criterion(forecast, target)
                else:
                    forecast, target, loss_dict = self._forecast_one_batch(
                        batch_x, batch_y, batch_x_mark, batch_y_mark,
                        with_loss=True,
                        epoch_idx_1based=epoch + 1
                    )
                    if loss_dict is not None and "L_total" in loss_dict:
                        loss = loss_dict["L_total"]
                    else:
                        loss = criterion(forecast, target)

                train_loss.append(loss.item())

                if loss_dict is not None:
                    meter_total.append(self._safe_item(loss_dict.get("L_total", loss)))
                    meter_sup.append(self._safe_item(loss_dict.get("L_sup", 0.0)))
                    meter_pred.append(self._safe_item(loss_dict.get("L_pred", 0.0)))
                    meter_dyn.append(self._safe_item(loss_dict.get("L_dyn", 0.0)))
                    meter_rank.append(self._safe_item(loss_dict.get("L_rank", 0.0)))
                    meter_gate.append(self._safe_item(loss_dict.get("L_gate", 0.0)))
                    meter_innov.append(self._safe_item(loss_dict.get("L_innov", 0.0)))
                    meter_rank_first.append(self._safe_item(loss_dict.get("rank_first", 0.0)))
                    meter_rank_last.append(self._safe_item(loss_dict.get("rank_last", 0.0)))
                    meter_band.append(self._safe_item(loss_dict.get("band_gate_mean", 0.0)))

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)

                    if loss_dict is not None:
                        print(
                            "\titers: {0}, epoch: {1} | loss: {2:.7f} | "
                            "L_sup: {3:.7f} L_pred: {4:.7f} L_dyn: {5:.7f} "
                            "L_rank: {6:.7f} L_gate: {7:.7f} L_innov: {8:.7f} "
                            "rank0: {9:.4f} rankH: {10:.4f}".format(
                                i + 1, epoch + 1,
                                self._safe_item(loss),
                                self._safe_item(loss_dict.get("L_sup", 0.0)),
                                self._safe_item(loss_dict.get("L_pred", 0.0)),
                                self._safe_item(loss_dict.get("L_dyn", 0.0)),
                                self._safe_item(loss_dict.get("L_rank", 0.0)),
                                self._safe_item(loss_dict.get("L_gate", 0.0)),
                                self._safe_item(loss_dict.get("L_innov", 0.0)),
                                self._safe_item(loss_dict.get("rank_first", 0.0)),
                                self._safe_item(loss_dict.get("rank_last", 0.0)),
                            )
                        )
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item())
                        )

                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if getattr(self.args, 'use_amp', False):
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss_avg = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            msg = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss
            )

            if len(meter_total) > 0:
                msg += (
                    " | Avg L_total: {:.7f} L_sup: {:.7f} L_pred: {:.7f} "
                    "L_dyn: {:.7f} L_rank: {:.7f} L_gate: {:.7f} L_innov: {:.7f} "
                    "rank0: {:.4f} rankH: {:.4f} band_gate: {:.4f}".format(
                        np.average(meter_total),
                        np.average(meter_sup) if len(meter_sup) > 0 else 0.0,
                        np.average(meter_pred) if len(meter_pred) > 0 else 0.0,
                        np.average(meter_dyn) if len(meter_dyn) > 0 else 0.0,
                        np.average(meter_rank) if len(meter_rank) > 0 else 0.0,
                        np.average(meter_gate) if len(meter_gate) > 0 else 0.0,
                        np.average(meter_innov) if len(meter_innov) > 0 else 0.0,
                        np.average(meter_rank_first) if len(meter_rank_first) > 0 else 0.0,
                        np.average(meter_rank_last) if len(meter_rank_last) > 0 else 0.0,
                        np.average(meter_band) if len(meter_band) > 0 else 0.0,
                    )
                )
            print(msg)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        forecasts = []
        targets = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                forecast, target, _ = self._forecast_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, with_loss=False
                )

                forecast = forecast.detach().cpu().numpy()
                target = target.detach().cpu().numpy()

                forecasts.append(forecast)
                targets.append(target)

                if i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_x[0, :, -1], target[0, :, -1]), axis=0)
                    pd = np.concatenate((input_x[0, :, -1], forecast[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        forecasts = np.concatenate(forecasts, axis=0)
        targets = np.concatenate(targets, axis=0)

        forecasts = forecasts.reshape(-1, forecasts.shape[-2], forecasts.shape[-1])
        targets = targets.reshape(-1, targets.shape[-2], targets.shape[-1])

        print('test shape:', forecasts.shape, targets.shape)

        mae, mse, rmse, mape, mspe = metric(forecasts, targets)
        print('mse:{}, mae:{}'.format(mse, mae))

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        print('skip saving pred.npy and true.npy to save disk space')

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        forecasts = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                forecast, _, _ = self._forecast_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, with_loss=False
                )
                forecasts.append(forecast.detach().cpu().numpy())

        forecasts = np.array(forecasts)
        forecasts = forecasts.reshape(-1, forecasts.shape[-2], forecasts.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', forecasts)