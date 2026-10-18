"""A–G history integrations preserving the original controlled RoPE backbone."""
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.controlled_model import ControlledDailyPriceTransformer
from src.constraints import project_hourly
from src.features import structured_hourly_features, structured_numerical_features
from src.interpolation import makima_residual
from src.temporal import absolute_time_features
from .encoders import HistoryEncoder, HourShapeEncoder
from .rope_integration import CrossAdapter, concatenated_block, history_positions


class RopeContextTransformer(ControlledDailyPriceTransformer):
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 ff_dim: int = 128, dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 normalization: dict[str, Any] | None = None,
                 conditioning: dict[str, Any] | None = None,
                 position: dict[str, Any] | None = None,
                 teacher_forcing: dict[str, Any] | None = None,
                 tokenization: dict[str, Any] | None = None,
                 prediction: dict[str, Any] | None = None,
                 interpolation: dict[str, Any] | None = None,
                 local_features: dict[str, Any] | None = None,
                 control: dict[str, Any] | None = None,
                 integration: dict[str, str] | None = None) -> None:
        integration = {'architecture': 'BASE', 'encoder': 'none', 'injection': 'additive',
                       'representation': 'correction', 'registers': '4', **(integration or {})}
        if set(integration) - {'architecture', 'encoder', 'injection', 'representation', 'registers',
                              'hour_aggregation', 'output_mode', 'context_input', 'confidence_gate'}:
            raise ValueError('Unknown integration configuration fields')
        architecture = integration['architecture']
        shape = integration.get('hour_aggregation', 'mean') == 'shape'
        hour_gate = integration.get('output_mode', 'gate_delta') == 'hour_gate'
        if integration.get('context_input', 'prices') not in ('prices', 'correction'):
            raise ValueError('Unknown historical input type')
        if integration.get('confidence_gate', 'disabled') not in ('disabled', 'enabled'):
            raise ValueError('Unknown confidence gate policy')
        if integration.get('context_input') == 'correction' and not (architecture == 'F' or architecture == 'E' and shape):
            raise ValueError('Direct retrieved corrections require F or shape-preserving E')
        if integration.get('confidence_gate') == 'enabled' and integration.get('context_input') != 'correction':
            raise ValueError('Confidence gating requires explicit retrieved corrections')
        if integration.get('hour_aggregation', 'mean') not in ('mean', 'shape'):
            raise ValueError('Unknown hourly aggregation')
        if integration.get('output_mode', 'gate_delta') not in ('gate_delta', 'hour_gate'):
            raise ValueError('Unknown output conditioning mode')
        if (shape and architecture != 'E') or (hour_gate and architecture != 'G'):
            raise ValueError('Shape aggregation is E-only; hour_gate is G-only')
        if architecture not in ('BASE', 'A', 'B', 'C', 'D', 'E', 'F', 'G'):
            raise ValueError('Unknown context architecture')
        if integration['encoder'] not in ('none', 'linear', 'cnn') or (architecture == 'BASE') != (integration['encoder'] == 'none'):
            raise ValueError('BASE uses none; history architectures use linear or cnn')
        if integration['injection'] not in ('additive', 'concat') or integration['representation'] not in ('correction', 'residual'):
            raise ValueError('Invalid injection or historical representation')
        if int(integration['registers']) != 4:
            raise ValueError('The first register comparison fixes K=4')
        if teacher_forcing and any(float(value) != 0 for value in teacher_forcing.values()):
            raise ValueError('Current-day target revelation is disabled in context comparisons')
        super().__init__(d_model, n_heads, n_layers, ff_dim, dropout, lr, weight_decay,
                         normalization, conditioning, position, {'sample_probability': 0., 'hour_probability': 0.},
                         tokenization, prediction, interpolation, local_features, control)
        if self.control_config['encoder'] != 'transformer':
            raise ValueError('Context integrations require the original Transformer')
        if architecture == 'F' and self.tokenization_config['mode'] != 'masked':
            raise ValueError('Quarter-aligned F is confined to the original 96-slot models')
        self.integration_config = integration
        self.context_config = integration
        self.hparams['integration'] = dict(integration)
        self.history_encoder = None
        self.adapters = nn.ModuleList()
        self.injection = self.register_attention = self.output_gate = self.output_delta = None
        self.confidence_gate = None
        self.register_parameter('register_queries', None)
        self.register_parameter('history_metadata', None)
        # Preserve base weights AND the caller's RNG stream. A seed-matched BASE
        # has exactly the original constructor state and training dropout draws.
        if architecture != 'BASE':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed() + 1701)
                self.history_encoder = (HourShapeEncoder(integration['encoder'], d_model, dropout)
                                        if shape or hour_gate else
                                        HistoryEncoder(integration['encoder'], d_model, n_heads, ff_dim, dropout))
                self.history_metadata = nn.Parameter(torch.zeros(2, d_model))
                if architecture == 'F':
                    self.injection = nn.Linear(d_model if integration['injection'] == 'additive' else 2*d_model,
                                               d_model, bias=False)
                    nn.init.zeros_(self.injection.weight)
                    if integration['injection'] == 'concat':
                        with torch.no_grad():
                            self.injection.weight[:, :d_model].copy_(torch.eye(d_model))
                elif architecture in ('B', 'C'):
                    self.adapters = nn.ModuleList([CrossAdapter(d_model, n_heads) for _ in range(n_layers)])
                    if architecture == 'C':
                        self.register_queries = nn.Parameter(torch.randn(4, d_model) * .02)
                        self.register_attention = nn.MultiheadAttention(d_model, n_heads, dropout=0., batch_first=True)
                elif architecture in ('D', 'E'):
                    self.adapters = nn.ModuleList([nn.Linear(d_model, d_model if architecture == 'D' else 2*d_model)
                                                   for _ in range(n_layers)])
                    for adapter in self.adapters:
                        nn.init.zeros_(adapter.weight)
                        nn.init.zeros_(adapter.bias)
                elif architecture == 'G':
                    self.output_gate = nn.Linear(d_model, 1)
                    if not hour_gate:
                        self.output_delta = nn.Linear(d_model, 1)
                    for module in (self.output_gate, self.output_delta):
                        if module is None:
                            continue
                        nn.init.zeros_(module.weight)
                        nn.init.zeros_(module.bias)
                if integration.get('confidence_gate') == 'enabled':
                    self.confidence_gate = nn.Linear(3, 1)
                    nn.init.zeros_(self.confidence_gate.weight)
                    nn.init.zeros_(self.confidence_gate.bias)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if checkpoint['hyper_parameters'].get('integration') != self.integration_config:
            raise RuntimeError('Checkpoint context integration differs')

    def history_features(self, hourly, history, available):
        if history is None or available is None:
            history = hourly.new_zeros(hourly.shape[0], 96)
            available = torch.zeros(hourly.shape[0], dtype=torch.bool, device=hourly.device)
        if history.shape != (hourly.shape[0], 96) or available.shape != (hourly.shape[0],):
            raise ValueError('History must be [B,96], availability [B]')
        available = available.bool()
        history = torch.where(available[:, None], history, torch.zeros_like(history))
        if not torch.isfinite(history).all():
            raise ValueError('Available history must be finite')
        if self.integration_config.get('context_input') == 'correction':
            _, scale = self.normalizer.statistics(hourly)
            signal = history.reshape(-1, 24, 4)
            signal = (signal-signal.mean(-1, keepdim=True)).reshape(-1, 96)
            zeros = torch.zeros_like(signal)
            features = torch.stack((signal/scale, zeros, zeros, available[:, None].expand(-1, 96)), -1)
            return features * available[:, None, None], available
        previous = history.reshape(-1, 24, 4).mean(-1)
        signal = history - previous.repeat_interleave(4, 1)
        if self.integration_config['representation'] == 'correction':
            signal = signal - makima_residual(previous, 2.)
        _, scale = self.normalizer.statistics(hourly)
        _, previous_scale = self.normalizer.statistics(previous)
        summaries = torch.cat((torch.log(previous_scale / scale),
                               (previous.mean(1, keepdim=True) - hourly.mean(1, keepdim=True)) / scale), -1)
        features = torch.cat(((signal / previous_scale)[..., None], summaries[:, None].expand(-1, 96, -1),
                              available[:, None, None].expand(-1, 96, 1)), -1)
        return features * available[:, None, None], available

    def _history_time(self, hourly):
        time = absolute_time_features(96, [1, 2], device=hourly.device, dtype=hourly.dtype)
        if self.time_projection is not None:
            return self.time_projection(time)[None]
        # B1's original all19 packet encodes hour starts (not centers). Keep that
        # legacy phase origin: quarter center minus 0.5 physical hour. Share the
        # very same four projection columns, without a new current time encoder.
        t = (torch.arange(96, device=hourly.device, dtype=hourly.dtype) + .5) / 4 - .5
        phase = torch.stack((2*torch.pi*t/24, 4*torch.pi*t/24), -1)
        time = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        return F.linear(time, self.control_feature_projection.weight[:, 13:17])[None]

    def _history(self, hourly, history, available, confidence=None):
        features, available = self.history_features(hourly, history, available)
        time = self._history_time(hourly) + self.history_metadata.sum(0)
        # Historical dropout has its own draw scope, preserving the main model's
        # RNG stream for matched-seed comparison with the unchanged backbone.
        devices = [hourly.device.index] if hourly.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            encoded = self.history_encoder(features, time, available)
        if self.confidence_gate is not None:
            if confidence is None or confidence.shape != (len(hourly), 24, 3):
                raise ValueError('Retrieval confidence must have shape [B,24,3]')
            confidence = torch.where(available[:, None, None], confidence, torch.zeros_like(confidence))
            if not torch.isfinite(confidence).all() or (confidence < 0).any() or (confidence[..., 2] > 1).any():
                raise ValueError('Invalid retrieval confidence')
            _, scale = self.normalizer.statistics(hourly)
            gate_features = torch.stack((confidence[..., 0].log1p(),
                                         (confidence[..., 1]/scale).log1p(), confidence[..., 2]), -1)
            gate = self.confidence_gate(gate_features).sigmoid() * available[:, None, None]
            encoded = encoded * (gate.repeat_interleave(4, 1) if encoded.shape[1] == 96 else gate)
        return encoded, features, available

    def _current_embedding(self, hourly, zone, weekday, start_hour):
        # This is the original controlled model's input path, kept in the same
        # arithmetic order. Tests compare its zero-adapter outputs and base state.
        tokens, types = self.build_tokens(hourly, zone, weekday)
        if self.time_projection is not None:
            time = absolute_time_features(tokens.shape[1], self.position_config['absolute_time']['harmonics'],
                                          device=tokens.device, dtype=tokens.dtype, start_hour=start_hour)
            projected = self.time_projection(time)
            tokens = tokens + (projected if start_hour is not None else projected[None])
        feature_set = self.control_config['feature_set']
        if feature_set == 'local4':
            return tokens, types
        global_scale = float(self.normalizer.global_std.detach().cpu())
        features = (structured_numerical_features(hourly, global_scale) if feature_set == 'numerical11'
                    else structured_hourly_features(hourly, zone, global_scale))
        projected = self.control_feature_projection(features)
        x = projected if self.tokenization_config['mode'] == 'hourly' else tokens + projected.repeat_interleave(4, 1)
        if self.tokenization_config['mode'] == 'hourly' and self.time_projection is not None:
            projected_time = self.time_projection(time)
            x = x + (projected_time if start_hour is not None else projected_time[None])
        return x, types

    def _encode_context(self, hourly, zone, weekday, start_hour, hist, available):
        x, _ = self._current_embedding(hourly, zone, weekday, start_hour)
        condition = self._condition(hourly, zone, weekday)
        architecture = self.integration_config['architecture']
        length = x.shape[1]
        if architecture == 'F':
            if self.integration_config['injection'] == 'additive':
                x = x + self.injection(hist) * available[:, None, None]
            else:
                injected = self.injection(torch.cat((x, hist), -1))
                x = torch.where(available[:, None, None], injected, x)
        memory = hist
        if architecture == 'C':
            queries = self.register_queries[None].expand(x.shape[0], -1, -1)
            memory = self.register_attention(queries, hist, hist, need_weights=False)[0] * available[:, None, None]
        if architecture == 'A':
            positions = torch.cat((history_positions(length, x.device), torch.arange(length, device=x.device)))
            valid = torch.cat((available[:, None].expand(-1, 96), torch.ones(x.shape[0], length, dtype=torch.bool, device=x.device)), 1)
            x = torch.cat((hist, x), 1)
        for index, block in enumerate(self.blocks):
            if architecture == 'A':
                x = concatenated_block(block, x, condition, positions, valid)
                continue
            block_condition = condition
            if architecture == 'D':
                block_condition = condition + self.adapters[index](hist.mean(1)) * available[:, None]
            if architecture == 'E':
                hourly_hist = (hist if self.integration_config.get('hour_aggregation') == 'shape' else
                               hist.reshape(-1, 24, 4, self.d_model).mean(2))
                film = self.adapters[index](hourly_hist) * available[:, None, None]
                if length == 96:
                    film = film.repeat_interleave(4, 1)
                gamma, beta = film.chunk(2, -1)
                x = x * (1 + gamma) + beta
            x = block(x, block_condition)
            if architecture in ('B', 'C'):
                x = self.adapters[index](x, memory, available)
        x = x[:, -length:]
        shift, scale_mod = self.final_modulation(condition).chunk(2, -1)
        return self.final_norm(x) * (1 + scale_mod[:, None]) + shift[:, None]

    def _decode_context(self, x, hourly):
        _, price_scale = self.normalizer.statistics(hourly)
        if self.control_config['head'] == 'centered':
            raw = self.head(x)
            if self.tokenization_config['mode'] == 'masked':
                raw = raw.squeeze(-1)
            if raw.ndim == 3:
                raw = raw.reshape(-1, 96)
            pred = project_hourly(raw, hourly, price_scale)
            if self.interpolation_config['enabled']:
                pred = pred + makima_residual(hourly, float(self.interpolation_config['amplitude']))
                q = pred.reshape(-1, 24, 4)
                pred = (q + (hourly.unsqueeze(-1) - q.mean(-1, keepdim=True))).reshape(-1, 96)
        else:
            residual = self.control_signed_head(x)
            pred_q = hourly.unsqueeze(-1) + price_scale.unsqueeze(-1) * residual
            if self.interpolation_config['enabled']:
                pred_q = pred_q + makima_residual(hourly, float(self.interpolation_config['amplitude'])).reshape(-1, 24, 4)
            pred_q = pred_q + hourly.unsqueeze(-1) - pred_q.mean(-1, keepdim=True)
            pred = pred_q.reshape(-1, 96)
        return pred

    def forward(self, hourly, zone=None, weekday=None, start_hour=None, history=None, history_available=None,
                history_confidence=None):
        self._validate_start_hour(start_hour, hourly.shape[0])
        if start_hour is not None and torch.any(start_hour != 0):
            raise ValueError('History integrations require calendar-day windows')
        architecture = self.integration_config['architecture']
        if architecture == 'BASE':
            return super().forward(hourly, zone, weekday, start_hour)
        if architecture == 'G':
            base = super().forward(hourly, zone, weekday, start_hour)
            hist, features, available = self._history(hourly, history, history_available)
            if self.integration_config.get('output_mode') == 'hour_gate':
                gate = self.output_gate(hist).tanh()
                _, scale = self.normalizer.statistics(hourly)
                previous_scale = scale * features[:, 0, 1:2].exp()
                correction = gate * features[..., 0].reshape(-1, 24, 4) * previous_scale.unsqueeze(-1)
                correction = correction - correction.mean(-1, keepdim=True)
                return base + (correction * available[:, None, None]).reshape(-1, 96)
            gate = self.output_gate(hist.reshape(-1, 24, 4, self.d_model).mean(2)).tanh()
            correction = gate * features[..., 0].reshape(-1, 24, 4) + self.output_delta(hist).reshape(-1, 24, 4)
            correction = correction - correction.mean(-1, keepdim=True)
            _, scale = self.normalizer.statistics(hourly)
            return base + (scale.unsqueeze(-1) * correction * available[:, None, None]).reshape(-1, 96)
        hist, _, available = self._history(hourly, history, history_available, history_confidence)
        x = self._encode_context(hourly, zone, weekday, start_hour, hist, available)
        return self._decode_context(x, hourly)

    def batch_prediction(self, batch):
        return self(batch['hourly'], batch.get('zone'), batch.get('weekday'), batch.get('start_hour'),
                    batch.get('history'), batch.get('history_available'), batch.get('history_confidence'))

    def training_step(self, batch, batch_idx):
        loss = (self.batch_prediction(batch) - batch['target']).abs().mean()
        self.log('train/mae', loss, on_step=False, on_epoch=True, batch_size=len(batch['hourly']))
        return loss

    def _metric_update(self, metric, batch):
        pred = self.batch_prediction(batch)
        metric.update(pred, batch['target'], batch['delivery_date'], batch['zone'], hourly=batch['hourly'])
        return {'predictions': pred, 'hourly': batch.get('hourly_original', batch['hourly']),
                'zone': batch['zone'], 'delivery_date': batch['delivery_date'], 'target': batch['target']}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return {'predictions': self.batch_prediction(batch), 'hourly': batch.get('hourly_original', batch['hourly']),
                'zone': batch['zone'], 'delivery_date': batch['delivery_date']}

    def approximate_flops(self):
        width, ff = self.hparams.d_model, self.hparams.ff_dim
        n = 96 if self.tokenization_config['mode'] == 'masked' else 24
        architecture = self.integration_config['architecture']
        total = self.hparams.n_layers * (8*n*width**2 + 4*n*n*width + 4*n*width*ff)
        if architecture == 'A':
            m = n + 96
            total = self.hparams.n_layers * (8*m*width**2 + 4*m*m*width + 4*m*width*ff)
        if self.history_encoder is not None:
            history_n = (24 if self.integration_config.get('hour_aggregation') == 'shape' or
                         self.integration_config.get('output_mode') == 'hour_gate' else 96)
            total += 2*history_n*4*width
        if self.integration_config['encoder'] == 'cnn':
            total += 2*4*history_n*width*width*5
        if architecture in ('B', 'C'):
            k = 96 if architecture == 'B' else 4
            total += self.hparams.n_layers * (4*(n+k)*width**2 + 4*n*k*width)
            if architecture == 'C':
                total += 4*(4+96)*width**2 + 4*4*96*width
        if architecture == 'E':
            total += self.hparams.n_layers * 4*24*width**2
        if architecture == 'D':
            total += self.hparams.n_layers * 2*width**2
        if architecture == 'G':
            total += 2*(24 if self.output_delta is None else 24+96)*width
        if architecture == 'F':
            total += 2*n*width**2*(2 if self.integration_config['injection'] == 'concat' else 1)
        if self.confidence_gate is not None:
            total += 2*24*3
        return int(total)
