import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

# ==================== Utility Functions ====================

def scale_zeropoint(rmin, rmax, qmin, qmax, asymmetric=True):
    """Calculate scale and zero point for quantization."""
    if asymmetric:
        scale = (rmax - rmin) / (qmax - qmin)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zeropoints = torch.round(qmin - rmin / scale)
        zeropoints = torch.clamp(zeropoints, qmin, qmax)
    else:
        absmax = torch.max(rmin.abs(), rmax.abs())
        scale = absmax / ((qmax - qmin) / 2)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zeropoints = torch.zeros_like(scale)
    return scale, zeropoints


def scale_zero_channel(weight_tensor, qmin, qmax, asymmetric=True):
    """Calculate per-channel scale and zero point for weights."""
    rmin = weight_tensor.view(weight_tensor.shape[0], -1).min(dim=1)[0]
    rmax = weight_tensor.view(weight_tensor.shape[0], -1).max(dim=1)[0]
    
    if asymmetric:
        scales = (rmax - rmin) / (qmax - qmin)
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        zeropoints = torch.round(qmin - rmin / scales)
        zeropoints = torch.clamp(zeropoints, qmin, qmax)
    else:
        absmax = torch.max(rmin.abs(), rmax.abs())
        scales = absmax / ((qmax - qmin) / 2)
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        zeropoints = torch.zeros_like(scales)
    
    return scales, zeropoints


def fold_bn_into_conv(conv, bn):
    """Fold batch normalization into convolution layer."""
    W = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(W.size(0), device=W.device)
    
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    std = torch.sqrt(var + eps)
    W_folded = W * (gamma / std).reshape(-1, 1, 1, 1)
    b_folded = beta + (b - mean) * (gamma / std)
    
    conv.weight.data = W_folded
    if conv.bias is None:
        conv.bias = nn.Parameter(b_folded)
    else:
        conv.bias.data = b_folded
    
    return conv


def bnfold_vgg(model, device="cuda"):
    """Fold all batch normalization layers in VGG model."""
    model = model.to(device)
    layers = list(model.features)
    new_layers = []
    i = 0
    
    while i < len(layers):
        if (isinstance(layers[i], nn.Conv2d) and 
            i + 1 < len(layers) and 
            isinstance(layers[i + 1], nn.BatchNorm2d)):
            conv_folded = fold_bn_into_conv(layers[i], layers[i + 1])
            new_layers.append(conv_folded)
            i += 2
        else:
            new_layers.append(layers[i])
            i += 1
    
    model.features = nn.Sequential(*new_layers)
    return model


# ==================== Packing/Unpacking for INT4 ====================

def pack_int4(x: torch.Tensor) -> torch.ByteTensor:
    """Pack int8 tensor into int4 format (2 values per byte)."""
    assert x.dtype == torch.int8
    assert torch.all((x >= -8) & (x <= 7)), "Values must be in [-8, 7] range for int4"
    
    flat = x.flatten()
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8, device=x.device)])
    
    flat_unsigned = (flat & 0x0F).to(torch.uint8)
    high = flat_unsigned[0::2] << 4
    low = flat_unsigned[1::2]
    packed = high | low
    
    return packed


def unpack_int4(packed: torch.ByteTensor, original_shape) -> torch.Tensor:
    """Unpack int4 format back to int8 tensor."""
    assert packed.dtype == torch.uint8
    
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    vals = torch.stack((high, low), dim=-1).flatten()
    vals = vals.to(torch.int8)
    vals = (vals ^ 0x08) - 0x08
    
    return vals[:torch.prod(torch.tensor(original_shape))].view(original_shape)


# ==================== Calibration for PTQ ====================

class ActivationObserver:
    """Observer to collect activation statistics for PTQ."""
    def __init__(self, qmin, qmax):
        self.qmin = qmin
        self.qmax = qmax
        self.min_val = None
        self.max_val = None
    
    def update(self, tensor):
        """Update min/max statistics."""
        batch_min = tensor.min()
        batch_max = tensor.max()
        
        if self.min_val is None:
            self.min_val = batch_min
            self.max_val = batch_max
        else:
            self.min_val = torch.min(self.min_val, batch_min)
            self.max_val = torch.max(self.max_val, batch_max)
    
    def get_scale_zeropoint(self):
        """Calculate scale and zero point from collected statistics."""
        return scale_zeropoint(self.min_val, self.max_val, self.qmin, self.qmax)


# ==================== (NEW) Robust Calibration and PTQ Application ====================

def calibrate_model(model, dataloader, device, num_batches=100, bitwidth=8):
    model.eval()
    
    qmin = -2**(bitwidth - 1)
    qmax = 2**(bitwidth - 1) - 1
    act_in_mins, act_in_maxs = {}, {}
    act_out_mins, act_out_maxs = {}, {}

    def register_hook(name):
        def forward_hook(module, input_val, output_val):
            if isinstance(input_val, tuple): input_val = input_val[0]
            act_in_mins.setdefault(name, []).append(input_val.detach().min().item())
            act_in_maxs.setdefault(name, []).append(input_val.detach().max().item())
            act_out_mins.setdefault(name, []).append(output_val.detach().min().item())
            act_out_maxs.setdefault(name, []).append(output_val.detach().max().item())
        return forward_hook

    hook_handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handle = module.register_forward_hook(register_hook(name))
            hook_handles.append(handle)

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            model(images.to(device))
    
    for handle in hook_handles:
        handle.remove()
        
    act_quant_params = {}
    for name in act_in_mins:
        rmin_in = np.mean(act_in_mins[name])
        rmax_in = np.mean(act_in_maxs[name])
        rmin_out = np.mean(act_out_mins[name])
        rmax_out = np.mean(act_out_maxs[name])

        scale_in, zp_in = scale_zeropoint(
            torch.tensor(rmin_in, dtype=torch.float32), torch.tensor(rmax_in, dtype=torch.float32), qmin, qmax
        )
        scale_out, zp_out = scale_zeropoint(
            torch.tensor(rmin_out, dtype=torch.float32), torch.tensor(rmax_out, dtype=torch.float32), qmin, qmax
        )
        
        act_quant_params[name] = {
            "scale_act": scale_in.to(device),
            "zp_act": zp_in.to(device),
            "scale_out": scale_out.to(device),
            "zp_out": zp_out.to(device)
        }
        
    return act_quant_params

# ==================== Post-Training Quantization (PTQ) ====================

def quantize_weight(weight, bitwidth=8, per_channel=True, asymmetric=True):
    qmin = -2**(bitwidth - 1)
    qmax = 2**(bitwidth - 1) - 1
    
    if per_channel:
        scales, zeropoints = scale_zero_channel(weight, qmin, qmax, asymmetric=asymmetric)
        scales = scales.view(-1, 1, 1, 1) if weight.dim() == 4 else scales.view(-1, 1)
        zeropoints = zeropoints.view(-1, 1, 1, 1) if weight.dim() == 4 else zeropoints.view(-1, 1)
    else:
        rmin = weight.min()
        rmax = weight.max()
        scales, zeropoints = scale_zeropoint(rmin, rmax, qmin, qmax, asymmetric=asymmetric)
    
    weight_q = torch.clamp(torch.round(weight / scales + zeropoints), qmin, qmax)
    
    return weight_q.to(torch.int8), scales.squeeze(), zeropoints.squeeze()

def apply_ptq(model, act_quant_params, bitwidth=8, weight_q_config=None):
    if weight_q_config is None:
        weight_q_config = {'per_channel': True, 'asymmetric': True}
    for name, module in model.named_modules():
        if name in act_quant_params:
            weight_q, scale_w, zp_w = quantize_weight(
                module.weight.data, 
                bitwidth, 
                **weight_q_config
            )
            
            del module._parameters['weight']
            # module.register_parameter("weight", nn.Parameter(weight_q, requires_grad=False))
            module.register_buffer("weight_q", weight_q)
            module.register_buffer("scale_w", scale_w)
            module.register_buffer("zp_w", zp_w)

            if module.bias is not None:
                module.register_buffer("bias_fp32", module.bias.data.clone())
                del module._parameters['bias']
            
            # 5. Attach the pre-calibrated activation parameters
            params = act_quant_params[name]
            module.register_buffer("scale_act", params["scale_act"])
            module.register_buffer("zp_act", params["zp_act"])
            module.register_buffer("scale_out", params["scale_out"])
            module.register_buffer("zp_out", params["zp_out"])
    
    return model


# ==================== Quantized Layers ====================
class QIntLayer(nn.Module):
    """Statically quantized integer layer."""
    def __init__(self, layer, bitwidth=8):
        super().__init__()
        self.is_conv = isinstance(layer, nn.Conv2d)
        self.bitwidth = bitwidth
        
        self.qmin = -2**(self.bitwidth - 1)
        self.qmax = 2**(self.bitwidth - 1) - 1

        self.register_buffer("weight_q", layer.get_buffer("weight_q"))
        self.register_buffer("scale_w", layer.get_buffer("scale_w"))
        self.register_buffer("zp_w", layer.get_buffer("zp_w"))
        self.register_buffer("scale_act", layer.get_buffer("scale_act"))
        self.register_buffer("zp_act", layer.get_buffer("zp_act"))
        self.register_buffer("scale_out", layer.get_buffer("scale_out"))
        self.register_buffer("zp_out", layer.get_buffer("zp_out"))

        # self.scale_w = layer.get_buffer("scale_w")
        # self.zp_w = layer.get_buffer("zp_w")
        # self.scale_act = layer.get_buffer("scale_act")
        # self.zp_act = layer.get_buffer("zp_act")
        # self.scale_out = layer.get_buffer("scale_out")
        # self.zp_out = layer.get_buffer("zp_out")
        # self.weight_q = layer.get_buffer("weight_q")
        # self.register_parameter("weight_q", layer.weight)
        
        if "bias_fp32" in dict(layer.named_buffers()):
            bias_fp32 = layer.get_buffer("bias_fp32")
            scale_bias = self.scale_act * self.scale_w.view(1, -1)
            bias_q = torch.round(bias_fp32.view(1, -1) / (scale_bias + 1e-8)).to(torch.int32)
            self.register_buffer("bias_q", bias_q.squeeze())
        else:
            self.bias_q = None
            
        if self.is_conv:
            self.stride = layer.stride
            self.padding = layer.padding
            self.dilation = layer.dilation
            self.groups = layer.groups
        M_unshaped = (self.scale_act * self.scale_w) / (self.scale_out + 1e-8)
        if self.is_conv:
            M = M_unshaped.view(1, -1, 1, 1)
        else: 
            M = M_unshaped.view(1, -1)

        self.register_buffer("M", M)

    def forward(self, x_fp: torch.Tensor):
        x_q = torch.clamp(torch.round(x_fp / self.scale_act + self.zp_act), 
                         self.qmin, self.qmax).to(torch.int8)
        
        x_q_shifted = x_q.to(torch.int32) - self.zp_act.to(torch.int32)
        w_q_shifted = self.weight_q.to(torch.int32)
        zp_w_shifted = self.zp_w.to(torch.int32)

        if self.is_conv:
            w_q_shifted = w_q_shifted - zp_w_shifted.view(-1, 1, 1, 1)
        else: # Linear layer
            w_q_shifted = w_q_shifted - zp_w_shifted.view(-1, 1)
        if self.is_conv:
            acc = F.conv2d(x_q_shifted.float(), w_q_shifted.float(), bias=None, 
                          stride=self.stride, padding=self.padding, 
                          dilation=self.dilation, groups=self.groups)
        else:
            acc = F.linear(x_q_shifted.float(), w_q_shifted.float(), bias=None)
        
        if self.bias_q is not None:
            if self.is_conv:
                acc += self.bias_q.view(1, -1, 1, 1)
            else:
                acc += self.bias_q
                
        y_q = torch.round(acc * self.M + self.zp_out)
        y_q = torch.clamp(y_q, self.qmin, self.qmax)
        y_fp = (y_q - self.zp_out) * self.scale_out
        return y_fp


class QIntDynamicLayer(nn.Module):
    """Dynamically quantized integer layer (quantizes activations on-the-fly)."""
    def __init__(self, layer, bitwidth=8):
        super().__init__()
        self.is_conv = isinstance(layer, nn.Conv2d)
        self.bitwidth = bitwidth
        
        self.qmin = -2**(self.bitwidth - 1)
        self.qmax = 2**(self.bitwidth - 1) - 1

        self.weight_q = layer.get_buffer("weight_q")
        self.scale_w = layer.get_buffer("scale_w")
        self.zp_w = layer.get_buffer("zp_w")

        if "bias_fp32" in dict(layer.named_buffers()):
            self.bias_fp32 = layer.get_buffer("bias_fp32")
        else:
            self.bias_fp32 = None
            
        if self.is_conv:
            self.stride = layer.stride
            self.padding = layer.padding
            self.dilation = layer.dilation
            self.groups = layer.groups

    def forward(self, x_fp: torch.Tensor):
        # Dynamic quantization of activations
        rmin = x_fp.min()
        rmax = x_fp.max()
        
        scale_act = (rmax - rmin) / (self.qmax - self.qmin)
        scale_act = torch.where(scale_act == 0, torch.ones_like(scale_act), scale_act)
        zp_act = torch.round(self.qmin - rmin / scale_act)
        zp_act = torch.clamp(zp_act, self.qmin, self.qmax)

        x_q = torch.clamp(torch.round(x_fp / scale_act + zp_act), 
                         self.qmin, self.qmax).to(torch.int8)
        
        x_q_shifted = x_q.to(torch.int32) - zp_act.to(torch.int32)
        w_q_acc = self.weight_q.to(torch.int32)
        
        if self.is_conv:
            acc = F.conv2d(x_q_shifted.float(), w_q_acc.float(), bias=None,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
            scale_w_reshaped = self.scale_w.view(1, -1, 1, 1)
        else:
            acc = F.linear(x_q_shifted.float(), w_q_acc.float(), bias=None)
            scale_w_reshaped = self.scale_w.view(1, -1)
        
        y_fp = acc * scale_act * scale_w_reshaped
        
        if self.bias_fp32 is not None:
            if self.is_conv:
                y_fp += self.bias_fp32.view(1, -1, 1, 1)
            else:
                y_fp += self.bias_fp32
        
        return y_fp


# ==================== Fake Quantization for QAT ====================

class FakeQuantize(nn.Module):
    def __init__(self, bitwidth=8, per_channel=False, asymmetric=True, channel_dim=0, ema_momentum=0.1):
        super().__init__()
        self.bitwidth = bitwidth
        self.qmin = -2**(bitwidth - 1)
        self.qmax = 2**(bitwidth - 1) - 1
        self.per_channel = per_channel
        self.asymmetric = asymmetric
        self.channel_dim = channel_dim
        self.ema_momentum = ema_momentum
        
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))

    def forward(self, x):
        if self.training:
            if self.per_channel and x.dim() > 1:
                dims = list(range(x.dim()))
                dims.pop(self.channel_dim)
                batch_min = x.amin(dim=dims, keepdim=True).detach()
                batch_max = x.amax(dim=dims, keepdim=True).detach()
            else:
                batch_min = x.min().detach()
                batch_max = x.max().detach()

            if self.min_val.isinf():
                self.min_val.copy_(batch_min)
                self.max_val.copy_(batch_max)
            else:
                self.min_val.mul_(1.0 - self.ema_momentum).add_(batch_min * self.ema_momentum)
                self.max_val.mul_(1.0 - self.ema_momentum).add_(batch_max * self.ema_momentum)
            
            rmin, rmax = self.min_val, self.max_val
            current_scale, current_zero_point = scale_zeropoint(rmin, rmax, self.qmin, self.qmax, self.asymmetric)
        else:
            current_scale, current_zero_point = self.scale, self.zero_point

        x_q = torch.clamp(torch.round(x / current_scale + current_zero_point), self.qmin, self.qmax)
        x_dq = (x_q - current_zero_point) * current_scale
        return x + (x_dq - x).detach()

    def freeze_quantization_params(self):
        final_scale, final_zero_point = scale_zeropoint(self.min_val, self.max_val, self.qmin, self.qmax, self.asymmetric)
        self.scale.copy_(final_scale)
        self.zero_point.copy_(final_zero_point)

class QATConv2d(nn.Module):
    def __init__(self, conv_layer, bitwidth=8, weight_q_config=None):
        super().__init__()
        self.conv = conv_layer
        self.bitwidth = bitwidth
        
        if weight_q_config is None:
            weight_q_config = {'per_channel': True, 'asymmetric': True}
        self.weight_fake_quant = FakeQuantize(bitwidth, **weight_q_config, channel_dim=0)
        self.act_fake_quant = FakeQuantize(bitwidth, per_channel=False, asymmetric=True)
        
    def forward(self, x):
        x = self.act_fake_quant(x)
        weight_q = self.weight_fake_quant(self.conv.weight)
        output = F.conv2d(x, weight_q, self.conv.bias, 
                         self.conv.stride, self.conv.padding,
                         self.conv.dilation, self.conv.groups)
        return output

class QATLinear(nn.Module):
    def __init__(self, linear_layer, bitwidth=8, weight_q_config=None):
        super().__init__()
        self.linear = linear_layer
        self.bitwidth = bitwidth
        
        if weight_q_config is None:
            weight_q_config = {'per_channel': True, 'asymmetric': True}
            
        self.weight_fake_quant = FakeQuantize(bitwidth, **weight_q_config, channel_dim=0)
        self.act_fake_quant = FakeQuantize(bitwidth, per_channel=False, asymmetric=True)
        
    def forward(self, x):
        x = self.act_fake_quant(x)
        weight_q = self.weight_fake_quant(self.linear.weight)
        output = F.linear(x, weight_q, self.linear.bias)
        return output


def prepare_qat_model(model, bitwidth=8, weight_q_config=None):
    def _prepare(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(module, name, QATConv2d(child, bitwidth, weight_q_config))
            elif isinstance(child, nn.Linear):
                setattr(module, name, QATLinear(child, bitwidth, weight_q_config))
            else:
                _prepare(child)
        return module
    
    return _prepare(model)


def convert_qat_to_int_model(qat_model, bitwidth=8, sym=True):
    qat_model.eval()
    for module in qat_model.modules():
        if isinstance(module, FakeQuantize):
            module.freeze_quantization_params()

    model_to_convert = deepcopy(qat_model)
    
    if sym:
        weight_q_config = {'per_channel': False, 'asymmetric': False}
    else:
        weight_q_config = {'per_channel': True, 'asymmetric': True}

    def _convert(module):
        for name, child in list(module.named_children()):
            if isinstance(child, (QATConv2d, QATLinear)):
                original_layer = child.conv if isinstance(child, QATConv2d) else child.linear
                
                weight_q, scale_w, zp_w = quantize_weight(
                    original_layer.weight.data, bitwidth, **weight_q_config
                )
                
                del original_layer._parameters['weight']
                original_layer.register_buffer("weight_q", weight_q)
                original_layer.register_buffer("scale_w", scale_w)
                original_layer.register_buffer("zp_w", zp_w)

                if original_layer.bias is not None:
                    original_layer.register_buffer("bias_fp32", original_layer.bias.data.clone())
                    del original_layer._parameters['bias']
                
                act_quantizer = child.act_fake_quant
                output_quantizer = child.output_fake_quant
                original_layer.register_buffer("scale_act", act_quantizer.scale)
                original_layer.register_buffer("zp_act", act_quantizer.zero_point)
                original_layer.register_buffer("scale_out", output_quantizer.scale)
                original_layer.register_buffer("zp_out", output_quantizer.zero_point)
                
                quantized_int_layer = QIntLayer(original_layer, bitwidth)
                setattr(module, name, quantized_int_layer)
            else:
                _convert(child)
        return module

    return _convert(model_to_convert)

def unwrap_qat_model(model):
    unwrapped_model = deepcopy(model)
    def _unwrap(module):
        for name, child in list(module.named_children()):
            if isinstance(child, (QATConv2d, QATLinear)):
                original_layer = child.conv if isinstance(child, QATConv2d) else child.linear
                setattr(module, name, original_layer)
            else:
                _unwrap(child)
    
    _unwrap(unwrapped_model)
    return unwrapped_model


# ==================== Model Conversion ====================

def convert_to_qint_model(model, bitwidth=8):
    """Convert to static quantized integer model."""
    def _convert(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            return QIntLayer(module, bitwidth)
        for name, child in list(module.named_children()):
            setattr(module, name, _convert(child))
        return module
    
    model = _convert(model)
    return model


def convert_to_qint_dynamic_model(model, bitwidth=8):
    """Convert to dynamic quantized integer model."""
    def _convert(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            return QIntDynamicLayer(module, bitwidth)
        for name, child in list(module.named_children()):
            setattr(module, name, _convert(child))
        return module
    
    model = _convert(model)
    return model


# ==================== Helper Functions ====================

def get_model_size(model, bitwidth=None):
    size_in_bytes = 0
    state_dict = model.state_dict()
    
    for name, tensor in state_dict.items():
        is_quantized_weight = name.endswith('.weight_q')

        if is_quantized_weight and bitwidth is not None:
            if bitwidth == 4:
                packed_tensor = pack_int4(tensor.cpu())
                size_in_bytes += packed_tensor.nelement() * packed_tensor.element_size()
            else:
                size_in_bytes += tensor.nelement() * (bitwidth / 8)
        else:
            size_in_bytes += tensor.nelement() * tensor.element_size()
            
    size_mb = size_in_bytes / (1024 * 1024)
    return size_mb
# ==================== Main PTQ Pipeline ====================

def ptq_pipeline(model, calibration_loader, device, bitwidth=8, static=True, sym=True):
    # print(f"Starting PTQ with bitwidth={bitwidth}, static={static}")
    
    model = bnfold_vgg(model, device)
    
    print("Calibrating model...")
    act_quant_params = calibrate_model(model, calibration_loader, device, bitwidth=bitwidth)
    if sym:
        # print("Using Symmetric, Per-Tensor weights.")
        weight_q_config = {'per_channel': False, 'asymmetric': False}
    else:
        # print("Using Asymmetric, Per-Channel weights.")
        weight_q_config = {'per_channel': True, 'asymmetric': True}

    print("Applying quantization...")
    model = apply_ptq(model, act_quant_params, bitwidth, weight_q_config=weight_q_config)
    
    if static:
        model = convert_to_qint_model(model, bitwidth)
    else:
        model = convert_to_qint_dynamic_model(model, bitwidth)
    
    print("PTQ completed!")
    return model

from utils import evaluate
def qat_pipeline(model, train_loader, val_loader, calibration_loader, device, bitwidth=8, 
                 epochs=5, lr=1e-5, sym=True): 
    print(f"Starting Integrated QAT with bitwidth={bitwidth}, epochs={epochs}, lr={lr}")
    model_copy = deepcopy(model)
    model_copy = bnfold_vgg(model_copy, device)
    
    if sym:
        print("Configuring QAT for Symmetric, Per-Tensor weights.")
        weight_q_config = {'per_channel': False, 'asymmetric': False}
    else:
        print("Configuring QAT for Asymmetric, Per-Channel weights.")
        weight_q_config = {'per_channel': True, 'asymmetric': True}

    print("Preparing model for Integrated QAT...")
    qat_model = prepare_qat_model(model_copy, bitwidth, weight_q_config)
    qat_model = qat_model.to(device)
    
    print("Training QAT model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        qat_model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = qat_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        qat_model.eval()
        for module in qat_model.modules():
            if isinstance(module, FakeQuantize):
                module.freeze_quantization_params()

        val_acc = evaluate(qat_model, val_loader, device)
        avg_loss = running_loss / len(train_loader)
        
        print(f"QAT Epoch {epoch+1}/{epochs} | Training Loss: {avg_loss:.3f} | Validation Accuracy (Q-Sim): {val_acc:.2f}%")

    print("\nConverting trained QAT model to final integer format...")
    final_quantized_model = convert_qat_to_int_model(qat_model, bitwidth, sym=sym)
    
    print("Integrated QAT completed!")
    return final_quantized_model

# ==================== Mixed-Precision Quantization ====================

class MixedPrecisionManager:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.variances = {}

    def _profile_hook(self, name):
        def hook(module, input, output):
            var = torch.var(output.detach()).item()
            if name in self.variances:
                self.variances[name].append(var)
            else:
                self.variances[name] = [var]
        return hook

    def profile_activation_variance(self, num_batches=50, device="cuda"):
        self.model.eval()
        self.model.to(device)
        self.variances = {}
        hooks = []

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(self._profile_hook(name)))
        
        with torch.no_grad():
            for i, (images, _) in enumerate(self.dataloader):
                if i >= num_batches:
                    break
                images = images.to(device)
                self.model(images)
        
        for hook in hooks:
            hook.remove()
        
        avg_variances = {name: np.mean(vals) for name, vals in self.variances.items()}
        return avg_variances

    @staticmethod
    def get_quantization_policy(variances, sensitivity_percentile=0.8, 
                                high_precision_bits=32, low_precision_bits=8):
        policy = {}
        variance_values = list(variances.values())
        if not variance_values:
            return {}
            
        threshold = np.percentile(variance_values, sensitivity_percentile * 100)
        
        for name, var in variances.items():
            if var >= threshold:
                policy[name] = high_precision_bits
                print(f"Layer '{name}' is sensitive (variance {var:.4f} >= threshold {threshold:.4f}). Assigning {high_precision_bits}-bit.")
            else:
                policy[name] = low_precision_bits
        
        return policy

def apply_mixed_precision_ptq(model, act_quant_params, policy, low_bitwidth=8, sym=True):
    if not sym:
        weight_q_config = {'per_channel': True, 'asymmetric': True}
    else:
        weight_q_config = {'per_channel':False, 'asymmetric':False}
        
    for name, module in model.named_modules():
        assigned_bits = policy.get(name, low_bitwidth)

        if isinstance(module, (nn.Conv2d, nn.Linear)) and assigned_bits == low_bitwidth:
            print(f"Quantizing layer '{name}' to {low_bitwidth}-bit.")
            
            # 1. Quantize weights and get parameters
            weight_q, scale_w, zp_w = quantize_weight(
                module.weight.data, 
                low_bitwidth,
                **weight_q_config
            )
            del module._parameters['weight']

            module.register_buffer("weight_q", weight_q)
            module.register_buffer("scale_w", scale_w)
            module.register_buffer("zp_w", zp_w)
            
            if module.bias is not None:
                module.register_buffer("bias_fp32", module.bias.data.clone())
                del module._parameters['bias']

            if name in act_quant_params:
                params = act_quant_params[name]
                module.register_buffer("scale_act", params["scale_act"])
                module.register_buffer("zp_act", params["zp_act"])
                module.register_buffer("scale_out", params["scale_out"])
                module.register_buffer("zp_out", params["zp_out"])
            else:
                print(f"Warning: No activation parameters found for quantizable layer {name}. Static quantization might be incorrect.")

    return model


def convert_to_mixed_precision_model(model, policy, low_bitwidth=8):
    model_copy = deepcopy(model)
    
    def _convert(module, prefix=''):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            assigned_bits = policy.get(full_name, low_bitwidth)
            if isinstance(child, (nn.Conv2d, nn.Linear)) and assigned_bits == low_bitwidth:
                if hasattr(child, 'weight_q'): 
                    setattr(module, name, QIntLayer(child, low_bitwidth))
            else:
                _convert(child, prefix=full_name)
        return module
    return _convert(model_copy)

def prepare_mixed_precision_qat_model(model, policy, low_bitwidth=8, weight_q_config=None):
    """
    Prepares a model for mixed-precision QAT by only wrapping layers specified
    by the policy.
    """
    def _prepare(module, prefix=''):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            assigned_bits = policy.get(full_name, low_bitwidth)

            if isinstance(child, (nn.Conv2d, nn.Linear)) and assigned_bits == low_bitwidth:
                print(f"Preparing layer '{full_name}' for QAT with {low_bitwidth}-bit.")
                if isinstance(child, nn.Conv2d):
                    setattr(module, name, QATConv2d(child, low_bitwidth, weight_q_config))
                else:
                    setattr(module, name, QATLinear(child, low_bitwidth, weight_q_config))
            else:
                # Recurse into submodules
                _prepare(child, full_name)
        return module
    
    return _prepare(deepcopy(model))



# In your quantization.py, find the section for clipped PTQ and replace it entirely with this.

DEVICE = "cuda"  # Make sure this is defined, or pass it as an argument.

# ==================== Clipped/Percentile Calibration ====================

class ClippedActivationObserver:
    """
    Observer that finds activation ranges based on percentiles to ignore outliers.
    This version is memory-efficient by downsampling the collected activations.
    """
    def __init__(self, qmin, qmax, percentile=99.9, max_samples_per_batch=8192):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.percentile = percentile
        self.collected_inputs = []
        self.collected_outputs = []
        self.max_samples_per_batch = max_samples_per_batch

    def _update_with_sampling(self, collection, tensor):
        """Helper to sample a tensor before adding it to the collection list."""
        flat_tensor = tensor.detach().flatten()
        if flat_tensor.numel() > self.max_samples_per_batch:
            indices = torch.randperm(flat_tensor.numel(), device=flat_tensor.device)[:self.max_samples_per_batch]
            collection.append(flat_tensor[indices].cpu())
        else:
            collection.append(flat_tensor.cpu())

    def update_input(self, tensor):
        self._update_with_sampling(self.collected_inputs, tensor)

    def update_output(self, tensor):
        self._update_with_sampling(self.collected_outputs, tensor)

    def _get_clipped_scale_zeropoint(self, collected_tensors):
        if not collected_tensors:
            # Return valid tensors on the correct device
            return torch.tensor(1.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)

        all_vals = torch.cat(collected_tensors)
        lower_bound = (100 - self.percentile) / 100
        upper_bound = self.percentile / 100
        clipped_min = torch.quantile(all_vals, lower_bound)
        clipped_max = torch.quantile(all_vals, upper_bound)
        print(f"  - (Sampled) Original min/max: {all_vals.min():.3f}/{all_vals.max():.3f}. Clipped min/max: {clipped_min:.3f}/{clipped_max:.3f}")
        return scale_zeropoint(clipped_min.to(DEVICE), clipped_max.to(DEVICE), self.qmin, self.qmax)

    def get_final_params(self):
        """Calculates and returns all final scale/zp values in a dictionary."""
        print("  Input stats:")
        scale_in, zp_in = self._get_clipped_scale_zeropoint(self.collected_inputs)
        print("  Output stats:")
        scale_out, zp_out = self._get_clipped_scale_zeropoint(self.collected_outputs)
        
        return {
            "scale_act": scale_in, "zp_act": zp_in,
            "scale_out": scale_out, "zp_out": zp_out
        }

def calibrate_model_clipped(model, dataloader, device, num_batches=100, bitwidth=8, percentile=99.9):
    """
    Calibrates using the memory-efficient ClippedActivationObserver and returns a dictionary of parameters
    in the same format as calibrate_model.
    """
    model.eval()
    qmin = -2**(bitwidth - 1)
    qmax = 2**(bitwidth - 1) - 1
    
    observers = {
        name: ClippedActivationObserver(qmin, qmax, percentile=percentile)
        for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))
    }

    def register_hook(name):
        def forward_hook(module, input_val, output_val):
            if isinstance(input_val, tuple): input_val = input_val[0]
            observers[name].update_input(input_val)
            observers[name].update_output(output_val)
        return forward_hook

    hook_handles = []
    for name, module in model.named_modules():
        if name in observers:
            hook_handles.append(module.register_forward_hook(register_hook(name)))

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches: break
            model(images.to(device))
    
    for handle in hook_handles: handle.remove()
        
    act_quant_params = {}
    for name, observer in observers.items():
        print(f"Calculating clipped params for layer '{name}':")
        act_quant_params[name] = observer.get_final_params()
        
    return act_quant_params

def ptq_pipeline_clipped(model, calibration_loader, device, bitwidth=8, percentile=99.9, sym=True):
    """
    A corrected, robust PTQ pipeline that uses percentile clipping for activation ranges.
    """
    print(f"Starting CLIPPED PTQ with bitwidth={bitwidth}, percentile={percentile}")
    
    model = bnfold_vgg(model, device)
    
    # Step 1: Calibrate using the dedicated clipped calibration function.
    print("Calibrating model with percentile clipping...")
    act_quant_params = calibrate_model_clipped(
        model, calibration_loader, device, bitwidth=bitwidth, percentile=percentile
    )
    
    # Step 2: Configure weight quantization scheme.
    if sym:
        weight_q_config = {'per_channel': False, 'asymmetric': False}
    else:
        weight_q_config = {'per_channel': True, 'asymmetric': True}
    
    # Step 3: Apply quantization using the existing, proven 'apply_ptq' function.
    # It now receives the correctly formatted `act_quant_params` dictionary.
    print("Applying quantization...")
    model_prepared = deepcopy(model)
    model_prepared = apply_ptq(model_prepared, act_quant_params, bitwidth, weight_q_config=weight_q_config)
    
    # Step 4: Convert to the final integer model.
    final_model = convert_to_qint_model(model_prepared, bitwidth)
    
    print("Clipped PTQ completed!")
    return final_model

import matplotlib.pyplot as plt

class ClippedActivationVisualizer:
    """A dedicated visualizer to show the effect of percentile clipping on activation ranges."""
    def __init__(self, model, layers_to_watch):
        self.model = model
        self.layers_to_watch = layers_to_watch
        self.activations = {name: [] for name in layers_to_watch}
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, name):
        def hook(module, input, output):
            # We only need the output activations for this visualization
            self.activations[name].append(output.detach().cpu())
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layers_to_watch:
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def capture_activations(self, data_loader, device, num_batches=5):
        self.model.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader):
                if i >= num_batches:
                    break
                self.model(images.to(device))

    def plot_clipped_distributions(self, percentile=99.9):
            num_layers = len(self.layers_to_watch)

            nrows = 2
            ncols = (num_layers + nrows - 1) // nrows  
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4)) 
            axes = axes.flatten() 
            
            fig.suptitle(f'Effect of {percentile}% Percentile Clipping on Activation Ranges', fontsize=16, fontweight='bold')

            for ax, name in zip(axes, self.layers_to_watch):
                acts_tensor = torch.cat(self.activations[name]).flatten()
                acts_numpy = acts_tensor.numpy()
                orig_min, orig_max = acts_numpy.min(), acts_numpy.max()
                lower_percentile = 100.0 - percentile
                clipped_min, clipped_max = np.percentile(acts_numpy, [lower_percentile, percentile])
                ax.hist(acts_numpy, bins=100, color='royalblue', alpha=0.6, density=True, label='Original Distribution')
                ax.axvspan(clipped_min, clipped_max, color='limegreen', alpha=0.3, label=f'Clipped Range ({percentile}%)')
                ax.axvline(orig_min, color='red', linestyle='--', linewidth=1.5, label='Original Min/Max')
                ax.axvline(orig_max, color='red', linestyle='--', linewidth=1.5)
                ax.set_title(f"Layer: {name}", fontsize=11)
                ax.set_xlabel('Activation Value', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.4)
            for i in range(num_layers, len(axes)):
                axes[i].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            plt.savefig('clipped_distributions_visualization_grid.png', dpi=300)
            plt.show()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class ActivationVisualizer:
    def __init__(self, model, layers_to_watch):
        self.model = model
        self.layers_to_watch = layers_to_watch
        self.activations = {name: [] for name in layers_to_watch}
        self.hooks = []
        self._register_hooks()

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name].append(output.detach().cpu())
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layers_to_watch:
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def capture_activations(self, data_loader, device, num_batches=1):
        self.model.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader):
                if i >= num_batches:
                    break
                self.model(images.to(device))

    def plot_histograms(self):
        num_layers = len(self.layers_to_watch)
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(num_layers * 2.5, 8))
        axes = axes.flatten()
        fig.suptitle('Activation Distributions Across VGG11 Layers', fontsize=18, fontweight='bold')

        for ax, name in zip(axes, self.layers_to_watch):
            acts = torch.cat(self.activations[name]).flatten().numpy()
            ax.hist(acts, bins=100, color='royalblue', alpha=0.8, density=True)
            ax.set_title(f"{name}", fontsize=10)
            ax.set_xlabel('Activation', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.grid(alpha=0.3)

        for ax in axes[len(self.layers_to_watch):]:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()