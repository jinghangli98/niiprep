#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

# Weights shipped with niiprep (standard model; --no-csf still needs FreeSurfer)
BUNDLED_MODEL = os.path.join(os.path.dirname(__file__), 'models', 'synthstrip.1.pt')

# Define model architecture (same as original)
class StripModel(nn.Module):
    def __init__(self,
                 nb_features=16,
                 nb_levels=7,
                 feat_mult=2,
                 max_features=64,
                 nb_conv_per_level=2,
                 max_pool=2,
                 return_mask=False):

        super().__init__()

        # dimensionality
        ndims = 3

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == None:
            self.activation = None
        else:
            raise ValueError(f'Unknown activation: {activation}')

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


def extend_sdt(sdt, border=1, voxsize=None):
    """Extend SynthStrip's narrow-band signed distance transform (SDT).
    
    Parameters
    ----------
    sdt : np.ndarray or torch.Tensor
        Narrow-band signed distance transform estimated by SynthStrip.
    border : float, optional
        Mask border threshold in millimeters.
    voxsize : tuple or list, optional
        Voxel size in millimeters. If None, assumes 1mm isotropic.
    
    Returns
    -------
    sdt : np.ndarray or torch.Tensor
        Extended SDT.
    """
    is_tensor = torch.is_tensor(sdt)
    if is_tensor:
        device = sdt.device
        sdt_np = sdt.cpu().numpy()
    else:
        sdt_np = sdt
    
    if border < int(sdt_np.max()):
        return sdt
    
    # Find bounding box
    mask = sdt_np < 1
    keep = np.nonzero(mask)
    low = np.min(keep, axis=-1)
    upp = np.max(keep, axis=-1)
    
    # Add requested border
    gap = int(border + 0.5)
    low = [max(i - gap, 0) for i in low]
    upp = [min(i + gap, d - 1) for i, d in zip(upp, mask.shape)]
    
    # Compute EDT within bounding box
    ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
    out = np.full_like(sdt_np, fill_value=100)
    
    # Simple Euclidean distance transform for the mask region
    from scipy.ndimage import distance_transform_edt
    out[ind] = distance_transform_edt(~mask[ind])
    out[keep] = sdt_np[keep]
    
    if is_tensor:
        return torch.from_numpy(out).to(device)
    return out


class SynthStripHelper:
    """Helper class for skull stripping with numpy arrays or torch tensors."""
    
    def __init__(self, model_path=None, device='auto', no_csf=False):
        """
        Initialize the SynthStrip helper.
        
        Parameters
        ----------
        model_path : str, optional
            Path to model weights. If None, uses FreeSurfer's default model.
        device : str or torch.device, optional
            Device to run on. 'auto' automatically selects GPU if available.
        no_csf : bool, optional
            If True, uses model that excludes CSF from brain boundary.
        """
        # Configure device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Initialize model
        self.model = StripModel()
        self.model.to(self.device)
        self.model.eval()
        
        # Load weights
        if model_path is not None:
            modelfile = model_path
        elif not no_csf and os.path.isfile(BUNDLED_MODEL):
            modelfile = BUNDLED_MODEL
        else:
            version = '1'
            fshome = os.environ.get('FREESURFER_HOME')
            if fshome is None:
                raise RuntimeError('FREESURFER_HOME env variable must be set or provide model_path!')
            if no_csf:
                modelfile = os.path.join(fshome, 'models', f'synthstrip.nocsf.{version}.pt')
            else:
                modelfile = os.path.join(fshome, 'models', f'synthstrip.{version}.pt')
        
        checkpoint = torch.load(modelfile, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def strip(self, data, border=1, fill_value=0, return_sdt=False):
        """
        Perform skull stripping on input data.
        
        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            3D brain image data. Can be numpy array or torch tensor.
        border : float, optional
            Mask border threshold in millimeters. Default is 1.
        fill_value : float, optional
            Value to use for background. Default is 0.
        return_sdt : bool, optional
            If True, also returns the signed distance transform.
        
        Returns
        -------
        mask : np.ndarray or torch.Tensor
            Binary brain mask (same type as input).
        brain : np.ndarray or torch.Tensor
            Brain-extracted image (same type as input).
        sdt : np.ndarray or torch.Tensor, optional
            Signed distance transform (if return_sdt=True).
        """
        # Check input type
        is_tensor = torch.is_tensor(data)
        if is_tensor:
            input_device = data.device
            input_dtype = data.dtype
            data_np = data.cpu().numpy().astype(np.float32)
        else:
            data_np = data.astype(np.float32)
            
        # Store original shape and data
        original_shape = data_np.shape
        original_data = data_np.copy()
        
        # Conform to 1mm isotropic and pad to multiples of 64
        # For simplicity, we'll work with the data as-is but pad if needed
        target_shape = np.clip(np.ceil(np.array(original_shape) / 64).astype(int) * 64, 192, 320)
        
        # Pad the data to target shape
        pad_widths = []
        for orig, target in zip(original_shape, target_shape):
            total_pad = target - orig
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_widths.append((pad_before, pad_after))
        
        padded_data = np.pad(data_np, pad_widths, mode='constant', constant_values=data_np.min())
        
        # Normalize
        padded_data -= padded_data.min()
        max_val = np.percentile(padded_data[padded_data > 0], 99) if np.any(padded_data > 0) else padded_data.max()
        if max_val > 0:
            padded_data = np.clip(padded_data / max_val, 0, 1)
        
        # Prepare input tensor
        inp = torch.from_numpy(padded_data[np.newaxis, np.newaxis]).float().to(self.device)
        
        # Predict SDT
        with torch.no_grad():
            sdt_padded = self.model(inp).squeeze().cpu().numpy()
        
        # Extend SDT if needed
        sdt_padded = extend_sdt(sdt_padded, border=border)
        
        # Extract mask from SDT
        mask_padded = (sdt_padded < border).astype(np.uint8)
        
        # Find largest connected component
        from scipy.ndimage import label
        labeled_array, num_features = label(mask_padded)
        if num_features > 0:
            # Find largest component
            component_sizes = np.bincount(labeled_array.ravel())
            component_sizes[0] = 0  # Ignore background
            largest_component = component_sizes.argmax()
            mask_padded = (labeled_array == largest_component).astype(np.uint8)
        
        # Unpad to original shape
        unpad_slices = tuple(slice(pad_before, pad_before + orig) 
                            for (pad_before, _), orig in zip(pad_widths, original_shape))
        mask = mask_padded[unpad_slices]
        sdt = sdt_padded[unpad_slices]
        
        # Apply mask to get brain
        brain = original_data.copy()
        brain[mask == 0] = fill_value
        
        # Convert back to original type if needed
        if is_tensor:
            mask = torch.from_numpy(mask).to(input_device).to(torch.uint8)
            brain = torch.from_numpy(brain).to(input_device).to(input_dtype)
            sdt = torch.from_numpy(sdt).to(input_device).to(torch.float32)
        
        if return_sdt:
            return mask, brain, sdt
        else:
            return mask, brain


# Convenience function for direct use
def strip(data, border=1, fill_value=0, model_path=None, device='auto', no_csf=False, return_sdt=False):
    """
    Convenience function for skull stripping.
    
    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        3D brain image data.
    border : float, optional
        Mask border threshold in millimeters. Default is 1.
    fill_value : float, optional
        Value to use for background. Default is 0.
    model_path : str, optional
        Path to model weights. If None, uses FreeSurfer's default model.
    device : str or torch.device, optional
        Device to run on. 'auto' automatically selects GPU if available.
    no_csf : bool, optional
        If True, uses model that excludes CSF from brain boundary.
    return_sdt : bool, optional
        If True, also returns the signed distance transform.
    
    Returns
    -------
    mask : np.ndarray or torch.Tensor
        Binary brain mask.
    brain : np.ndarray or torch.Tensor
        Brain-extracted image.
    sdt : np.ndarray or torch.Tensor, optional
        Signed distance transform (if return_sdt=True).
    
    Examples
    --------
    >>> import numpy as np
    >>> from synthstrip_helper import strip
    >>> 
    >>> # With numpy array
    >>> data = np.random.randn(256, 256, 256)
    >>> mask, brain = strip(data)
    >>> 
    >>> # With torch tensor
    >>> import torch
    >>> data_tensor = torch.randn(256, 256, 256)
    >>> mask, brain = strip(data_tensor)
    >>> 
    >>> # Get SDT as well
    >>> mask, brain, sdt = strip(data, return_sdt=True)
    """
    helper = SynthStripHelper(model_path=model_path, device=device, no_csf=no_csf)
    return helper.strip(data, border=border, fill_value=fill_value, return_sdt=return_sdt)


# Original command-line interface
if __name__ == "__main__":
    import surfa as sf

    ref = '''
    If you use SynthStrip in your analysis, please cite:
    ----------------------------------------------------
    SynthStrip: Skull-Stripping for Any Brain Image
    A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann
    NeuroImage 206 (2022), 119474
    https://doi.org/10.1016/j.neuroimage.2022.119474

    Website: https://synthstrip.io
    '''

    # parse command line
    p = argparse.ArgumentParser(description='Robust, universal skull-stripping for brain images of any type.')
    p.add_argument('-i', '--image', metavar='FILE', required=True, help='input image to skullstrip')
    p.add_argument('-o', '--out', metavar='FILE', help='save stripped image to file')
    p.add_argument('-m', '--mask', metavar='FILE', help='save binary brain mask to file')
    p.add_argument('-d', '--sdt', metavar='FILE', help='save distance transform to file')
    p.add_argument('-g', '--gpu', action='store_true', help='use the GPU')
    p.add_argument('-b', '--border', default=1, type=float, help='mask border threshold in mm, defaults to 1')
    p.add_argument('-t', '--threads', type=int, help='PyTorch CPU threads, PyTorch default if unset')
    p.add_argument('-f', '--fill', type=float, help='BG fill value, defaults to min(image.min, 0)')
    p.add_argument('--no-csf', action='store_true', help='exclude CSF from brain border')
    p.add_argument('--model', metavar='FILE', help='alternative model weights')
    
    if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
        p.print_help()
        print(ref)
        exit(1)
    
    args = p.parse_args()
    
    # sanity check on the inputs
    if not args.out and not args.mask and not args.sdt:
        sf.system.fatal('Must provide at least one -o, -m, or -d output flag.')
    
    # Configure threads
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    
    # Configure device
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Load input volume
    image = sf.load_volume(args.image)
    print(f'Input image read from: {args.image}')
    
    # Initialize helper
    helper = SynthStripHelper(model_path=args.model, device=device, no_csf=args.no_csf)
    
    # Process frames
    print(f'Processing frame (of {image.nframes}):', end=' ', flush=True)
    masks = []
    brains = []
    sdts = []
    
    fill = np.min([image.min(), 0]) if args.fill is None else args.fill
    
    for f in range(image.nframes):
        print(f + 1, end=' ', flush=True)
        frame_data = image.framed_data[..., f]
        
        mask, brain, sdt = helper.strip(frame_data, 
                                       border=args.border, 
                                       fill_value=fill,
                                       return_sdt=True)
        
        masks.append(mask)
        brains.append(brain)
        sdts.append(sdt)
    
    print('done')
    
    # Stack frames if needed
    if image.nframes > 1:
        mask_vol = sf.stack([image.new(m) for m in masks])
        brain_vol = sf.stack([image.new(b) for b in brains])
        sdt_vol = sf.stack([image.new(s) for s in sdts])
    else:
        mask_vol = image.new(masks[0])
        brain_vol = image.new(brains[0])
        sdt_vol = image.new(sdts[0])
    
    # Write outputs
    if args.out:
        brain_vol.save(args.out)
        print(f'Set background to: {fill}')
        print(f'Masked image saved to: {args.out}')
    
    if args.mask:
        mask_vol.save(args.mask)
        print(f'Binary brain mask saved to: {args.mask}')
    
    if args.sdt:
        sdt_vol.save(args.sdt)
        print(f'Distance transform saved to: {args.sdt}')
    
    print(ref)