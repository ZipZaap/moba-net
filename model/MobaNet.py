import torch
import torch.nn as nn
from utils.util import logits_to_lbl, logits_to_msk

# --------------------------------------------------------------------------------------
# Basic Convolutional Block
# --------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
        )
        
    def forward(self, x):
        return self.convblock(x)
    
# --------------------------------------------------------------------------------------
# Encoder Block
# --------------------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        skip = self.conv(x)
        x = self.drop(skip)
        x = self.pool(x)
        return x, skip
    
# --------------------------------------------------------------------------------------
# Decoder Block
# --------------------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c + out_c, out_c)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.drop(x)
        return x
    
# --------------------------------------------------------------------------------------
# Encoder Body (Downsampling path)
# --------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.encoders = nn.ModuleList()

        for i in range(len(channels) - 2):
            self.encoders.append(EncoderBlock(channels[i], channels[i+1], dropout))
        
        self.bottleneck = ConvBlock(channels[-2], channels[-1])
        
    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        x = self.bottleneck(x)
        return x, skips
    
# --------------------------------------------------------------------------------------
# Segmentation Head (Decoder path)
# --------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, channels, dropout):
        super().__init__()
        self.decoders = nn.ModuleList()

        for i in range(len(channels) - 2):
            self.decoders.append(DecoderBlock(channels[i], channels[i+1], dropout))

        self.seg_out = nn.Conv2d(channels[-2], channels[-1], kernel_size=1)
        self.sdm_out = nn.Conv2d(channels[-2], 1, kernel_size=1)
        
    def forward(self, x, skips):
        for decoder, skip in zip(self.decoders, skips[::-1]):
            x = decoder(x, skip)
        x1 = self.seg_out(x)
        x2 = self.sdm_out(x)
        return x1, x2

# --------------------------------------------------------------------------------------
# Classification Head (Optional)
# --------------------------------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self, 
                 bottleneck_channels, 
                 cls_classes, 
                 cls_dropout,
                 pool_size = 2
                 ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        
        self.dense = nn.Sequential(
            nn.Linear(bottleneck_channels, bottleneck_channels // 2),
            nn.BatchNorm1d(bottleneck_channels // 2),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(bottleneck_channels // 2, cls_classes)
        )
        
        # # Preserve regional spatial information by pooling to a small grid (e.g., 2x2)
        # self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        # self.flat = nn.Flatten()
        
        # flattened_features = bottleneck_channels * (pool_size ** 2)
        # self.dense = nn.Sequential(
        #     nn.Linear(flattened_features, bottleneck_channels // 2),
        #     nn.BatchNorm1d(bottleneck_channels // 2),
        #     nn.ReLU(),
        #     nn.Dropout(cls_dropout),
        #     nn.Linear(bottleneck_channels // 2, cls_classes)
        # )
        
    def forward(self, x):
        x = self.pool(x)
        x = self.flat(x)
        x = self.dense(x)
        return x
    
# --------------------------------------------------------------------------------------
# MobaNet (Combined Model)
# --------------------------------------------------------------------------------------
class MobaNet(nn.Module):
    def __init__(self, 
                 *,
                 model: str,
                 unet_depth: int,
                 conv_depth: int,
                 in_channels: int,
                 seg_classes: int,
                 cls_classes: int,
                 seg_dropout: float = 0.0,
                 cls_dropout: float = 0.0,
                 inference: bool = False
                 ):
        """
        Initializes the MobaNet model.

        Parameters
        ----------
        model : str
            Model type

        unet_depth : int
            Number of layers in the encoder/decoder

        conv_depth : int
            Depth Conv2d block in 1st layer (doubles with each layer).

        in_channels : int
            Number of input channels.

        seg_classes : int
            Number of segmentation classes.

        cls_classes : int
            Number of classification classes.

        seg_dropout : float
            Dropout rate for segmentation layers.

        cls_dropout : float
            Dropout rate for classification layers.

        inference : bool
            Whether the model is in inference mode.
        """

        super().__init__()

        self.model = model
        self.inference = inference
        self.seg_classes = seg_classes
        self.cls_classes = cls_classes

        enc_ch = [conv_depth * (2 ** i) for i in range(unet_depth)]
        dec_ch = enc_ch[::-1]

        enc_ch = [in_channels] + enc_ch
        dec_ch = dec_ch + [seg_classes]

        self.encoder = Encoder(enc_ch, seg_dropout)
        self.decoder = Decoder(dec_ch, seg_dropout)
        self.classifier = Classifier(enc_ch[-1], cls_classes, cls_dropout)

    def forward(self, 
                x: torch.Tensor,
                cls_threshold: float | None = None
                ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the MobaNet model.

        Parameters
        ----------
        x : torch.Tensor (B, C, H, W)
            Input tensor.
            
        cls_threshold : float | None
            Threshold for classifying logits into labels. If the maximum probability 
            is < threshold, the label is set to `seg_classes - 1`, i.e. boundary class. 
            If `None`, the argmax of the logits is used to determine the class labels.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the output tensors.
        """
        if self.inference:

            B, _, H, W = x.shape
            x, skips = self.encoder(x)
            
            if 'UNet' in self.model:
                # feed the input through the decoder
                seg_logits, _ = self.decoder(x, skips)

                # convert logits to segmentation probabilities; (B, C, H, W)
                seg_probs = logits_to_msk(seg_logits, 'softmax')

            else:
                # generate a test cls_logits tensor
                cls_logits = self.classifier(x)
                
                # convert logits to predicted class labels & probabilities; (B, C) → (B,)
                cls_probs, pd_cls = logits_to_lbl(cls_logits, cls_threshold)
                
                # class masks
                boundary = (pd_cls == self.seg_classes)
                uniform  = (pd_cls != self.seg_classes)
                
                # initialize segmentation probabilities; (B, C, H, W)
                seg_probs = torch.zeros((B, self.seg_classes, H, W), 
                                        dtype=torch.float32, 
                                        device=x.device)
                
                # filter indices and probabilities
                batch_idx = torch.arange(B, device=x.device)
                b_idx = batch_idx[uniform]
                c_idx = pd_cls[uniform]
                probs = cls_probs[uniform]
                seg_probs[b_idx, c_idx] = probs[:, None, None]
            
                # run segmentation head only for images belonging to boundary class.
                if boundary.any():
                    x = x[boundary]
                    skips = [s[boundary] for s in skips]
                    seg_logits, _ = self.decoder(x, skips)
                    seg_probs[boundary] = logits_to_msk(seg_logits, "softmax")

            return {'seg': seg_probs}

        else:
            x, skips = self.encoder(x)
            seg_logits, sdm_logits = self.decoder(x, skips)

            if 'UNet' in self.model:
                return {'seg': seg_logits, 'sdm': sdm_logits}
            
            else: # 'MobaNet' in self.model
                cls_logits = self.classifier(x)
                return {'seg': seg_logits, 'sdm': sdm_logits, 'cls': cls_logits}