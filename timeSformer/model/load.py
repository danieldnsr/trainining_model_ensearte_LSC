from transformers import TimesformerForVideoClassification, AutoImageProcessor


def load_model(
    model_ckpt: str,
    label2id: dict,
    id2label: dict,
    device: str,
    freeze_base: bool = True,
    unfreeze_last_n: int = 0
):
    """
    Loads the VideoMAE model and image processor.

    Args:
        model_ckpt (str): Path or identifier of the pretrained checkpoint.
        label2id (dict): Mapping from labels to IDs.
        id2label (dict): Mapping from IDs to labels.
        device (str): Device to load the model on (e.g., 'cuda' or 'cpu').
        freeze_base (bool): If True, freeze the base 'videomae' layers except last `unfreeze_last_n`.
        unfreeze_last_n (int): Number of last 'videomae' encoder layers to unfreeze (only relevant if freeze_base=True).

    Returns:
        model (TimesformerForVideoClassification): The loaded model ready for training or inference.
    """
    print(f"Unique classes: {list(label2id.keys())}.")
    model = TimesformerForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    if not freeze_base:
        print("Unfreezing all layers of the model...")
        for param in model.parameters():
            param.requires_grad = True
    else:
        print(f"Freezing base layers except the last {unfreeze_last_n} encoder layers...")
        # Primero congelamos todo el backbone videomae
        for name, param in model.named_parameters():
            if name.startswith("videomae."):
                param.requires_grad = False
            else:
                # Descongelamos cabeza (fc_norm, classifier, etc)
                param.requires_grad = True

        # Descongelar las últimas unfreeze_last_n capas del encoder
        if unfreeze_last_n > 0:
            encoder_layers = model.timesformer.encoder.layer
            total_layers = len(encoder_layers)
            layers_to_unfreeze = encoder_layers[total_layers - unfreeze_last_n :]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

    print(f"Model moved to device: {device}")
    return model


def load_image_processor(model_ckpt: str,):
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
    return image_processor  # Placeholder, as we are not using an image processor in this context