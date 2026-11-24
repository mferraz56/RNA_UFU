import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import CLASSES, SolarDataModule
from model import SimpleSolarCNN


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = (
    REPOSITORY_ROOT
    / "Dataset"
    / "2020-02-14_InfraredSolarModules"
    / "InfraredSolarModules"
)


def grad_cam(model: SimpleSolarCNN, image: torch.Tensor, target_index: int = None):
    activations = []
    gradients = []

    def save_activation(module, inputs, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.features[-4]
    handle_forward = target_layer.register_forward_hook(save_activation)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)
    model.zero_grad(set_to_none=True)
    logits = model(image)
    if target_index is None:
        target_index = int(logits.argmax(dim=1).item())
    logits[0, target_index].backward()
    handle_forward.remove()
    handle_backward.remove()

    activation = activations[0].detach()[0]
    gradient = gradients[0].detach()[0]
    weights = gradient.mean(dim=(1, 2))
    heatmap = torch.relu((weights[:, None, None] * activation).sum(dim=0))
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.cpu().numpy(), target_index, torch.softmax(logits, dim=1)[0].detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera um mapa Grad-CAM para uma imagem")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "resultados" / "gradcam.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SimpleSolarCNN(len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    module = SolarDataModule(args.dataset, seed=checkpoint["seed"])
    image = torch.from_numpy(module.load_image(args.index)).unsqueeze(0).unsqueeze(0).float().to(device)
    heatmap, predicted_index, probabilities = grad_cam(model, image)

    from matplotlib import pyplot as plt
    original = module.load_image(args.index)
    upscaled = np.asarray(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((40, 24)))
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="inferno")
    plt.title("Imagem termica")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(original, cmap="gray")
    plt.imshow(upscaled, cmap="jet", alpha=0.45)
    plt.title(CLASSES[predicted_index])
    plt.axis("off")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Classe prevista: {CLASSES[predicted_index]} ({probabilities[predicted_index]:.3f})")
    print(f"Saida: {args.output}")


if __name__ == "__main__":
    main()
