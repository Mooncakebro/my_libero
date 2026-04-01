# qwen3_vl_embedder.py
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Union, Optional, Tuple

class Qwen3VLEmbedder(nn.Module):
    """
    A PyTorch module that extracts embeddings from a Qwen3-VL model
    given an image and a text instruction.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[Union[str, torch.device]] = None,
        quantize_4bit: bool = False,
        pooling: str = "none",
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model identifier (e.g., Qwen/Qwen3-VL-2B-Instruct)
            device: Device to place the model on. If None, uses cuda if available else cpu.
            quantize_4bit: Whether to load model in 4-bit quantization (requires bitsandbytes).
            pooling: Pooling strategy over sequence length:
                - "none": return full sequence (batch, seq_len, hidden_dim)
                - "mean": mean pooling -> (batch, hidden_dim)
                - "max": max pooling -> (batch, hidden_dim)
                - "first": first token (e.g., <|im_start|>) -> (batch, hidden_dim)
            dtype: Data type for model weights when not quantized.
            trust_remote_code: Required for Qwen models.
        """
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.model_name = model_name

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        # Load model with optional quantization
        if quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                )
                print(f"Loaded {model_name} in 4-bit quantization.")
            except ImportError:
                raise ImportError("bitsandbytes is required for 4-bit quantization. Install with: pip install bitsandbytes")
        else:
            torch_dtype = dtype if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=trust_remote_code,
            )
            print(f"Loaded {model_name} with dtype {torch_dtype} on {self.device}.")

        # Freeze model parameters by default (set to eval)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Move processor to device (though processor is not a tensor, it's just for tokenization)
        # Ensure model is on correct device if not using device_map="auto"
        if not quantize_4bit:
            self.model.to(self.device)

    def forward(
        self,
        image: Union[str, Image.Image],
        instruction: str,
        return_full_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Extract embeddings from the VLM.

        Args:
            image: Path to an image file or a PIL Image object.
            instruction: Text instruction (e.g., "Describe this image.").
            return_full_sequence: If True, ignores the pooling strategy and returns the full hidden state sequence.
                                   Useful for downstream networks that want per-token embeddings.

        Returns:
            Embeddings as a torch.Tensor on the same device as the model.
            Shape depends on pooling:
                - "none" or return_full_sequence=True: (1, seq_len, hidden_dim)
                - other pooling: (1, hidden_dim)
        """
        # Convert image path to PIL if needed
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be a file path string or a PIL Image.")

        # Build chat message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        # Apply chat template and process inputs
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():  # No gradients needed for frozen embeddings
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden_state = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)

        if return_full_sequence:
            return last_hidden_state

        # Apply pooling
        if self.pooling == "mean":
            return last_hidden_state.mean(dim=1)  # (1, hidden_dim)
        elif self.pooling == "max":
            return last_hidden_state.max(dim=1)[0]  # (1, hidden_dim)
        elif self.pooling == "first":
            return last_hidden_state[:, 0, :]  # (1, hidden_dim)
        else:  # "none"
            return last_hidden_state

    def get_hidden_dim(self) -> int:
        """Return the hidden dimension size of the model."""
        # For Qwen3-VL, the hidden size is the last dimension of the model's config
        return self.model.config.hidden_size


def test_embedder():
    """Simple test function to demonstrate usage and inspect embeddings."""
    import argparse
    parser = argparse.ArgumentParser(description="Test Qwen3VL Embedder")
    parser.add_argument("--image", type=str, default="./zidane.jpg", help="Path to test image")
    parser.add_argument("--instruction", type=str, default="Locate the left person", help="Instruction")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Model name")
    parser.add_argument("--quantize_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--pooling", type=str, default="none", choices=["none","mean","max","first"], help="Pooling strategy")
    args = parser.parse_args()

    # Create embedder
    embedder = Qwen3VLEmbedder(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        quantize_4bit=True,
        pooling="none",
    )
    print(f"Embedder created. Hidden dimension: {embedder.get_hidden_dim()}")

    # Forward pass
    embeddings = embedder(args.image, args.instruction)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings device: {embeddings.device}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    if embeddings.numel() > 0:
        print(f"First 10 values: {embeddings.flatten()[:10]}")


if __name__ == "__main__":
    test_embedder()