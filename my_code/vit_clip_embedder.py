import torch
import torch.nn as nn
# from transformers import CLIPTextModel, ViTModel, AutoProcessor, AutoImageProcessor
from transformers import CLIPTextModel, ViTModel, CLIPProcessor, ViTFeatureExtractor
from PIL import Image
import os

class MultimodalEmbedder(nn.Module):
    def __init__(self, text_model_path, image_model_path, common_dim=768):
        super().__init__()
        
        # 1. Load Text Encoder (CLIP)
        # We load the full model but will only use the text_model part if needed, 
        # but CLIPTextModel is the specific text encoder class.
        self.text_encoder = CLIPTextModel.from_pretrained(
            text_model_path, 
            local_files_only=True
        )
        self.text_processor = CLIPProcessor.from_pretrained(
            text_model_path, 
            local_files_only=True
        )
        
        # 2. Load Image Encoder (ViT)
        self.image_encoder = ViTModel.from_pretrained(
            image_model_path, 
            local_files_only=True
        )
        self.image_processor = ViTFeatureExtractor.from_pretrained(
            image_model_path, 
            local_files_only=True
        )

        # 3. Freeze Backbones
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        # 4. Define Dimensions
        self.text_dim = self.text_encoder.config.hidden_size  # Usually 512 for CLIP Base
        self.image_dim = self.image_encoder.config.hidden_size # Usually 768 for ViT Base
        
        # 5. Projection Layers to Common Dimension
        # Project Text Embedding -> Common Dim
        self.text_projection = nn.Linear(self.text_dim, common_dim)
        # Project Image Embedding -> Common Dim (Identity if dims match, but we use Linear for flexibility)
        self.image_projection = nn.Linear(self.image_dim, common_dim)
        
        self.common_dim = common_dim

    def forward(self, input_ids, attention_mask, pixel_values):
        """
        Args:
            input_ids: Tokenized text input (Batch, SeqLen)
            attention_mask: Text attention mask (Batch, SeqLen)
            pixel_values: Processed image tensor (Batch, Channels, H, W)
        
        Returns:
            combined_tokens: Tensor of shape (Batch, 2, CommonDim)
                             [Projected_Text_Token, Projected_Image_Token]
        """
        self.text_encoder.eval()
        self.image_encoder.eval()

        # --- Text Encoding ---
        # Get CLIP text embeddings
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output (EOS token representation) as the text token
        # Shape: (Batch, 512)
        text_embeds = text_outputs.pooler_output 
        
        # --- Image Encoding ---
        # Get ViT image embeddings
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        # Use the CLS token (first token) as the image token
        # Shape: (Batch, 768)
        image_embeds = image_outputs.pooler_output 

        # --- Projection ---
        # Project to common dimension
        text_proj = self.text_projection(text_embeds)   # (Batch, CommonDim)
        image_proj = self.image_projection(image_embeds) # (Batch, CommonDim)

        # --- Concatenation ---
        # Stack them to create a sequence of 2 tokens: [Text, Image]
        # Unsqueeze to add sequence dimension: (Batch, 1, Dim)
        text_proj = text_proj.unsqueeze(1)
        image_proj = image_proj.unsqueeze(1)
        
        # Concat along sequence dim (dim=1)
        combined_tokens = torch.cat([text_proj, image_proj], dim=1) 
        
        return combined_tokens

def test_embedder():
    """
    Simple test function to verify shapes and execution.
    """
    print("--- Starting Multimodal Embedder Test ---")
    
    # Setup Paths (Ensure you ran download_weights.py first)
    text_path = "../ckpts/clip-vit-base-patch32"
    image_path = "../ckpts/vit-base-patch16-224"
    
    if not os.path.exists(text_path) or not os.path.exists(image_path):
        print("Error: Local weights not found. Please run download_weights.py first.")
        return

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalEmbedder(text_path, image_path, common_dim=768).to(device)
    print(f"Model loaded on {device}")

    # Prepare Dummy Data
    # Text
    dummy_text = ["A picture of a cat", "An instruction to rotate the image"]
    text_inputs = model.text_processor(text=dummy_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)
    
    # Image (Create a dummy black image)
    dummy_image = Image.new('RGB', (224, 224), color='black')
    # Process a batch of 2 images to match text batch size
    images = [dummy_image, dummy_image] 
    image_inputs = model.image_processor(images=images, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)

    # Forward Pass
    with torch.no_grad():
        output = model(input_ids, attention_mask, pixel_values)

    # Verify Shapes
    # Expected: (Batch_Size, Num_Tokens, Common_Dim) -> (2, 2, 768)
    print(f"\nInput Text Batch: {input_ids.shape}")
    print(f"Input Image Batch: {pixel_values.shape}")
    print(f"Output Tensor Shape: {output.shape}")
    
    expected_shape = (2, 2, 768) # 2 samples, 2 tokens (text+img), 768 dim
    if output.shape == expected_shape:
        print("\n[SUCCESS] Output shape matches expected (Batch, 2 Tokens, Common_Dim).")
    else:
        print(f"\n[WARNING] Output shape {output.shape} does not match expected {expected_shape}.")

    # Check if gradients are frozen
    text_param_frozen = all(not p.requires_grad for p in model.text_encoder.parameters())
    image_param_frozen = all(not p.requires_grad for p in model.image_encoder.parameters())
    proj_param_trainable = any(p.requires_grad for p in model.text_projection.parameters())
    
    print(f"Text Encoder Frozen: {text_param_frozen}")
    print(f"Image Encoder Frozen: {image_param_frozen}")
    print(f"Projection Layers Trainable: {proj_param_trainable}")

if __name__ == "__main__":
    test_embedder()