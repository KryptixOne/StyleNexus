"""
Stable Diffusion With Diffusers Library

Three separate pre-trained Models:
1. The Variational Auto-Encoder
2. The Unet
3. The CLIP Text Encoder

"""
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

from tqdm.auto import tqdm


civitaiModel = r'D:\StableDiffusionWebUI\stable-diffusion-webui\models\Stable-diffusion\lyriel_v16.safetensors'
convertedModel = r'D:\Ecommerce_FakeModel\Models_Converted\Lyriel_Diffusers' #diffusion_pytorch_model

# Preload Models
print('loading...')
vae = AutoencoderKL.from_pretrained(convertedModel,subfolder = 'vae')
tokenize = CLIPTokenizer.from_pretrained(convertedModel,subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained(convertedModel,subfolder= 'text_encoder')
unet = UNet2DConditionModel.from_pretrained(convertedModel,subfolder='unet')
scheduler = PNDMScheduler.from_pretrained(convertedModel,subfolder='scheduler')
print('...loaded')

# Load to GPU
print('Moving to Cuda...')
torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)
print('... Moved')

# Text embeddings:
prompt = ['capitals girl with a sailor red cap,red and black color clothes anime key visual full body portrait character concept art, commander with flowing brunette hair green eyes,long straight black hair, brutalist grimdark fantasy, kuudere noble dictator, trending pixiv fanbox, rule of thirds golden ratio, by greg rutkowski wlop makoto shinkai takashi takeuchi studio ghibli jamie wyeth,bw, (natural skin texture, hyperrealism, soft light, sharp)']
height = width = 512
num_inference_steps = 25  # Number of Denoising Steps
guidance_scale = 7  # Classifier Free Guidance (CFG)
generator = torch.manual_seed(0)  # init Latent Noise Seed
batch_size = len(prompt)

# Tokenize the text and generate the embeddings:
text_input = tokenize(prompt, padding='max_length', max_length=tokenize.model_max_length, truncation=True,
                      return_tensors='pt')

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# Generate Unconditional Text Embeddings which are the embeddings for the padding token.
# Must have same shape  as the conditional text_embeddings
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenize([''] * batch_size, padding='max_length', max_length=max_length, return_tensors='pt')
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

# Concat the Conditional and unconditional embeddings into a batch to avoid a doing a two forward pass:
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Create Random Noise:
# We will start by creating the latent representation so it will be smaller than the final image.
# 2 ** (len(vae.config.block_out_channels) - 1) == 8 because the model has 3 downsampling layers, hence we divide by 8

latents = torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8),
                      generator=generator)  # generator is the seed

# Denoising the Image:
# We start by scaling the input with the init noise distribution, sigma, the noise scale value.
latents = latents * scheduler.init_noise_sigma

# Finally, we create the denoising loop that will progressively transform the pure noise in the latents to an image.
# Loop must do the following:
# 1. Set the schedulers timesteps to use during denoising
# 2. Iterate over the timesteps
# 3. At each timestep call the UNet to predict the noise residual and pass it to the scheduler to compute the prev sample

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    latent_model_input = torch.cat(
        [latents] * 2)  # expand latents if we are doing CFG to avoid doing two forward passes

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    with torch.no_grad():
        noise_pred = unet(latent_model_input.to(torch_device), t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text-noise_pred_uncond)

    # compute the previous noisy sample x_t --> x_t -1
    latents = scheduler.step(noise_pred, t, latents.to(torch_device)).prev_sample

# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample


# And then just convert image to a PIL Image:
image = image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
plt.imshow(pil_images[0])
plt.show()
