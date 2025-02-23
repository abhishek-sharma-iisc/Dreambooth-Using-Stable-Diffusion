
# Dreambooth using Stable Diffusion
<img src="https://github.com/user-attachments/assets/f0f900e2-a13f-4460-8952-5c428b9c01ff">

<p align="center"><b>Abstract</b></p>
Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for “personalization” of text-to-image diffusion models. Given as input just a few images of a subject, we finetune a pretrained text-to-image model such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can be used to synthesize novel photorealistic images of the subject contextualized in different scenes. By leveraging the semantic prior embedded in the model with a new autogenous class-specific prior preservation loss, our technique enables synthesizing the subject in diverse scenes, poses, views and lighting conditions that do not appear in the reference images. We apply our technique to several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, and artistic rendering, all while preserving the subject’s key features. We also provide a new dataset and evaluation protocol for this new task of subject-driven generation.
</br>
</br>

Complete Paper can be accessed here: [DreamBooth: Fine Tuning Text-to-Image Diffusion Models
for Subject-Driven Generation](https://arxiv.org/pdf/2208.12242)

**NOTE:** Since, Google has not made Dreambooth code and U-Net model public, I have used the stable-diffusionv1.5 to fine tune subject specific images and code for training is taken from [Textual-Inversion](https://github.com/rinongal/textual_inversion) repository. I have made some changes in code in order to align it for our purpose since textual inversion base code doesn't finetune the U-Net model which is a requirement in dreambooth implementation.

# Index

- [Introduction](#introduction)
- [Implementation Details](#implementation)
  - [Preparation](#preparation)
  - [Training](#training)
  - [Conda Setup](#running-locally-conda)
    
- [Configuration File and Command Line Reference](#config-file-and-command-line-reference)
  - [For Training using Custom configuration](#custom-config) 
- [Results using this Generation Model](#results)
- [Textual Inversion vs. Dreambooth](#text-vs-dreamb)

## <a name="introduction"></a> Introduction
This repository presents an adaptation of Google's Dreambooth, utilizing Stable Diffusion. The original Dreambooth was built upon the Imagen text-to-image model. However, neither the model nor its pre-trained weights are accessible. To facilitate fine-tuning of a text-to-image model with limited examples, I've incorporated the concept of Dreambooth into Stable Diffusion.

The foundation of this code repository is based on Textual Inversion. It's important to note that Textual Inversion solely optimizes word embedding, whereas Dreambooth fine-tunes the entire diffusion model.

## <a name="implementation"></a>Implementation Details
As already mentioned that we are using the most architecture part of [Textual Inversion]() repository since Google has not made dreambooth code public. Note that Textual inversion paper only discusses about training the embedding vector and not the U-Net architecture which is used for generation. But since dreambooth implementation requires fine tuning the U-Net architecture hence I will be modifying the codebase at this [line](), which disable gradient checkpointing in a hard-code way. This is because in textual inversion, the Unet is not optimized. However, in Dreambooth we optimize the Unet, so we can turn on the gradient checkpoint pointing trick, as in the original [Stable Diffusion]() repo here. The gradient checkpoint is default to be True in config. I have updated the codes.

- Onto the technical side:
  - We can now run this on a GPU with **24GB of VRAM** (e.g. 3090). Training will be slower, and you'll need to be sure this is the *only* program running.
  - I have trained it on my server provided by my college but I'm including a Jupyter notebook here to help you run it on a rented cloud computing platform if you don't own one. 
  
  
- This implementation does not fully implement Google's ideas on how to preserve the latent space.

  - Most images that are similar to what you're training will be shifted towards that.
  - e.g. If you're training a person, all people will look like you. If you're training an object, anything in that class will look like your object.

  
### <a name="preparation"></a> Preparation
First set-up the ldm enviroment following the instruction from textual inversion repo, or the original Stable Diffusion repo.

To fine-tune a stable diffusion model, we need to obtain the pre-trained stable diffusion models following their instructions. Weights can be downloaded from HuggingFace. You can decide which version of checkpoint to use, but I use `sd_v1-5_vae.ckpt` present in [hugging_face](https://huggingface.co/panopstor/EveryDream/tree/main).

We also need to create a set of images for regularization, as the fine-tuning algorithm of Dreambooth requires that. Details of the algorithm can be found in the paper. Note that in the original paper, the regularization images seem to be generated on-the-fly. However, here I generated a set of regularization images before the training. The text prompt for generating regularization images can be `photo of a <class>`, where `<class>` is a word that describes the class of your object, such as `dog`. The command is

```
python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 1 --scale 10.0 --ddim_steps 50  --ckpt /path/to/original/stable-diffusion/sd-v1-4-full-ema.ckpt --prompt "a photo of a <class>" 
```

I generated 1500 images for regularization. After generating regularization images, save them in `/root/to/regularization_images` folder.
If the generated regularization images are highly unrealistic ("man" or "woman"), you can find a diverse set of images (of man/woman) online, and use them as regularization images. This can give a better result.

### <a name="training"></a>Training 

I have trained it in a conda environment using python 3.10 on my server. One can implement in any virtual environment system. One can use any particular subject to train this model. In my case, I have used my own images for training purpose. To train using my images, I have taken 22 images of mine in which my face has frontal pose in most images. Some images also contain side poses for diversity and novel viewpoint training.

### <a name="generation"></a>Generation 

After training, personalized samples can be obtained by running the command

```
python scripts/stable_txt2img.py --ddim_eta 0.0 
                                 --n_samples 8 
                                 --n_iter 1 
                                 --scale 10.0 
                                 --ddim_steps 100  
                                 --ckpt /path/to/saved/checkpoint/from/training
                                 --prompt "photo of a sks <class>" 
```

In particular, `abhishek` is the identifier, which should be replaced by your choice if you happen to change the identifier, and `<class>` is the class word --class_word for training.
During Generation, all subject token is reffered as `[subject_identifier][subject_class]` as mentioned in Dreambooth paper. In the main paper, they suggest to use 3-4 letter token for subject identifier but I used 8 letter token whihch worked quite well. In this case, I have used `abhishek person` as token where `abhishek` is subject identifier and `person` is subject class.

### <a name="running-locally-conda"></a>  Setup - Conda

### Pre-Requisites
1. [Git](https://gitforwindows.org/)
2. [Python 3.10](https://www.python.org/downloads/)
2. [miniconda3](https://docs.conda.io/en/latest/miniconda.html)
3. Open `Anaconda Prompt (miniconda3)`
4. Clone the repository
   1. `(base) C:\>git clone https://github.com/abhisheksh1304/Dreambooth-Using-Stable-Diffusion.git`
5. Navigate into the repository
   1. `(base) C:\>cd Dreambooth-using-Stable-Diffusion`

Install Dependencies and Activate Environment
```
(base) C:\Dreambooth-Stable-Diffusion> conda env create -f environment.yaml
(base) C:\Dreambooth-Stable-Diffusion> conda activate dreambooth_joepenna
```

### Run

```
cmd> python "main.py" --project_name "ProjectName" --training_model "C:\v1-5-pruned-emaonly-pruned.ckpt" --regularization_images "C:\regularization_images" --training_images "C:\training_images" --max_training_steps 2000 --class_word "person" --token "zwx" --flip_p 0 --learning_rate 1.0e-06 --save_every_x_steps 250
```

### Cleanup
```
cmd> conda deactivate

```

# <a name="configure"></a> Configuration File and Command Line Reference

## Example Configuration file

```
{
    "class_word": "woman",
    "config_date_time": "2023-04-08T16-54-00",
    "debug": false,
    "flip_percent": 0.0,
    "gpu": 0,
    "learning_rate": 1e-06,
    "max_training_steps": 3500,
    "model_path": "D:\\stable-diffusion\\models\\v1-5-pruned-emaonly-pruned.ckpt",
    "model_repo_id": "",
    "project_config_filename": "my-config.json",
    "project_name": "<token> project",
    "regularization_images_folder_path": "D:\\stable-diffusion\\regularization_images\\Stable-Diffusion-Regularization-Images-person_ddim\\person_ddim",
    "save_every_x_steps": 250,
    "schema": 1,
    "seed": 23,
    "token": "<token>",
    "token_only": false,
    "training_images": [
        "001@a photo of <token> looking down.png",
        "002-DUPLICATE@a close photo of <token> smiling wearing a black sweatshirt.png",
        "002@a photo of <token> wearing a black sweatshirt sitting on a blue couch.png",
        "003@a photo of <token> smiling wearing a red flannel shirt with a door in the background.png",
        "004@a photo of <token> wearing a purple sweater dress standing with her arms crossed in front of a piano.png",
        "005@a close photo of <token> with her hand on her chin.png",
        "005@a photo of <token> with her hand on her chin wearing a dark green coat and a red turtleneck.png",
        "006@a close photo of <token>.png",
        "007@a close photo of <token>.png",
        "008@a photo of <token> wearing a purple turtleneck and earings.png",
        "009@a close photo of <token> wearing a red flannel shirt with her hand on her head.png",
        "011@a close photo of <token> wearing a black shirt.png",
        "012@a close photo of <token> smirking wearing a gray hooded sweatshirt.png",
        "013@a photo of <token> standing in front of a desk.png",
        "014@a close photo of <token> standing in a kitchen.png",
        "015@a photo of <token> wearing a pink sweater with her hand on her forehead sitting on a couch with leaves in the background.png",
        "016@a photo of <token> wearing a black shirt standing in front of a door.png",
        "017@a photo of <token> smiling wearing a black v-neck sweater sitting on a couch in front of a lamp.png",
        "019@a photo of <token> wearing a blue v-neck shirt in front of a door.png",
        "020@a photo of <token> looking down with her hand on her face wearing a black sweater.png",
        "021@a close photo of <token> pursing her lips wearing a pink hooded sweatshirt.png",
        "022@a photo of <token> looking off into the distance wearing a striped shirt.png",
        "023@a photo of <token> smiling wearing a blue beanie holding a wine glass with a kitchen table in the background.png",
        "024@a close photo of <token> looking at the camera.png"
    ],
    "training_images_count": 24,
    "training_images_folder_path": "D:\\stable-diffusion\\training_images\\24 Images - captioned"
}
```

## <a name="custom-config"></a>For Training using Custom configuration

## Command Line Parameters

[dreambooth_helpers\arguments.py]()

| Command | Type | Example | Description |
| ------- | ---- | ------- | ----------- |
| `--config_file_path` | string | `"C:\\Users\\David\\Dreambooth Configs\\my-config.json"` | The path the configuration file to use |
| `--project_name` | string | `"My Project Name"` | Name of the project |
| `--debug` | bool | `False` | *Optional* Defaults to `False`. Enable debug logging |
| `--seed` | int | `23` | *Optional* Defaults to `23`. Seed for seed_everything |
| `--max_training_steps` | int | `3000` | Number of training steps to run |
| `--token` | string | `"owhx"` | Unique token you want to represent your trained model. |
| `--token_only` | bool | `False` | *Optional* Defaults to `False`. Train only using the token and no class. |
| `--training_model` | string | `"D:\\stable-diffusion\\models\\v1-5-pruned-emaonly-pruned.ckpt"` | Path to model to train (model.ckpt) |
| `--training_images` | string | `"D:\\stable-diffusion\\training_images\\24 Images - captioned"` | Path to training images directory |
| `--regularization_images` | string | `"D:\\stable-diffusion\\regularization_images\\Stable-Diffusion-Regularization-Images-person_ddim\\person_ddim"` | Path to directory with regularization images |
| `--class_word` | string | `"woman"` | Match class_word to the category of images you want to train. Example: `man`, `woman`, `dog`, or `artstyle`. |
| `--flip_p` | float | `0.0` | *Optional* Defaults to `0.5`. Flip Percentage. Example: if set to `0.5`, will flip (mirror) your training images 50% of the time. This helps expand your dataset without needing to include more training images. This can lead to worse results for face training since most people's faces are not perfectly symmetrical. |
| `--learning_rate` | float | `1.0e-06` | *Optional* Defaults to `1.0e-06` (0.000001). Set the learning rate. Accepts scientific notation. |
| `--save_every_x_steps` | int | `250` | *Optional* Defaults to `0`. Saves a checkpoint every x steps.   At `0` only saves at the end of training when `max_training_steps` is reached. |
| `--gpu` | int | `0` | *Optional* Defaults to `0`. Specify a GPU other than 0 to use for training.  Multi-GPU support is not currently implemented.

## Using your configuration for training

```
python "main.py" --project_name "My Project Name" --max_training_steps 3000 --token "owhx" --training_model "D:\\stable-diffusion\\models\\v1-5-pruned-emaonly-pruned.ckpt" --training_images "D:\\stable-diffusion\\training_images\\24 Images - captioned" --regularization_images "D:\\stable-diffusion\\regularization_images\\Stable-Diffusion-Regularization-Images-person_ddim\\person_ddim" --class_word "woman" --flip_p 0.0 --save_every_x_steps 500
```

# <a name="results"></a>Results Using this Generation Model 

As already mentioned above, I have used my own images for personalized training of this dreambooth model.

## Ground Truth Images
The `ground truth` (real pictures used for training)
<br><img src="https://github.com/user-attachments/assets/5951095d-9265-40ea-bd57-2a5d653251e4" width="190">
<img src="https://github.com/user-attachments/assets/c3a06220-cf28-4a29-a65a-1526a19bcff8" width="190">
<img src="https://github.com/user-attachments/assets/79b51f73-eb97-40c1-ae8a-683a4d10afaf6" width="190">
<img src="https://github.com/user-attachments/assets/1ad63a09-c41a-4136-8531-01460866880b" width="190">


## Model Generated Images

| Prompt: `abhishek person as a masterpiece portrait painting by John Singer Sargent in the style of Rembrandt` |
|---------|
| <img src="https://github.com/user-attachments/assets/74377c6a-f0ed-45b2-8bc7-0078cbfad749" width="190"> <img src="https://github.com/user-attachments/assets/fc2e3739-c1e5-4161-889c-e190cebea162" width="190"> <img src="https://github.com/user-attachments/assets/9d240c1c-fd87-49b6-8ce7-2f7f85ab9041" width="190"> <img src="https://github.com/user-attachments/assets/ac3c95cb-ec2b-49eb-b82e-85bc79727ff3" width="190"> |   

| Prompt: `abhishek person with a tree and autumn leaves in the background` |
|---------|
| <img src="https://github.com/user-attachments/assets/f85416b8-309b-494a-b86f-ae04b5bdb5ce" width="190"> <img src="https://github.com/user-attachments/assets/532873c4-0d60-4b66-bcd9-7d8d6117ba70" width="190"> <img src="https://github.com/user-attachments/assets/1c790751-3c8d-4ee9-980b-5f2ad6b83793" width="190"> <img src="https://github.com/user-attachments/assets/4e1532ef-3782-45c7-b3e8-464aa22d38e3" width="190"> |  

| Prompt: `abhishek person with the Eiffel Tower in the background` |
|---------|
| <img src="https://github.com/user-attachments/assets/71f71093-24a2-4a8b-94e9-cb070dafb666" width="190"> <img src="https://github.com/user-attachments/assets/fdf38387-0a48-4266-a724-dd838e879200" width="190"> <img src="https://github.com/user-attachments/assets/3c0c08d2-f6ac-4089-9206-1440a0a8ca4e" width="190"> <img src="https://github.com/user-attachments/assets/ae681448-a061-41bf-ae9b-2b1011265361" width="190"> |  

| Prompt: `abhishek token person walking on the beach` |
|---------|
| <img src="https://github.com/user-attachments/assets/6a06b0c8-3ec2-46e3-9e39-b84057249dac" width="190"> <img src="https://github.com/user-attachments/assets/96c32c2d-1075-4f87-8d7f-67bb282c216b" width="190"> <img src="https://github.com/user-attachments/assets/f743aff2-fa8f-487f-80be-0b3a498c2816" width="190"> <img src="https://github.com/user-attachments/assets/c674fd92-84c3-4daf-8ef2-729c8764c67f" width="190"> |  

# <a name="text-vs-dreamb"></a>  Textual Inversion vs. Dreambooth
The majority of the code in this repo was written by Rinon Gal et. al, the authors of the Textual Inversion research paper. Though a few ideas about regularization images and prior loss preservation (ideas from "Dreambooth") were added in. It's important to give credit for textual inversion repo to both the MIT team and the Google researchers.
