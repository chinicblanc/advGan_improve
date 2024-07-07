# Adversarial Examples with Self-Attention GAN

This project is an enhancement of the work presented in the paper *Generating Adversarial Examples with Adversarial Networks*. It builds upon the code from [advGAN_pytorch](https://github.com/mathcbc/advGAN_pytorch.git) by incorporating self-attention mechanisms into both the generator and discriminator networks.

## Overview
The goal of this project is to improve the quality of adversarial examples by integrating self-attention mechanisms into the adversarial network architecture. The self-attention mechanism helps in capturing long-range dependencies, which can enhance the effectiveness of adversarial attacks.

## Key Features
- **Self-Attention Mechanism**: Added self-attention layers to both the generator and discriminator to improve the quality of generated adversarial examples.
- **Improved Adversarial Examples**: Aimed at generating more realistic and effective adversarial examples by leveraging self-attention.

## Usage
Under src/

### training the target model
```bash
python3 train_target_model.py
```
### training the advGAN
```bash
python3 main.py
```

### testing adversarial examples
```bash
python3 test_adversarial_examples.py
```