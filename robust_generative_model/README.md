# robust_generative_model

# Train Examples
```python 
python main.py --gpus 1 --kl_coeff 1 --enc_out_dim 1024 --recon_loss_type l2  --vae_type gfz 
```
```python 
python main.py --gpus 1 --kl_coeff 1 --enc_out_dim 1024 --vae_type gfz --binary --name binary 
```
```python 
python main.py --gpus 1 --kl_coeff 1 --enc_out_dim 1024 --recon_loss_type crf --vae_type gfz --binary --name binary_crf
```

# Test/Attack Examples
```python
python main.py --gpus 1 --kl_coeff 1 --enc_out_dim 1024 --recon_loss_type l2 --vae_type gfz  --attack --checkpoint mnist_tblogs/gfz/version_0/checkpoints/epoch\=141-step\=26695.ckpt --batch_size 16
```
```python 
python main.py --gpus 1 --kl_coeff 1 --enc_out_dim 1024 --vae_type gfz --binary --name binary --attack --checkpoint mnist_tblogs/gfz_binary/version_2/checkpoints/epoch\=80-step\=15227.ckpt --batch_size 16
```
The CRF version can be really slow.
```python
python main.py --gpus 1 --kl_coeff 1 --enc_out_dim 1024 --recon_loss_type crf --vae_type gfz --binary --name binary_crf --attack --checkpoint mnist_tblogs/gfz_binary_crf/version_13/checkpoints/epoch\=101-step\=19175.ckpt --batch_size 8
```
