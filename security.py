from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64

import torch

encryption_key = b'project_group14a'

def encrypt_tensor(tensor, key):
    """
    Encrypt a PyTorch tensor using AES encryption.
    """
    cipher = AES.new(key, AES.MODE_ECB)  # Use ECB mode for simplicity
    # Move tensor to CPU before converting to NumPy array
    tensor_bytes = tensor.cpu().numpy().tobytes()  # Convert tensor to bytes
    encrypted = cipher.encrypt(pad(tensor_bytes, AES.block_size))
    return base64.b64encode(encrypted).decode('utf-8')   # Encode as base64 for readability

def decrypt_tensor(encrypted_tensor, shape, dtype, key):
    """
    Decrypt a PyTorch tensor using AES decryption.
    """
    cipher = AES.new(key, AES.MODE_ECB)
    encrypted_bytes = base64.b64decode(encrypted_tensor)
    decrypted_bytes = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)
    tensor = torch.frombuffer(decrypted_bytes, dtype=dtype).reshape(shape)
    return tensor