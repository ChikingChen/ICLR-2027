import torch, time

dev = "cuda:0"
GB = 1024**3
size_gb = 4
repeat = 50
numel = size_gb * GB // 4

# CPU pinned memory -> GPU 显存
cpu = torch.empty(numel, dtype=torch.float32, pin_memory=True)
gpu = torch.empty(numel, dtype=torch.float32, device=dev)

for _ in range(5):
    gpu.copy_(cpu, non_blocking=True)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(repeat):
    gpu.copy_(cpu, non_blocking=True)
torch.cuda.synchronize()
t1 = time.time()

print(f"CPU RAM -> GPU VRAM: {size_gb * repeat / (t1 - t0):.2f} GB/s")

# GPU 显存内部 copy，近似 GPU 核心读写显存速度
a = torch.empty(numel, dtype=torch.float32, device=dev)
b = torch.empty_like(a)

for _ in range(5):
    b.copy_(a)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(repeat):
    b.copy_(a)
torch.cuda.synchronize()
t1 = time.time()

# copy 是读 a + 写 b，所以乘 2
print(f"GPU VRAM <-> GPU core bandwidth approx: {2 * size_gb * repeat / (t1 - t0):.2f} GB/s")
