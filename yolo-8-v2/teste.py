import torch

print(f"Versão do PyTorch: {torch.__version__}")
print("--- TESTE DE CUDA ---")
available = torch.cuda.is_available()
print(f"CUDA disponível? {available}")

if available:
    print(f"Quantidade de GPUs: {torch.cuda.device_count()}")
    print(f"GPU Atual: {torch.cuda.current_device()}")
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    
    # Este é o teste mais importante:
    try:
        print("Testando alocação de tensor na GPU...")
        a = torch.tensor([1.0, 2.0]).cuda()
        print(f"Tensor alocado com sucesso na GPU: {a}")
        print("AMBIENTE CUDA PARECE ESTAR OK!")
    except Exception as e:
        print(f"\n!!!!!!!! FALHA AO USAR A GPU !!!!!!!!")
        print(f"Erro: {e}")
else:
    print("\n!!!!!!!! NENHUMA GPU DETECTADA PELO PYTORCH !!!!!!!!")