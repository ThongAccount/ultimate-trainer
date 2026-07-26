import torch

class TrainStepGraph:
    def __init__(self, layer, batch_size):
        self.layer = layer
        self.B = batch_size
        self.K = layer.in_features
        self.N = layer.out_features
        
        # Static tensors
        self.static_x = torch.zeros(self.B, self.K, dtype=torch.float16, device="cuda")
        self.static_y_target = torch.zeros(self.B, self.N, dtype=torch.float16, device="cuda")
        
        # Warmup and capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                y = self.layer(self.static_x)
                loss = torch.nn.functional.mse_loss(y, self.static_y_target)
                loss.backward()
        torch.cuda.current_stream().wait_stream(s)
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_y = self.layer(self.static_x)
            self.static_loss = torch.nn.functional.mse_loss(self.static_y, self.static_y_target)
            self.static_loss.backward()
            
    def step(self, x, y_target):
        self.static_x.copy_(x)
        self.static_y_target.copy_(y_target)
        self.graph.replay()
        return self.static_loss.item()
