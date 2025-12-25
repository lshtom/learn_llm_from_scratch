# 数据并行DDP
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd layder
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs)
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return logits
    
torch.manual_seed(123)
# model = NeuralNetwork(num_inputs=2, num_outputs=2)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0
    for idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
                logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions              #A
        correct += torch.sum(compare)                #B
        total_examples += len(compare)
    return (correct / total_examples).item()         #C

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]
    
train_ids = ToyDataset(X_train, y_train)
test_ids = ToyDataset(X_test, y_test)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch) # shape (B,L,V)
    loss = torch.nn.functional.cross_entropy(logits.permute(0,2,1), target_batch)
    return loss

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def prepare_dataset():
    # 数据集
    train_loader = DataLoader(dataset=train_ids, batch_size=2, shuffle=False, drop_last=True, sampler=DistributedSampler(train_ids))
    test_loader = DataLoader(dataset=test_ids, batch_size=2, shuffle=False, drop_last=True, sampler=DistributedSampler(test_ids))
    return train_loader, test_loader

def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            loss = calc_loss_batch(features, labels, model, rank)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batchsize {labels.shape[0]:03d}"
              f" | Train/Val Loss: {loss:.2f}")
    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)
    destroy_process_group()


if __name__ == "__main__":
    # windows不支持NCCL
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
