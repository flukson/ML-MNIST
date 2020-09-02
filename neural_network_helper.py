import torch
import torch.nn.functional as F

log_interval = 10

def train(network, epoch, data_loader, optimizer):

  network.train()
  for batch_idx, (data, target) in enumerate(data_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(data_loader.dataset),
        100. * batch_idx / len(data_loader), loss.item()))
      loss = loss.item()
      counter = batch_idx*64 + (epoch-1)*len(data_loader.dataset)
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')
  return loss, counter

def test(network, data_loader):

  network.eval()
  loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in data_loader:
      output = network(data)
      loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  loss /= len(data_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
  return loss
