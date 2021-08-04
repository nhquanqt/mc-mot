def train_model(model, optimizer, criterion, scheduler, dataloader, n_epochs=100, val_interval=5):
    model.train()

    for epoch in range(n_epochs):
        print(f'epoch {epoch}')

        features, adj = dataloader()

        output = model(features, adj)

        loss = criterion(output)

        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f'loss = {loss.item()}')

    return model