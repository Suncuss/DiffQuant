def train_model(train_csv_path, eval_csv_path, model_weight_path):
    import torch
    import config
    import torch.nn as nn
    import torch.optim as optim
    from data_loader import create_data_loaders, load_expanded_dataset_from_csv
    from model import getResNet18RegressionModel
    import torch.nn.functional as F

    # Set up hyperparameters
    batch_size = config.BATCH_SIZE
    num_epochs = config.EPOCHS
    device = torch.device(config.DEVICE)

    # Load the dataset
    train_dataset = load_expanded_dataset_from_csv(train_csv_path)
    eval_dataset = load_expanded_dataset_from_csv(eval_csv_path)

    # Create data loaders
    train_loader, eval_loader = create_data_loaders(train_dataset, eval_dataset, batch_size)

    # Instantiate the model and Define the loss function and optimizer
    model = getResNet18RegressionModel.to(device)
    criterion = nn.MSELoss()
    learning_rate = config.LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if config.REGRESSION:
                loss = criterion(outputs.squeeze(), labels)  # Squeeze the output tensor to match the shape of the labels
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model on the validation set
        model.eval()
        total_mae = 0.0
        total = 0
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # Calculate Mean Absolute Error (MAE) for the current batch
                mae = F.l1_loss(outputs.squeeze(), labels, reduction='sum').item()

                total_mae += mae
                total += labels.size(0)
        mean_mae = total_mae / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation MAE: {mean_mae:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_weight_path)
