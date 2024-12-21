if __name__ == "__main__":
    # Define dataset root and split file paths
    root_dir = {
        'word': '/path/to/WLASL/word'  # Replace with your dataset path
    }
    split_file = '/path/to/split_file.json'  # Replace with your split file path

    # Create DataLoaders
    dataloaders = get_dataloaders(
        root_dir=root_dir,
        split_file=split_file,
        batch_size=8,
        num_workers=4,
        num_frames=64
    )

    # Get number of classes
    num_classes = get_num_class(split_file=split_file)
    print(f"Number of classes: {num_classes}")

    # Initialize the model
    model = get_timesformer_model(num_classes=num_classes, pretrained=True)
    model = customize_model(model, num_classes)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Define loss function, optimizer, and scheduler
    freeze_layers(model, freeze_until=6)

    # Re-define optimizer to only update unfrozen parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # Optionally, define a different scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    num_epochs = 25
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device
    )

    # Save the best model
    torch.save(trained_model.state_dict(), 'best_timesformer_wlasl.pth')

    # Evaluate the model
    '''label_encoder = dataloaders['train'].dataset.label_encoder
    test_dataloader = dataloaders['test']
    evaluate_model(trained_model, test_dataloader, device, label_encoder)'''
