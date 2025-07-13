import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import Accuracy
import copy

def train(device, model, num_epochs, train_loader, val_loader, criterion, optimizer, results_dir):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    accuracy = Accuracy(task='multiclass', num_classes=model.output_dim, average='micro').to(device)

    #early stopping
    best_val_loss = float('inf')
    patience = 5  # you can tune this
    epochs_no_improve = 0
    early_stop = False

    best_model_state = None  # to store the best model


    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        #reset metrics for next epoch
        val_loss = 0.0
        accuracy.reset()
        #val_correct = 0
        #val_total = 0


        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()


                #_, predicted = torch.max(outputs.data, 1)
                #val_total += targets.size(0)
                #val_correct += (predicted == targets).sum().item()

                preds = torch.argmax(outputs, dim=1)

                accuracy.update(preds, targets)
                #precision.update(preds, targets)
                #recall.update(preds, targets)
                #f1.update(preds, targets)

        val_epoch_loss = val_loss / len(val_loader)
        #val_epoch_acc = 100 * val_correct / val_total
        val_epoch_acc = accuracy.compute().item() * 100
        #val_epoch_precision = precision.compute().item()
        #val_epoch_recall = recall = recall.compute().item()
        #val_epoch_f1 = f1.compute().item()

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
            f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
        
        # save best model if improvement
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())  # save best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                early_stop = True
                break


    #Visualize loss and accuracy
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.savefig(results_dir + 'loss_plot.png')  # Save loss plot
    plt.close()  # Close the figure so it doesn't display


    plt.figure()
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.savefig(results_dir + 'accuracy_plot.png')  # Save accuracy plot
    plt.close()  # Close the figure

    #reload best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the model
    #torch.save(model, 'clustered_protein_mlp.pth')

    return model