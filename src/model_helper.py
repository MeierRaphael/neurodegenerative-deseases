import os
import pickle
import time
import torch
import math
import seaborn as sns
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from datetime import datetime

from load_data import split_dataset
from colors import colors_dark

sns.set_context('talk')


class ModelHelper:
    device = None

    model = None
    model_name = ""
    labels = None
    labels_to_idx = None

    history = {}
    computation_time = None
    evaluation = None

    dataloaders = None

    train_dataset = None
    val_dataset = None
    test_dataset = None

    is_transformer = None

    def __init__(self, model, dataset, device, batch_size=16, model_name="NN", is_transformer=False,
                 class_names=None):
        self.device = device

        self.model = model
        self.model_name = model_name

        if class_names is None:
            class_names = ['Alzheimer', 'Gesund', 'Parkinson']
        self.labels = class_names
        self.labels_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.train_dataset, self.val_dataset, self.test_dataset = split_dataset(dataset)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

        self.is_transformer = is_transformer

    def train_model(self, criterion, optimizer, num_epochs=20):
        dataset_sizes = {'train': len(self.train_dataset),
                         'val': len(self.val_dataset),
                         'test': len(self.test_dataset)}

        since = time.time()

        best_model_wts = self.model.state_dict()
        best_acc = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [] }

        scaler = GradScaler(str(self.device))

        print('Device: ' + str(self.device))
        for epoch in range(num_epochs):
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with autocast(str(self.device)):
                        outputs = self.model(inputs) if not self.is_transformer else self.model(inputs).logits
                        loss = criterion(outputs, labels)

                    with torch.set_grad_enabled(phase == 'train'):
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            # Backward pass with scaled loss
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100

                print(f'{phase} Loss: {epoch_loss:.4f} {phase} Acc: {epoch_acc:.2f}%')

                # statistics
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc.item())
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc.item())

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()
            print('--------------------------------------------------------------')

        time_elapsed = time.time() - since
        self.computation_time = time_elapsed

        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        self.model.load_state_dict(best_model_wts)  # TODO richtig?

    def evaluate_model(self, criterion=torch.nn.CrossEntropyLoss()):
        self.model.eval()
        correct, total, test_loss = 0, 0, 0.0

        y_trues, y_scores, y_preds = [], [], []
        incorrect_predictions = {"images": [], "true_labels": [], "predicted_labels": []}

        with torch.no_grad():
            for images, labels in self.dataloaders['test']:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images) if not self.is_transformer else self.model(images).logits
                test_loss += criterion(outputs, labels).item() * images.size(0)  # Batch loss

                _, predicted = torch.max(outputs, 1)
                y_trues.append(labels.cpu())
                y_scores.append(outputs.cpu())
                y_preds.append(predicted.cpu())

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Store incorrect predictions
                incorrect_mask = (predicted != labels).cpu()
                incorrect_predictions["images"].extend(images[incorrect_mask].cpu())
                incorrect_predictions["true_labels"].extend(labels[incorrect_mask].cpu().tolist())
                incorrect_predictions["predicted_labels"].extend(predicted[incorrect_mask].cpu().tolist())

        test_accuracy = 100 * correct / total
        test_loss /= total
        print(f'Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}')

        # Convert list of tensors to numpy arrays
        y_trues = torch.cat(y_trues).cpu().numpy()
        y_scores = torch.cat(y_scores).cpu().numpy()
        y_preds = torch.cat(y_preds).cpu().numpy()

        self.evaluation = {
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "y_trues": y_trues,
            "y_scores": y_scores,
            "y_preds": y_preds,
            "incorrect_predictions": incorrect_predictions
        }

    def plot_incorrect_images(self, max_columns=5):
        incorrect_predictions = self.evaluation['incorrect_predictions']
        combined = list(zip(incorrect_predictions["images"], incorrect_predictions["true_labels"],
                            incorrect_predictions["predicted_labels"]))
        sorted_combined = sorted(combined, key=lambda x: x[2])  # Sort by predicted_labels
        sorted_images, sorted_labels, sorted_predictions = zip(*sorted_combined)

        num_images = len(sorted_images)
        num_columns = min(max_columns, num_images)
        num_rows = math.ceil(num_images / num_columns)

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 3, num_rows * 3))
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                img = sorted_images[i].numpy().transpose(1, 2, 0)
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(
                    f'R: {self.labels[sorted_labels[i]]}, \nV: {self.labels[sorted_predictions[i]]}')
                ax.axis('off')
            else:
                ax.axis('off')  # Disable unused subplots
        plt.tight_layout()
        plt.show()

    def plot_multiclass_roc(self, n_classes=3, zoom_area=(-0.02, 0.05, 0.95, 1.02)):
        y_trues_binarized = label_binarize(self.evaluation['y_trues'], classes=list(range(n_classes)))
        fpr, tpr, roc_auc = {}, {}, {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_trues_binarized[:, i], self.evaluation['y_scores'][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Create the main ROC plot
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'{self.labels[i]} (AUC = {roc_auc[i]:.3f})',
                    color=list(colors_dark.values())[i])

        # Plot settings for the main plot
        ax.set_xlabel('Falsch-Positive-Rate')
        ax.set_ylabel('Richtig-Positive-Rate')
        ax.set_title(f'Multiklassen-ROC-Kurve {self.model_name}')
        ax.legend(loc="lower right")

        # Create inset axis (zoomed area) in the center
        inset_ax = inset_axes(ax, width="40%", height="40%", loc='center')
        for i in range(n_classes):
            inset_ax.plot(fpr[i], tpr[i], label=f'{self.labels[i]} (AUC = {roc_auc[i]:.3f})',
                          color=list(colors_dark.values())[i])

        x1, x2, y1, y2 = zoom_area
        inset_ax.set_xlim(x1, x2)
        inset_ax.set_ylim(y1, y2)

        mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", lw=2, ls="--")

        plt.savefig(f'../images/roc/{self.model_name}.png', transparent=True)
        plt.show()

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.evaluation['y_trues'], self.evaluation['y_preds'])

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels,
                    yticklabels=self.labels, ax=ax, cbar=False)
        ax.set_xlabel('Vorhergesagte Klasse')
        ax.set_ylabel('Tatsächliche Klasse')
        ax.set_title(f'Konfusionsmatrix {self.model_name}')

        plt.savefig(f'../images/confusion_matrix/{self.model_name}.png', transparent=True)
        plt.show()

    def plot_losses_and_accuracies(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot Loss
        ax1.plot(self.history['train_loss'], label='Training Loss', color=colors_dark['blue'], marker='o')
        ax1.plot(self.history['val_loss'], label='Validation Loss', color=colors_dark['green'], marker='o')
        ax1.set_title(f'Trainings-/Validierungs-Verlust {self.model_name}')
        ax1.set_xlabel('Epochen')
        ax1.set_ylabel('Verlust')
        ax1.legend()
        ax1.grid(True)

        # Plot Accuracy
        ax2.plot(self.history['train_acc'], label='Training Accuracy', color=colors_dark['blue'], marker='o')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy', color=colors_dark['green'], marker='o')
        ax2.set_title(f'Trainings-/Validierungs-Genauigkeit {self.model_name}')
        ax2.set_xlabel('Epochen')
        ax2.set_ylabel('Genauigkeit')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        plt.savefig(f'../images/training/{self.model_name}.png', transparent=True)
        plt.show()

    def save_model(self, folder='models'):
        # Create the directory if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'{folder}/{self.model_name}_{timestamp}-{self.evaluation["test_accuracy"]:.2f}.pth'

        # Save the instance using pickle
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f'ModelHelper saved to {filename}')
        except Exception as e:
            print(f'Error saving ModelHelper: {e}')

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model_helper = pickle.load(f)
        return model_helper
