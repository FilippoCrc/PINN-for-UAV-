import torch
import matplotlib.pyplot as plt
from csv_dataset import QuadrotorDataset, create_dataloaders # OK
from pinn import QuadrotorPINN, PhysicsInformedLoss # OK
from trainer import train_pinn # OK
import numpy as np
import os # Per gestire i path

NUM_EPOCHS = 2000 # Riduci per debug iniziale se necessario
LEARNING_RATE = 1e-3
PHYSICS_LOSS_WEIGHT = 1000 # Puoi cambiare il peso qui
BATCH_SIZE = 128

def visualize_training_history(history):
    """Visualizza lo storico del training."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 10))

    # Loss Totale
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Total Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # MSE Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_mse'], label='Train MSE Loss')
    plt.plot(epochs, history['val_mse'], label='Validation MSE Loss')
    plt.title('MSE Loss Component')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Physics Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_physics'], label='Train Physics Loss')
    plt.plot(epochs, history['val_physics'], label='Validation Physics Loss')
    plt.title('Physics Loss Component')
    plt.xlabel('Epoch')
    plt.ylabel('Physics Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['lr'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, device, state_scaler):
    """Valuta il modello sul test set (MSE) e visualizza."""
    model.eval()
    test_mse_loss_accum = 0
    predictions_list = []
    targets_list = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        # --- MODIFICA: Unpack 3 items, usa solo i primi 2 per MSE ---
        for inputs, targets, _ in test_loader: # Ignora physics_info qui
            inputs = inputs.to(device)
            targets = targets.to(device) # Questi sono gli stati scalati

            predictions = model(inputs) # Predizioni scalate

            predictions_list.append(predictions.cpu())
            targets_list.append(targets.cpu())

            # Calcola MSE tra predizioni scalate e target scalati
            test_mse_loss_accum += torch.nn.functional.mse_loss(predictions, targets).item()

    all_predictions_scaled = torch.cat(predictions_list)
    all_targets_scaled = torch.cat(targets_list)

    avg_test_mse = test_mse_loss_accum / len(test_loader)
    print(f"Test MSE (on scaled data): {avg_test_mse:.6f}")

    # --- Visualizzazione (opzionale: denormalizza per vedere valori reali) ---
    if state_scaler:
        try:
            all_predictions_unscaled = state_scaler.inverse_transform(all_predictions_scaled.numpy())
            all_targets_unscaled = state_scaler.inverse_transform(all_targets_scaled.numpy())

            num_points_to_plot = min(200, len(all_targets_unscaled))
            plt.figure(figsize=(14, 7))

            # Esempio: Confronta la prima coordinata (x) e la prima velocità angolare (wx)
            plt.subplot(1, 2, 1)
            plt.plot(all_targets_unscaled[:num_points_to_plot, 0], label='True State[0] (x)', linestyle='--')
            plt.plot(all_predictions_unscaled[:num_points_to_plot, 0], label='Predicted State[0]', alpha=0.8)
            plt.title('Example: Position x (Test Set)')
            plt.xlabel('Time Step (in test sequence)')
            plt.ylabel('State Value (unscaled)')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            # wx è all'indice 9 nello stato x..wz
            plt.plot(all_targets_unscaled[:num_points_to_plot, 9], label='True State[9] (wx)', linestyle='--')
            plt.plot(all_predictions_unscaled[:num_points_to_plot, 9], label='Predicted State[9]', alpha=0.8)
            plt.title('Example: Angular Velocity wx (Test Set)')
            plt.xlabel('Time Step (in test sequence)')
            plt.ylabel('State Value (unscaled)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot unscaled data: {e}")
            # Fallback to scaled data plot if unscaling fails
            plt.figure(figsize=(12, 6))
            plt.plot(all_targets_scaled[:num_points_to_plot, 0].numpy(), label='True State[0] (scaled)', linestyle='--')
            plt.plot(all_predictions_scaled[:num_points_to_plot, 0].numpy(), label='Predicted State[0] (scaled)', alpha=0.8)
            plt.title('Example Trajectory Comparison (Test Set - SCALED)')
            plt.xlabel('Time Step (in test sequence)')
            plt.ylabel('State Value (scaled)')
            plt.legend()
            plt.grid(True)
            plt.show()

    return avg_test_mse

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Using device: {device}")

    # --- DEFINISCI I PATH CORRETTI QUI ---
    script_dir = os.path.dirname(__file__) # Directory dello script corrente
    base_dir = os.path.abspath(os.path.join(script_dir, '..')) # Directory PINN-for-UAV-
    state_folder = os.path.join(base_dir, "UAV_dataset", "state_dataset")
    input_folder = os.path.join(base_dir, "UAV_dataset", "input_dataset")
    print(f"Looking for data in:\n State: {state_folder}\n Input: {input_folder}")
    # ------------------------------------

    print("\nLoading dataset...")
    try:
        dataset = QuadrotorDataset(state_folder=state_folder, input_folder=input_folder)
        # --- MODIFICA: create_dataloaders ora ritorna anche gli scaler ---
        train_loader, val_loader, test_loader, state_scaler, input_scaler = create_dataloaders(
            dataset, batch_size=BATCH_SIZE
        )
    except FileNotFoundError:
        print(f"ERRORE: Path del dataset non trovato. Verifica:\n State: {state_folder}\n Input: {input_folder}")
        return
    except ValueError as e:
         print(f"ERRORE: Problema con i dati nel dataset: {e}")
         return
    except Exception as e:
        print(f"ERRORE durante il caricamento/scaling del dataset: {e}")
        return

    print(f"Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        print("ERRORE: Dataset vuoto.")
        return

    print("\nInitializing PINN...")
    model = QuadrotorPINN(input_dim=4, output_dim=12).to(device)

    # --- MODIFICA: Inizializza criterion, optimizer, scheduler qui ---
    criterion = PhysicsInformedLoss(state_scaler=state_scaler, lambda_physics=PHYSICS_LOSS_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=NUM_EPOCHS * len(train_loader),
        pct_start=0.3
    )

    print("\nStarting training...")
    # --- MODIFICA: Passa criterion, optimizer, scheduler a train_pinn ---
    history = train_pinn(
        model=model,
        criterion=criterion, # Passa l'istanza della loss
        optimizer=optimizer, # Passa l'optimizer
        scheduler=scheduler, # Passa lo scheduler
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS
    )

    print("\nVisualizing training history...")
    visualize_training_history(history)

    # --- MODIFICA: Passa lo state_scaler a evaluate_model ---
    print("\nEvaluating model on test set...")
    test_mse = evaluate_model(model, test_loader, device, state_scaler)

    print("\nSaving model...")
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), # Salva anche optimizer
            'scheduler_state_dict': scheduler.state_dict(), # Salva anche scheduler
            'history': history,
            'test_mse_scaled': test_mse,
            'input_dim': 4,
            'output_dim': 12,
             # Salva gli scaler per poter ricaricare e usare il modello
            'state_scaler_mean': state_scaler.mean_,
            'state_scaler_scale': state_scaler.scale_,
            'input_scaler_mean': input_scaler.mean_,
            'input_scaler_scale': input_scaler.scale_,
        }, 'trained_quadrotor_pinn.pth')
        print("Model saved to 'trained_quadrotor_pinn.pth'")
    except Exception as e:
        print(f"Errore durante il salvataggio del modello: {e}")

if __name__ == "__main__":
    main()