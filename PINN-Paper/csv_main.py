import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from csv_dataset import QuadrotorDataset, create_dataloaders
from pinn import QuadrotorPINN, PhysicsInformedLoss_Control # Import new loss
from trainer import train_pinn
import numpy as np
import os

NUM_EPOCHS = 2000
LEARNING_RATE = 5e-4 # Adjust as needed
PHYSICS_LOSS_WEIGHT = 1.2 # Weight for rotational dynamics loss term (tune this!)
BATCH_SIZE = 64 # Ensure > 1

# --- MODIFIED: visualize_training_history (re-add physics plot) ---
def visualize_training_history(history):
    """Visualizza lo storico del training (MSE + Physics Loss)."""
    epochs = range(1, len(history['train_loss']) + 1)
    has_validation = 'val_loss' in history and history['val_loss']

    plt.figure(figsize=(18, 10)) # Back to wider layout

    # Loss Totale
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Total Loss')
    if has_validation: plt.plot(epochs, history['val_loss'], label='Validation Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # MSE Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_mse'], label='Train MSE Loss')
    if has_validation: plt.plot(epochs, history['val_mse'], label='Validation MSE Loss')
    plt.title('MSE Loss Component (Control Input)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Physics Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_physics'], label='Train Physics Loss')
    if has_validation: plt.plot(epochs, history['val_physics'], label='Validation Physics Loss')
    plt.title('Physics Loss Component (Rotational Dynamics)')
    plt.xlabel('Epoch')
    plt.ylabel('Physics Loss (log scale)')
    plt.yscale('log') # Often needed for physics loss too
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

# evaluate_model remains the same (evaluates MSE on control inputs)
# Physics loss is primarily a training regularizer.
def evaluate_model(model, test_loader, device, input_scaler):
    """Valuta il modello sul test set (MSE on control inputs) e visualizza."""
    model.eval()
    test_mse_loss_accum = 0
    predictions_list = []
    targets_list = []

    print("\nEvaluating control input prediction on test set...")
    with torch.no_grad():
        # Unpack 3 items, use first 2 (model_input=state, target=control_input)
        for model_inputs, targets, _ in test_loader: # Ignore physics_info here
            model_inputs = model_inputs.to(device) # Scaled states
            targets = targets.to(device)           # Scaled control inputs

            predictions = model(model_inputs) # Predicted scaled control inputs

            predictions_list.append(predictions.cpu())
            targets_list.append(targets.cpu())

            # Calculate MSE between predicted scaled inputs and target scaled inputs
            test_mse_loss_accum += torch.nn.functional.mse_loss(predictions, targets).item()

    all_predictions_scaled = torch.cat(predictions_list)
    all_targets_scaled = torch.cat(targets_list)

    avg_test_mse = test_mse_loss_accum / len(test_loader) if len(test_loader) > 0 else 0
    print(f"Test MSE (on scaled control inputs): {avg_test_mse:.6f}")

    if input_scaler:
        try:
            all_predictions_unscaled = input_scaler.inverse_transform(all_predictions_scaled.numpy())
            all_targets_unscaled = input_scaler.inverse_transform(all_targets_scaled.numpy())

            num_points_to_plot = min(200, len(all_targets_unscaled))
            plt.figure(figsize=(14, 7))
            plt.subplot(1, 2, 1)
            plt.plot(all_targets_unscaled[:num_points_to_plot, 0], label='True Control[0] (Thrust)', linestyle='--')
            plt.plot(all_predictions_unscaled[:num_points_to_plot, 0], label='Predicted Control[0]', alpha=0.8)
            plt.title('Example: Thrust (Test Set)')
            plt.xlabel('Time Step (in test sequence)')
            plt.ylabel('Control Value (unscaled)')
            plt.legend(); plt.grid(True)
            plt.subplot(1, 2, 2)
            plt.plot(all_targets_unscaled[:num_points_to_plot, 1], label='True Control[1] (Tau_x)', linestyle='--')
            plt.plot(all_predictions_unscaled[:num_points_to_plot, 1], label='Predicted Control[1]', alpha=0.8)
            plt.title('Example: Torque Tau_x (Test Set)')
            plt.xlabel('Time Step (in test sequence)'); plt.ylabel('Control Value (unscaled)')
            plt.legend(); plt.grid(True)
            plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"Could not plot unscaled control data: {e}")
            # Fallback plot omitted for brevity

    return avg_test_mse


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)
    print(f"Using device: {device}")

    state_csv_for_model_input = "state_results_validation.csv"
    input_csv_for_model_target = "input_results_validation.csv"
    print(f"Looking for data files:\n State (Input): {state_csv_for_model_input}\n Control (Target): {input_csv_for_model_target}")

    print("\nLoading dataset...")
    try:
        dataset = QuadrotorDataset(state_csv_path=state_csv_for_model_input,
                                   input_csv_path=input_csv_for_model_target)
        # Create dataloaders with shuffle=False, drop_last=True for train/val
        train_loader, val_loader, test_loader, state_scaler, input_scaler = create_dataloaders(
            dataset, batch_size=BATCH_SIZE
        )
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return

    print("\nInitializing Model (State -> Control Input)...")
    model = QuadrotorPINN(input_dim=12, output_dim=4).to(device)

    # --- MODIFIED: Instantiate the new PhysicsInformedLoss_Control ---
    print(f"\nInitializing Physics Informed Loss (Control) with lambda={PHYSICS_LOSS_WEIGHT}...")
    try:
        criterion = PhysicsInformedLoss_Control(
            input_scaler=input_scaler, # Pass the scaler for predicted controls
            lambda_physics=PHYSICS_LOSS_WEIGHT
        )
    except ValueError as e:
        print(f"ERROR initializing loss function: {e}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Scheduler (adjust if needed)
    total_steps = NUM_EPOCHS * len(train_loader) if train_loader and len(train_loader) > 0 else NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=total_steps, pct_start=0.3
    ) if total_steps > 0 else None

    print("\nStarting training...")
    history = train_pinn(
        model=model,
        criterion=criterion, # Pass the physics-informed criterion
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS
    )

    print("\nVisualizing training history...")
    visualize_training_history(history) # Updated visualization

    print("\nEvaluating model on test set (Control Input MSE)...")
    if test_loader:
        test_mse = evaluate_model(model, test_loader, device, input_scaler)
    else:
        print("Test loader unavailable, skipping evaluation.")
        test_mse = float('nan')

    print("\nSaving model...")
    try:
        save_path = 'trained_control_predictor_pinn.pth' # New name
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'test_mse_scaled': test_mse,
            'input_dim': 12,
            'output_dim': 4,
            'state_scaler_mean': state_scaler.mean_,
            'state_scaler_scale': state_scaler.scale_,
            'input_scaler_mean': input_scaler.mean_,
            'input_scaler_scale': input_scaler.scale_,
            'physics_loss_weight': PHYSICS_LOSS_WEIGHT, # Save lambda
        }, save_path)
        print(f"Model saved to '{save_path}'")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()