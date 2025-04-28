# csv_main.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from csv_dataset import QuadrotorDataset, create_dataloaders
# --- MODIFIED: Import the new loss class ---
from pinn import QuadrotorPINN, LocalMonotonicityLoss
from trainer import train_pinn
import numpy as np
import os
from scipy.stats import pearsonr

# --- ADJUST THESE PARAMETERS ---
NUM_EPOCHS = 2000 # Maybe increase epochs if using annealing
LEARNING_RATE = 5e-4 # Might need tuning
BATCH_SIZE = 128 # Ensure >= 2

# --- Physics Loss Parameters (Tune these based on paper/experiments) ---
USE_ANNEALING = True      # Set to False to use fixed lambda_max
ANNEALING_CYCLES = 10       # M in paper (Number of cycles for lambda annealing)
ANNEALING_RATIO = 0.5    # R in paper (Proportion of cycle at max lambda)
LAMBDA_MAX = 1          # Max physics weight (lambda_max in paper)

# --- MODIFIED: visualize_training_history (Added lambda plot) ---
def visualize_training_history(history, loss_criterion=None):
    """Visualizza lo storico del training (MSE + Physics Loss + Lambda)."""
    epochs = range(1, len(history['train_loss']) + 1)
    has_validation = 'val_loss' in history and history['val_loss']

    plt.figure(figsize=(18, 12)) # Adjusted layout

    # Loss Totale
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Total Loss')
    if has_validation: plt.plot(epochs, history['val_loss'], label='Validation Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)'); plt.yscale('log')
    plt.legend(); plt.grid(True)

    # MSE Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_mse'], label='Train MSE Loss')
    if has_validation: plt.plot(epochs, history['val_mse'], label='Validation MSE Loss')
    plt.title('MSE Loss Component (Control Input)')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss (log scale)'); plt.yscale('log')
    plt.legend(); plt.grid(True)

    # Physics Loss
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['train_physics'], label='Train Physics Loss')
    if has_validation: plt.plot(epochs, history['val_physics'], label='Validation Physics Loss')
    plt.title('Physics Loss Component (Monotonicity)')
    plt.xlabel('Epoch'); plt.ylabel('Physics Loss (log scale)'); plt.yscale('log')
    plt.legend(); plt.grid(True)

    # Learning Rate
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['lr'], label='Learning Rate')
    plt.title('Learning Rate Schedule'); plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
    plt.legend(); plt.grid(True)

    # Lambda Physics Weight (Annealing Schedule)
    plt.subplot(2, 3, 5)
    if loss_criterion and hasattr(loss_criterion, '_calculate_annealing_lambda'):
        lambdas = [loss_criterion._calculate_annealing_lambda(e-1) for e in epochs]
        plt.plot(epochs, lambdas, label='Lambda Physics')
        plt.title('Physics Loss Weight (λ)')
    else:
        plt.title('Physics Loss Weight (λ) - N/A')
    plt.xlabel('Epoch'); plt.ylabel('Lambda Value')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

def validate_physical_correlation(dataset):
    """Plot torque (τ) vs angular acceleration (Δω) and compute Pearson correlation."""
    # Extract unscaled data from the dataset
    omega_unscaled = dataset.omega_unscaled.numpy()  # Shape (N, 3)
    times_unscaled = dataset.times_unscaled.numpy()  # Shape (N,)
    controls_unscaled = dataset.model_targets_unscaled.numpy()  # Shape (N, 4)

    # --- Compute angular acceleration (Δω) ---
    delta_t = times_unscaled[1:] - times_unscaled[:-1]  # Shape (N-1,)
    delta_omega = (omega_unscaled[1:] - omega_unscaled[:-1]) / delta_t.reshape(-1, 1)  # Shape (N-1, 3)

    # --- Extract torque values (τ_x, τ_y, τ_z) ---
    # Assuming columns 1-3 are torques (skip thrust in column 0)
    torques = controls_unscaled[:, 1:4]  # Shape (N, 3)
    # Align with delta_omega (exclude last torque sample)
    torques_aligned = torques[:-1, :]  # Shape (N-1, 3)

    # Plot for each axis (x, y, z)
    axes = ['x', 'y', 'z']
    for i in range(3):
        plt.figure(figsize=(8, 6))
        plt.scatter(delta_omega[:, i], torques_aligned[:, i], alpha=0.5, label='Data points')
        
        # Compute Pearson correlation
        pcc, p_value = pearsonr(delta_omega[:, i], torques_aligned[:, i])
        plt.title(f"Angular Acceleration (Δω_{axes[i]}) vs Torque (τ_{axes[i]})\nPCC: {pcc:.2f}, p-value: {p_value:.2e}")
        plt.xlabel(f"Δω_{axes[i]} [rad/s²]")
        plt.ylabel(f"τ_{axes[i]} [Nm]")
        plt.legend()
        plt.grid(True)
        plt.show()
# evaluate_model remains the same
def evaluate_model(model, test_loader, device, input_scaler):
    """Valuta il modello sul test set (MSE on control inputs) e visualizza."""
    model.eval()
    test_mse_loss_accum = 0
    predictions_list = []
    targets_list = []

    print("\nEvaluating control input prediction on test set...")
    with torch.no_grad():
        # Loop remains the same
        for model_inputs, targets, _ in test_loader: # Ignore physics_info
            model_inputs = model_inputs.to(device)
            targets = targets.to(device)

            predictions = model(model_inputs)

            predictions_list.append(predictions.cpu())
            targets_list.append(targets.cpu())

            # Ensure targets are float for mse_loss if they aren't
            test_mse_loss_accum += torch.nn.functional.mse_loss(predictions, targets.float()).item()

    all_predictions_scaled = torch.cat(predictions_list)
    all_targets_scaled = torch.cat(targets_list)

    avg_test_mse = test_mse_loss_accum / len(test_loader) if len(test_loader) > 0 else 0
    print(f"Test MSE (on scaled control inputs): {avg_test_mse:.6f}")

    if input_scaler:
        try:
            # Ensure data is numpy for scaler
            all_predictions_unscaled = input_scaler.inverse_transform(all_predictions_scaled.numpy())
            all_targets_unscaled = input_scaler.inverse_transform(all_targets_scaled.numpy())

            num_points_to_plot = min(200, len(all_targets_unscaled)) # Limit points for clarity

            # --- MODIFIED PLOTTING SECTION ---
            plt.figure(figsize=(14, 10)) # Adjusted figure size for 2x2 grid
            control_names = ['Thrust', 'Tau_x', 'Tau_y', 'Tau_z'] # Assuming this order
            num_outputs = all_targets_unscaled.shape[1] # Should be 4

            if num_outputs != 4:
                print(f"Warning: Expected 4 control outputs for plotting, but found {num_outputs}. Adjusting plot.")
                control_names = [f'Control[{i}]' for i in range(num_outputs)] # Generic names

            plot_rows = int(np.ceil(num_outputs / 2.0))
            plot_cols = 2

            for i in range(num_outputs):
                plt.subplot(plot_rows, plot_cols, i + 1) # Create subplot (1-based index)
                plt.plot(all_targets_unscaled[:num_points_to_plot, i], label=f'True {control_names[i]}', linestyle='--')
                plt.plot(all_predictions_unscaled[:num_points_to_plot, i], label=f'Predicted {control_names[i]}', alpha=0.8)
                plt.title(f'Example: {control_names[i]} (Output Index {i}) (Test Set)')
                plt.xlabel('Time Step (Sample Index)')
                plt.ylabel('Control Value (unscaled)')
                plt.legend()
                plt.grid(True)

            plt.tight_layout()
            plt.show()
            # --- END OF MODIFIED PLOTTING SECTION ---

        except AttributeError as e:
             print(f"Plotting Error: Input scaler might not have 'inverse_transform' or data format issue. {e}")
             print("Plotting scaled data instead as fallback.")
             # Fallback to plotting scaled data if unscaling fails
             num_points_to_plot = min(200, len(all_targets_scaled))
             plt.figure(figsize=(14, 10))
             control_names = [f'Scaled Control[{i}]' for i in range(all_targets_scaled.shape[1])]
             num_outputs = all_targets_scaled.shape[1]
             plot_rows = int(np.ceil(num_outputs / 2.0))
             plot_cols = 2
             for i in range(num_outputs):
                 plt.subplot(plot_rows, plot_cols, i + 1)
                 plt.plot(all_targets_scaled[:num_points_to_plot, i].numpy(), label=f'True {control_names[i]}', linestyle='--')
                 plt.plot(all_predictions_scaled[:num_points_to_plot, i].numpy(), label=f'Predicted {control_names[i]}', alpha=0.8)
                 plt.title(f'Example: {control_names[i]} (Output Index {i}) (Test Set - Scaled)')
                 plt.xlabel('Time Step (Sample Index)')
                 plt.ylabel('Control Value (scaled)')
                 plt.legend(); plt.grid(True)
             plt.tight_layout(); plt.show()

        except Exception as e:
            print(f"An unexpected error occurred during plotting: {e}")


    return avg_test_mse

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)
    print(f"Using device: {device}")

    state_csv_for_model_input = "state_results_long.csv"
    input_csv_for_model_target = "input_results_long.csv"
    print(f"Looking for data files:\n State (Input): {state_csv_for_model_input}\n Control (Target): {input_csv_for_model_target}")

    print("\nLoading dataset...")
    try:
        dataset = QuadrotorDataset(state_csv_path=state_csv_for_model_input,
                                   input_csv_path=input_csv_for_model_target)
        print("\nValidating physical correlation...")
        validate_physical_correlation(dataset)
        # MUST use shuffle=False, drop_last=True for train/val for physics loss
        train_loader, val_loader, test_loader, state_scaler, input_scaler = create_dataloaders(
            dataset, batch_size=BATCH_SIZE, shuffle_train_val=False, drop_last_train_val=True
        )
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nInitializing Model (State -> Control Input)...")
    # Ensure input_dim matches your state_results.csv columns (excluding time)
    model = QuadrotorPINN(input_dim=12, output_dim=4).to(device)

    # --- MODIFIED: Instantiate the new LocalMonotonicityLoss ---
    print(f"\nInitializing Local Monotonicity Loss...")
    try:
        criterion = LocalMonotonicityLoss(
            input_scaler=input_scaler, # Pass the scaler for predicted controls
            use_annealing=USE_ANNEALING,
            annealing_cycles=ANNEALING_CYCLES,
            annealing_ratio=ANNEALING_RATIO,
            lambda_max=LAMBDA_MAX,
            total_epochs=NUM_EPOCHS # Pass total epochs here
        )
        print(f" Loss params: Annealing={USE_ANNEALING}, Cycles={ANNEALING_CYCLES}, Ratio={ANNEALING_RATIO:.2f}, LambdaMax={LAMBDA_MAX:.3f}")
    except ValueError as e:
        print(f"ERROR initializing loss function: {e}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Scheduler (OneCycleLR often works well)
    total_steps = NUM_EPOCHS * len(train_loader) if train_loader and len(train_loader) > 0 else NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=total_steps, pct_start=0.3
    ) if total_steps > 0 else None

    print("\nStarting training...")
    history = train_pinn(
        model=model,
        criterion=criterion, # Pass the new physics-informed criterion
        optimizer=optimizer,
        scheduler=scheduler, # Pass scheduler for per-batch stepping
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS
    )

    print("\nVisualizing training history...")
    visualize_training_history(history, criterion) # Pass criterion to plot lambda

    print("\nEvaluating model on test set (Control Input MSE)...")
    if test_loader:
        test_mse = evaluate_model(model, test_loader, device, input_scaler)
    else:
        print("Test loader unavailable, skipping evaluation.")
        test_mse = float('nan')

    print("\nSaving model...")
    try:
        save_path = 'trained_control_predictor_monotonicity_pinn.pth' # New name
        # Save annealing parameters too
        save_dict = {
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
            'loss_params': { # Save loss config
                 'use_annealing': USE_ANNEALING,
                 'annealing_cycles': ANNEALING_CYCLES,
                 'annealing_ratio': ANNEALING_RATIO,
                 'lambda_max': LAMBDA_MAX,
                 'total_epochs': NUM_EPOCHS
            }
        }
        torch.save(save_dict, save_path)
        print(f"Model saved to '{save_path}'")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()