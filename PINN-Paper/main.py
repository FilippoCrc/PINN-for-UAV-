import torch
import matplotlib.pyplot as plt
from dataset import MidAirDataset, create_dataLoaders
from pinn import QuadrotorPINN, PhysicsInformedLoss
from trainer import train_pinn
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from controller import StateController
import os

def visualize_training_history(history):
    """Visualizes the training progress including both MSE and physics-informed losses."""
    plt.figure(figsize=(15, 5))
    
    # Plot total loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MSE component
    plt.subplot(1, 3, 2)
    plt.plot(history['train_mse'], label='Train')
    plt.plot(history['val_mse'], label='Validation')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    # Plot physics component
    plt.subplot(1, 3, 3)
    plt.plot(history['train_physics'], label='Train')
    plt.plot(history['val_physics'], label='Validation')
    plt.title('Physics Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Physics Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_covariance_confidence_ellipse(predictions, states, title):
    """Creates CCE visualization as described in the paper."""
    # Extract angular accelerations (omega_dot)
    angular_accels = states[:, 3:6].cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes_labels = ['Roll', 'Pitch', 'Yaw']
    
    for i in range(3):
        # Compute 2D mean and covariance
        data = np.column_stack([angular_accels[:, i], predictions[:, i]])
        mean = np.mean(data, axis=0)
        cov = np.cov(data.T)
        
        # Create ellipse points
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        # Plot data points and ellipse
        axes[i].scatter(data[:, 0], data[:, 1], alpha=0.5)
        axes[i].add_patch(plt.matplotlib.patches.Ellipse(
            mean, 3*np.sqrt(eigenvals[0]), 3*np.sqrt(eigenvals[1]),
            angle=angle, fill=False, color='red'
        ))
        axes[i].set_title(f'{axes_labels[i]} Motion')
        axes[i].set_xlabel('Angular Acceleration')
        axes[i].set_ylabel('PWM Signal')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device):
    """Evaluates the model on test data and visualizes results."""
    model.eval()
    test_loss = 0
    predictions_list = []
    states_list = []
    targets_list = []
    
    with torch.no_grad():
        for states, targets in test_loader:
            states, targets = states.to(device), targets.to(device)
            predictions = model(states)
            
            predictions_list.append(predictions)
            states_list.append(states)
            targets_list.append(targets)
            
            test_loss += torch.nn.functional.mse_loss(predictions, targets).item()
    
    all_predictions = torch.cat(predictions_list)
    all_states = torch.cat(states_list)
    all_targets = torch.cat(targets_list)
    
    print(f"\nTest MSE: {test_loss/len(test_loader):.6f}")
    
    plot_covariance_confidence_ellipse(
        all_predictions, all_states, 
        "Physical Consistency Visualization (Test Data)"
    )
    
    return test_loss/len(test_loader)

def plot_test_results(time_steps, states, pwm_signals, scenario_name):
    """
    Visualizza i risultati del test per ogni scenario di volo
    
    Args:
        time_steps: array dei passi temporali
        states: array degli stati del sistema
        pwm_signals: array dei segnali PWM
        scenario_name: nome dello scenario testato
    """
    plt.figure(figsize=(15, 10))
    
    # Plot degli stati
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, states)
    plt.title(f'Stati durante {scenario_name}')
    plt.xlabel('Time Step')
    plt.ylabel('State Values')
    plt.legend([f'State {i}' for i in range(states.shape[1])])
    
    # Plot dei segnali PWM
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, pwm_signals)
    plt.title('Segnali PWM')
    plt.xlabel('Time Step')
    plt.ylabel('PWM Values')
    plt.legend(['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4'])
    
    plt.tight_layout()
    plt.show()

def test_controller_extended(controller):
    """
    Test esteso del controller con diversi scenari di volo complessi
    """
    # Definiamo diversi scenari di test realistici
    test_scenarios = [
        {
            "name": "Hover stabile",
            "states": [(8, 0.1)],  # Mantieni quota
            "duration": 50
        },
        {
            "name": "Movimento diagonale",
            "states": [(6, 0.1), (7, 0.1)],  # x e y simultaneamente
            "duration": 100
        },
        {
            "name": "Rotazione con movimento",
            "states": [(6, 0.1), (12, 0.05)],  # Avanti con roll
            "duration": 150
        },
        {
            "name": "Manovra complessa",
            "states": [(6, 0.1), (7, 0.1), (8, -0.05), (12, 0.02)],
            "duration": 200
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nTest scenario: {scenario['name']}")
        
        # Inizializziamo gli array per salvare i dati
        current_state = torch.zeros(16)
        states_history = []
        pwm_history = []
        time_steps = []
        
        # Simulazione per la durata specificata
        for t in range(scenario['duration']):
            desired_state = torch.zeros(16)
            for state_idx, state_val in scenario['states']:
                desired_state[state_idx] = state_val
            
            # Aggiorniamo il controller
            result = controller.update(current_state, desired_state, dt=0.01)
            
            # Salviamo i dati per la visualizzazione
            states_history.append(current_state.numpy())
            pwm_history.append(result['pwm_signals'])
            time_steps.append(t)
            
            # Stampiamo i risultati ogni 10 step
            if t % 10 == 0:
                print(f"\nTimestep {t}:")
                print(f"PWM signals: {result['pwm_signals']}")
                print(f"Error: {result['error']}")
            
            # Aggiorniamo lo stato corrente (simulazione semplificata)
            current_state = current_state + (desired_state - current_state) * 0.01
        
        # Convertiamo le liste in array numpy per il plotting
        states_history = np.array(states_history)
        pwm_history = np.array(pwm_history)
        
        # Visualizziamo i risultati
        plot_test_results(time_steps, states_history, pwm_history, scenario['name'])
        
        print(f"\nScenario {scenario['name']} completato")
        print("Stato finale:")
        print(current_state)

def main():
    # Set device and random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    print(f"Using device: {device}")
    
    # Get the correct path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "sensor_records.hdf5")
    
    # Verify if file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Current directory:", current_dir)
        print("Files in directory:", os.listdir(current_dir))
        return

    # Create dataset and dataloaders
    print("\nLoading dataset...")
    try:
        dataset = MidAirDataset(data_path)
        train_loader, val_loader, test_loader = create_dataLoaders(
            dataset, batch_size=64
        )
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Current directory contains: {os.listdir(current_dir)}")
        return
    
    # Initialize model
    print("\nInitializing PINN...")
    model = QuadrotorPINN().to(device)
    
    # Train model
    print("\nStarting training...")
    history = train_pinn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=300,
        learning_rate=1e-3
    )
    
    # Visualize training progress
    print("\nVisualizing training history...")
    visualize_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss = evaluate_model(model, test_loader, device)
    
    # Test del controller con scenari estesi
    print("\nTesting state controller...")
    controller = StateController(pinn_model=model)
    
    print("\nRunning extended controller tests...")
    test_controller_extended(controller)

    # Save model
    print("\nSaving model...")
    try:
        save_path = os.path.join(current_dir, 'trained_pinn.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'test_loss': test_loss
        }, save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()