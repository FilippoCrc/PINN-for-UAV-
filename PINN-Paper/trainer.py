import torch
import torch.nn as nn
import torch.nn.functional as F
from pinn import QuadrotorPINN # Assicurati che pinn.py sia nello stesso path o installato
from pinn import PhysicsInformedLoss # Assicurati che pinn.py sia nello stesso path o installato

def train_pinn(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs):
    """
    Trains the Physics-Informed Neural Network.
    Ora accetta criterion, optimizer, scheduler come argomenti.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) # Assicurati che il modello sia sul device giusto

    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': [], 'train_physics': [], 'val_physics': [], 'lr': []}

    print(f"Starting training for {num_epochs} epochs on {device}...")
    # ... (altre stampe info) ...

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0.0
        train_mse_accum = 0.0
        train_physics_accum = 0.0

        # --- MODIFICA: Unpack 3 items dal loader ---
        for i, (inputs, states, physics_info) in enumerate(train_loader):
            # --- MODIFICA: Sposta tutti i tensori sul device ---
            inputs = inputs.to(device)
            states = states.to(device) # Questi sono i target scalati per MSE
            physics_info = physics_info.to(device) # Info non scalata per physics loss

            optimizer.zero_grad()
            pred_states = model(inputs) # pred_states sono scalati

            # --- MODIFICA: Chiama criterion con la nuova firma ---
            loss, mse_loss_item, physics_loss_item = criterion(pred_states, states, physics_info, epoch)

            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at epoch {epoch+1}, batch {i+1}. Skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step() # Aggiorna il learning rate

            train_loss_accum += loss.item()
            train_mse_accum += mse_loss_item
            train_physics_accum += physics_loss_item

        # Validation phase
        model.eval()
        val_loss_accum = 0.0
        val_mse_accum = 0.0
        val_physics_accum = 0.0

        with torch.no_grad():
             # --- MODIFICA: Unpack 3 items dal loader ---
            for inputs, states, physics_info in val_loader:
                 # --- MODIFICA: Sposta tutti i tensori sul device ---
                inputs = inputs.to(device)
                states = states.to(device)
                physics_info = physics_info.to(device)

                pred_states = model(inputs)

                # --- MODIFICA: Chiama criterion con la nuova firma ---
                loss, mse_loss_item, physics_loss_item = criterion(pred_states, states, physics_info, epoch)

                if not torch.isnan(loss):
                    val_loss_accum += loss.item()
                    val_mse_accum += mse_loss_item
                    val_physics_accum += physics_loss_item

        # Record metrics
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss_accum / len(train_loader))
        history['val_loss'].append(val_loss_accum / len(val_loader))
        history['train_mse'].append(train_mse_accum / len(train_loader))
        history['val_mse'].append(val_mse_accum / len(val_loader))
        history['train_physics'].append(train_physics_accum / len(train_loader))
        history['val_physics'].append(val_physics_accum / len(val_loader))
        history['lr'].append(current_lr)


        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.6f} (MSE: {history['train_mse'][-1]:.6f}, Phys: {history['train_physics'][-1]:.6f}) | "
              f"Val Loss: {history['val_loss'][-1]:.6f} (MSE: {history['val_mse'][-1]:.6f}, Phys: {history['val_physics'][-1]:.6f}) | "
              f"LR: {current_lr:.6f}")

    return history