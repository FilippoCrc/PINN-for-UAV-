import torch
import torch.nn as nn
import torch.nn.functional as F
# from pinn import PhysicsInformedLoss_Control # Loss passed as argument

def train_pinn(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs):
    """
    Trains the NN predicting control inputs, using the provided criterion
    (which now includes the physics loss component).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # --- MODIFIED: History includes physics loss again ---
    history = {'train_loss': [], 'val_loss': [],
               'train_mse': [], 'val_mse': [],
               'train_physics': [], 'val_physics': [],
               'lr': []}

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum, train_mse_accum, train_physics_accum = 0.0, 0.0, 0.0
        batches_processed_train = 0

        # --- Unpack 3 items: model_input (state), target (control), physics_info ---
        for i, (model_inputs, targets, physics_info) in enumerate(train_loader):
            # Check if batch size is sufficient for physics loss calculation
            if model_inputs.shape[0] < 2:
                 print(f"Warning: Skipping training batch {i+1} due to insufficient size ({model_inputs.shape[0]} < 2) for physics loss.")
                 continue

            model_inputs = model_inputs.to(device) # Scaled states
            targets = targets.to(device)           # Scaled control inputs
            physics_info = physics_info.to(device) # Unscaled t, omega

            optimizer.zero_grad()
            predictions = model(model_inputs) # Predict scaled control inputs

            # --- MODIFIED: Call criterion with physics_info ---
            loss, mse_loss_item, physics_loss_item = criterion(
                predictions, targets, physics_info, epoch
            )

            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at epoch {epoch+1}, batch {i+1}. Skipping batch.")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                 scheduler.step() # Step scheduler (e.g., OneCycleLR per batch)

            train_loss_accum += loss.item()
            train_mse_accum += mse_loss_item
            train_physics_accum += physics_loss_item
            batches_processed_train += 1

        # Validation phase
        model.eval()
        val_loss_accum, val_mse_accum, val_physics_accum = 0.0, 0.0, 0.0
        batches_processed_val = 0

        # Only calculate validation physics loss if val_loader exists and drop_last=True was used
        # (ensuring batch size >= 2)
        can_validate_physics = val_loader is not None and val_loader.drop_last

        if val_loader:
            with torch.no_grad():
                for model_inputs, targets, physics_info in val_loader:
                    # Check batch size *again* for validation, especially if drop_last=False was forced
                    is_valid_batch_for_physics = model_inputs.shape[0] >= 2 and can_validate_physics

                    model_inputs = model_inputs.to(device)
                    targets = targets.to(device)
                    physics_info = physics_info.to(device)

                    predictions = model(model_inputs)

                    # Calculate loss components
                    if is_valid_batch_for_physics:
                        loss, mse_loss_item, physics_loss_item = criterion(
                            predictions, targets, physics_info, epoch
                        )
                    else:
                        # Calculate only MSE if physics loss cannot be computed
                        loss = F.mse_loss(predictions, targets) # Use standard MSE
                        mse_loss_item = loss.item()
                        physics_loss_item = 0.0 # Assign 0 if not computed
                        # Note: total val loss might not perfectly reflect train loss if physics is skipped

                    if not torch.isnan(loss):
                        val_loss_accum += loss.item()
                        val_mse_accum += mse_loss_item
                        # Only accumulate physics loss if it was actually computed
                        if is_valid_batch_for_physics:
                            val_physics_accum += physics_loss_item
                        batches_processed_val += 1

        # Record metrics
        current_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = train_loss_accum / batches_processed_train if batches_processed_train > 0 else 0
        avg_train_mse = train_mse_accum / batches_processed_train if batches_processed_train > 0 else 0
        avg_train_physics = train_physics_accum / batches_processed_train if batches_processed_train > 0 else 0

        avg_val_loss = val_loss_accum / batches_processed_val if batches_processed_val > 0 else 0
        avg_val_mse = val_mse_accum / batches_processed_val if batches_processed_val > 0 else 0
        # Average physics loss only over batches where it was computed
        num_val_batches_with_physics = sum(1 for b in val_loader if b[0].shape[0] >= 2) if can_validate_physics and val_loader else 0
        avg_val_physics = val_physics_accum / num_val_batches_with_physics if num_val_batches_with_physics > 0 else 0


        history['train_loss'].append(avg_train_loss)
        history['train_mse'].append(avg_train_mse)
        history['train_physics'].append(avg_train_physics)
        if val_loader:
             history['val_loss'].append(avg_val_loss)
             history['val_mse'].append(avg_val_mse)
             history['val_physics'].append(avg_val_physics)
        history['lr'].append(current_lr)

        # --- MODIFIED: Print statement includes physics again ---
        val_print_str = f"Val Loss: {avg_val_loss:.6f} (MSE: {avg_val_mse:.6f}, Phys: {avg_val_physics:.6f})" if val_loader else "Val: N/A"
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} (MSE: {avg_train_mse:.6f}, Phys: {avg_train_physics:.6f}) | "
              f"{val_print_str} | "
              f"LR: {current_lr:.6f}")

        # Scheduler step (if per-epoch type, e.g., ReduceLROnPlateau)
        # ... (add if using such a scheduler)

    return history