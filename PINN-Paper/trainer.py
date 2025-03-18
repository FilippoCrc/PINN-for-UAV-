import torch
import torch.nn as nn
import torch.nn.functional as F
from pinn import QuadrotorPINN
from pinn import PhysicsInformedLoss

def train_pinn(model, train_loader, val_loader, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = PhysicsInformedLoss()

    # Update optimizer and criterion to use AdamW and OneCycleLR
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3,
        total_steps=num_epochs*len(train_loader),
        pct_start=0.3
    )
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': [], 'train_physics': [], 'val_physics': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_physics = 0.0
        
        for inputs, states in train_loader:
            inputs, states = inputs.to(device), states.to(device)
            
            optimizer.zero_grad()
            pred_states = model(inputs)  # Model now predicts states
            
            # Loss calculation uses ONLY predictions and targets
            loss, mse_loss, physics_loss = criterion(pred_states, states, epoch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse_loss
            train_physics += physics_loss
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_physics = 0.0
        
        with torch.no_grad():
            for inputs, states in val_loader:
                inputs, states = inputs.to(device), states.to(device)
                pred_states = model(inputs)
                loss, mse_loss, physics_loss = criterion(pred_states, states, epoch)
                
                val_loss += loss.item()
                val_mse += mse_loss
                val_physics += physics_loss
        
        # Record metrics
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss/len(val_loader))
        history['train_mse'].append(train_mse/len(train_loader))
        history['val_mse'].append(val_mse/len(val_loader))
        history['train_physics'].append(train_physics/len(train_loader))
        history['val_physics'].append(val_physics/len(val_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {history['val_loss'][-1]:.4f}")
    
    return history