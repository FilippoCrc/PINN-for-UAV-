import torch
import numpy as np
from pinn import QuadrotorPINN

class StateController:
    """
    Controller di stato per il quadricottero usando il modello PINN
    Gestisce stati 16D nella forma:
    [v_dot(3), omega_dot(3), v(3), omega(3), phi, theta, sin(psi), cos(psi)]
    """
    def __init__(self, pinn_model_path=None, pinn_model=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Carica il modello PINN
        if pinn_model is not None:
            self.pinn = pinn_model
        else:
            self.pinn = QuadrotorPINN()
            if pinn_model_path:
                checkpoint = torch.load(pinn_model_path, map_location=device)
                self.pinn.load_state_dict(checkpoint['model_state_dict'])
        
        self.pinn.eval()
        self.device = device
        self.pinn.to(device)
        
        # Parametri di controllo per stato completo
        # Mantengo la stessa struttura dei gain originali ma con valori ottimizzati
        self.Kp = torch.tensor([
            0.5, 0.5, 0.5,     # v_dot (3)
            0.3, 0.3, 0.3,     # omega_dot (3)
            1.0, 1.0, 1.0,     # v (3)
            0.5, 0.5, 0.5,     # omega (3)
            0.2, 0.2,          # phi, theta
            0.1, 0.1           # sin(psi), cos(psi)
        ], device=device)
        
        self.Kd = self.Kp * 0.2  # Ridotto da 0.5
        self.Ki = self.Kp * 0.05  # Ridotto da 0.1
        
        # Inizializzazione errori sul device corretto
        self.integral_error = torch.zeros(16, device=device)
        self.prev_error = torch.zeros(16, device=device)

    def update(self, current_state, desired_state, dt):
        """
        Aggiorna il controllore e calcola i segnali PWM
        
        Args:
            current_state: Tensore 16D [v_dot(3), omega_dot(3), v(3), omega(3), phi, theta, sin(psi), cos(psi)]
            desired_state: Tensore dello stesso formato
            dt: Intervallo di tempo
        """
        # Converti gli stati in tensori se non lo sono già
        if not isinstance(current_state, torch.Tensor):
            current_state = torch.tensor(current_state, dtype=torch.float32)
        if not isinstance(desired_state, torch.Tensor):
            desired_state = torch.tensor(desired_state, dtype=torch.float32)
        
        # Sposta sul device corretto
        current_state = current_state.to(self.device)
        desired_state = desired_state.to(self.device)
        
        # Calcola errore
        error = desired_state - current_state
        
        # Aggiorna l'errore integrale con clipping per evitare accumulo eccessivo
        self.integral_error = torch.clamp(
            self.integral_error + error * dt,
            -1.0, 1.0
        )
        
        # Calcola il termine derivativo con protezione per dt=0
        derivative = torch.clamp(
            (error - self.prev_error) / max(dt, 1e-6),
            -1.0, 1.0
        )
        self.prev_error = error.clone()
        
        # Calcola il controllo PID con clipping
        control = torch.clamp(
            self.Kp * error +
            self.Ki * self.integral_error +
            self.Kd * derivative,
            -1.0, 1.0
        )
        
        # Applica il controllo allo stato corrente
        controlled_state = current_state + control
        
        # Normalizzazione di sin(psi) e cos(psi)
        sin_psi = controlled_state[14]
        cos_psi = controlled_state[15]
        norm = torch.sqrt(sin_psi**2 + cos_psi**2 + 1e-6)  # Aggiunto epsilon
        controlled_state[14] = sin_psi / norm
        controlled_state[15] = cos_psi / norm
        
        # Debug prints
        print("Controlled state shape:", controlled_state.shape)
        print("Controlled state device:", controlled_state.device)
        
        # Usa il PINN per predire i segnali PWM
        with torch.no_grad():
            # Aggiunge dimensione del batch
            controlled_state = controlled_state.unsqueeze(0)
            pwm_signals = self.pinn(controlled_state)
            
            # Debug prints
            print("PWM signals before clamp:", pwm_signals)
            
            # Clip dei valori PWM tra 0 e 1
            pwm_signals = torch.clamp(pwm_signals, 0, 1)
            
            # Gestione NaN
            if torch.isnan(pwm_signals).any():
                print("WARNING: NaN detected in PWM signals")
                pwm_signals = torch.nan_to_num(pwm_signals, 0.5)
            
            print("Final PWM signals:", pwm_signals)
            
        return {
            'pwm_signals': pwm_signals.squeeze(0).cpu().numpy(),
            'control': control.cpu().numpy(),
            'error': error.cpu().numpy()
        }

    def reset(self):
        """Resetta lo stato del controllore"""
        self.integral_error = torch.zeros(16, device=self.device)
        self.prev_error = torch.zeros(16, device=self.device)