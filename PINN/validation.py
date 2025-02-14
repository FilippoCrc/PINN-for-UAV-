import torch
import matplotlib.pyplot as plt
import numpy as np

def validate_hover_condition(model):
    """
    Tests if the model predicts appropriate controls for hover condition
    (zero velocities, zero angles, zero angular rates)
    """
    # Create hover state
    hover_state = torch.zeros(1, 8)
    
    # Get model predictions
    with torch.no_grad():
        T, tau = model(hover_state)
        derivatives = model.dynamics.get_state_derivative(hover_state, T, tau)
    
    print("\nHover Condition Test:")
    print(f"Thrust: {T.item():.3f} N (should be close to {model.dynamics.m * model.dynamics.g:.3f} N)")
    print(f"Torques: {tau[0].numpy()} Nm (should be close to zero)")
    print(f"Derivatives: {derivatives[0].numpy()} (should be close to zero except v_dot_z)")
    
    # Check if predictions make physical sense
    hover_thrust_error = abs(T.item() - model.dynamics.m * model.dynamics.g)
    torques_magnitude = torch.norm(tau)
    
    return hover_thrust_error < 0.1 and torques_magnitude < 0.01

def validate_angle_response(model):
    """
    Tests if the model responds correctly to non-zero angles
    """
    # Create states with small angles
    test_states = torch.zeros(4, 8)
    test_states[0, 6] = 0.1    # Small positive roll
    test_states[1, 6] = -0.1   # Small negative roll
    test_states[2, 7] = 0.1    # Small positive pitch
    test_states[3, 7] = -0.1   # Small negative pitch
    
    # Get model predictions
    with torch.no_grad():
        thrusts = []
        taus = []
        derivatives = []
        for state in test_states:
            T, tau = model(state.unsqueeze(0))
            deriv = model.dynamics.get_state_derivative(state.unsqueeze(0), T, tau)
            thrusts.append(T.item())
            taus.append(tau[0].numpy())
            derivatives.append(deriv[0].numpy())
    
    print("\nAngle Response Test:")
    cases = ["Positive Roll", "Negative Roll", "Positive Pitch", "Negative Pitch"]
    for i, case in enumerate(cases):
        print(f"\n{case}:")
        print(f"Thrust: {thrusts[i]:.3f} N")
        print(f"Torques: {taus[i]}")
        print(f"Derivatives: {derivatives[i]}")
    
    return derivatives

def plot_trajectories(model, initial_states, title):
    """
    Creates an educational plot showing quadrotor response to different initial conditions
    with detailed explanations of what we're seeing.
    """
    dt = 0.01
    t_end = 2.0
    t = np.arange(0, t_end, dt)
    trajectories = []
    
    # Simulate trajectories
    for initial_state in initial_states:
        state = initial_state.clone()
        states = [state.numpy()]
        
        with torch.no_grad():
            for _ in range(len(t)-1):
                T, tau = model(state.unsqueeze(0))
                derivative = model.dynamics.get_state_derivative(state.unsqueeze(0), T, tau)
                
                full_derivative = torch.zeros_like(state)
                full_derivative[:6] = derivative[0]
                full_derivative[6] = state[3]  # φ̇ = p
                full_derivative[7] = state[4]  # θ̇ = q
                
                state = state + full_derivative * dt
                states.append(state.numpy())
        
        trajectories.append(np.array(states))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 3)
    
    # Main title with explanation
    fig.suptitle(f'{title}\n\n' + 
                'Comparing three scenarios:\n' +
                'Blue: Initial roll of 0.1 rad (5.7°) - Testing side tilt response\n' +
                'Orange: Initial pitch of 0.1 rad (5.7°) - Testing forward tilt response\n' +
                'Green: Initial forward velocity of 0.5 m/s - Testing velocity response\n',
                fontsize=12, y=0.99)
    
    # State variables and their explanations
    variables = [
        ('vx', 'Forward Velocity (m/s)', 
         'Forward velocity changes most with pitch angle'),
        ('vy', 'Sideward Velocity (m/s)', 
         'Sideward velocity changes most with roll angle'),
        ('vz', 'Vertical Velocity (m/s)',
         'Should stay near zero for stable hover'),
        ('p', 'Roll Rate (rad/s)',
         'Angular velocity around x-axis'),
        ('q', 'Pitch Rate (rad/s)',
         'Angular velocity around y-axis'),
        ('r', 'Yaw Rate (rad/s)',
         'Angular velocity around z-axis'),
        ('φ', 'Roll Angle (rad)',
         'Rotation around x-axis'),
        ('θ', 'Pitch Angle (rad)',
         'Rotation around y-axis')
    ]
    
    # Create subplots
    axes = []
    for i, (var, label, explanation) in enumerate(variables):
        if i < 6:  # First two rows
            ax = fig.add_subplot(gs[i//3, i%3])
        else:      # Last row, centered
            ax = fig.add_subplot(gs[2, i-6])
        axes.append(ax)
        
        # Plot all trajectories
        labels = ['Initial Roll', 'Initial Pitch', 'Initial Velocity']
        for j, traj in enumerate(trajectories):
            ax.plot(t, traj[:, i], label=labels[j])
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.set_title(f'{label}\n{explanation}', fontsize=10)
        ax.grid(True)
        
        # Add legend only to first plot
        if i == 0:
            ax.legend(loc='upper right')
    
    # Add a text box with physical interpretation
    interpretation = (
        "Physical Interpretation:\n\n"
        "1. For initial roll (blue):\n"
        "   - Expect significant vy (sideward velocity)\n"
        "   - Roll angle should stabilize back to zero\n\n"
        "2. For initial pitch (orange):\n"
        "   - Expect significant vx (forward velocity)\n"
        "   - Pitch angle should stabilize back to zero\n\n"
        "3. For initial velocity (green):\n"
        "   - Expect angles to change to slow down\n"
        "   - Velocity should gradually decrease"
    )
    
    plt.figtext(0.02, 0.02, interpretation, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def test_specific_conditions(model):
    """
    Tests the model's response to specific initial conditions
    """
    print("\nTesting specific conditions:")
    
    # Test cases
    conditions = [
        ("Hover with small roll", torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])),
        ("Hover with small pitch", torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])),
        ("Initial forward velocity", torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ]
    
    for name, state in conditions:
        print(f"\n{name}:")
        with torch.no_grad():
            T, tau = model(state.unsqueeze(0))
            derivative = model.dynamics.get_state_derivative(state.unsqueeze(0), T, tau)
            
            print(f"Thrust: {T.item():.3f} N")
            print(f"Torques: {tau[0].numpy()}")
            print(f"Derivatives: {derivative[0].numpy()}")

def comprehensive_validation(model):
    """
    Runs all validation tests
    """
    # Test 1: Hover condition
    hover_valid = validate_hover_condition(model)
    print(f"\nHover test passed: {hover_valid}")
    
    # Test 2: Angle response
    validate_angle_response(model)
    
    # Test 3: Specific conditions
    test_specific_conditions(model)
    
    # Test 4: Trajectory simulation
    initial_states = [
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),  # Small roll
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),  # Small pitch
        torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Initial velocity
    ]
    
    plot_trajectories(model, initial_states, "System Response to Different Initial Conditions")

