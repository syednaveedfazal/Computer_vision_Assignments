import numpy as np
import matplotlib.pyplot as plt

def run_kalman_filter(observations):
    
    dt = 0.1  # Time step
    
    # Noise parameters
    sp = 0.001  # Process noise parameter
    sm = 0.05   # Measurement noise parameter

    Psi = np.array([
        [1, 0, dt, 0, 0.5 * dt**2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    # Transition matrix that predicts the next state based on physics

    # Measurement Matrix
    Phi = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ])
    # extracts only the first two elements from the state vector

    
    Sigma_p = sp * np.eye(6)
    # Process Noise Covariance
    
    Sigma_m = sm * np.eye(2)
    # Measurement Noise Covariance

    x = np.array([-10, -150, 1, -2, 0, 0])
    # Initial State Vector

    P = np.eye(6)
    # Initial State Covariance

    estimated_positions = []
    # Storage for results

    # Kalman Filter Loop
    for z_k in observations:
        # Prediction Step
        x_pred = Psi @ x
        P_pred = Psi @ P @ Psi.T + Sigma_p

        # Correction Step
        # Check if the observation contains missing data
        if np.any(np.isnan(z_k)):
            # If measurement is missing, we trust the prediction entirely.
            # No update is performed.
            x = x_pred
            P = P_pred
        else:
            # Else, measurement is valid, perform the standard update

            
            innovation = z_k - (Phi @ x_pred)
            # Innovation (Residual), y = z_k - Phi * x_pred
            
            S = Phi @ P_pred @ Phi.T + Sigma_m
            # Innovation Covariance, S = Phi * P_pred * Phi^T + Sigma_m

            K = P_pred @ Phi.T @ np.linalg.inv(S)
            # Optimal Kalman Gain, K = P_pred * Phi^T * inv(S)

            x = x_pred + K @ innovation
            # Update State Estimate, x_new = x_pred + K * y

            I = np.eye(6)
            P = (I - K @ Phi) @ P_pred
            # Update Covariance Estimate, P_new = (I - K * Phi) * P_pred

        
        estimated_positions.append(x[:2])
        # Store the estimated position
    
    estimates = np.array(estimated_positions)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(observations[:, 0], observations[:, 1], 'r.', label='Noisy Observations', alpha=0.5)
    plt.plot(estimates[:, 0], estimates[:, 1], 'b-', label='Kalman Filter Estimate', linewidth=2)
    plt.title('Kalman Filter Tracking (2D Constant Acceleration Model)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Final State Vector:")
    print(estimates[-1])




def run_fixed_lag_smoother(observations, lag=5):

    dt = 0.1
    sp = 0.001
    sm = 0.05

    state_dim = 6
    # Size of a single state vector

    
    Psi = np.array([
        [1, 0, dt, 0, 0.5 * dt**2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    # Transition matrix that predicts the next state based on physics
    
    Phi = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ])
    # It extracts only the first two elements from the state vector
    
    Sigma_p = sp * np.eye(state_dim)
    # Process Noise Covariance matrix
    

    Sigma_m = sm * np.eye(2)
    # Measurement Noise Covariance matrix

    
    aug_dim = state_dim * (lag + 1)
    # Instead of tracking just the current state, we track the current state and the previous 5 states
    

    Psi_aug = np.zeros((aug_dim, aug_dim))
    # square matrix filled with zeros
    

    Psi_aug[0:state_dim, 0:state_dim] = Psi
    # top block of this matrix represents the current state
    # rest of the matrix is just responsible for shifting old data down the stack,
    # it doesn't calculate new physics
    
    
    # Shift blocks, The state at lag k becomes the state at lag k+1
    for i in range(lag):
        r_idx = (i + 1) * state_dim
        c_idx = i * state_dim
        # Identity matrix shifts the values down
        Psi_aug[r_idx : r_idx + state_dim, c_idx : c_idx + state_dim] = np.eye(state_dim)
        
    
    Phi_aug = np.zeros((2, aug_dim))
    Phi_aug[:, 0:state_dim] = Phi
    # We only observe the current state (lag 0), The rest are zeros
    # It pastes the standard measurement matrix into first six columns
    
    
    Sigma_p_aug = np.zeros((aug_dim, aug_dim))
    Sigma_p_aug[0:state_dim, 0:state_dim] = Sigma_p
    # Augmented Process Noise
    # Only the prediction of the new current state has process noise
    # The shifting of past states is deterministic
    

    
    x0 = np.array([-10, -150, 1, -2, 0, 0])
    # Initial state x0
    
    
    X_aug = np.tile(x0, lag + 1)
    # Initialize augmented state, Stack x0
    
    
    P_aug = np.eye(aug_dim)
    # Initialize augmented covariance
    

    smoothed_estimates = [np.full(state_dim, np.nan) for _ in range(len(observations))]
    # We will store the "smoothed" estimate for time t-L
    # This becomes available when the loop reaches step t


    for t, z_k in enumerate(observations):
        # Prediction Step
        X_aug_pred = Psi_aug @ X_aug
        P_aug_pred = Psi_aug @ P_aug @ Psi_aug.T + Sigma_p_aug
        
        # Update Step
        if np.any(np.isnan(z_k)):
            # Missing Data, Trust prediction
            X_aug = X_aug_pred
            P_aug = P_aug_pred
        else:
            # Valid Data, Update entire history
            innovation = z_k - (Phi_aug @ X_aug_pred)
            S = Phi_aug @ P_aug_pred @ Phi_aug.T + Sigma_m
            K = P_aug_pred @ Phi_aug.T @ np.linalg.inv(S)
            
            X_aug = X_aug_pred + K @ innovation
            I_aug = np.eye(aug_dim)
            P_aug = (I_aug - K @ Phi_aug) @ P_aug_pred

        # Extract Smoothed Estimate
        # The bottom-most block of X_aug corresponds to x_t-lag.
        # So at loop index t, we have the best estimate for time (t - lag).
        if t >= lag:
            
            x_smoothed = X_aug[-state_dim:]
            # Extract the last 6 elements (the state at lag L)
            
           
            smoothed_estimates[t - lag] = x_smoothed
            # Store it at the correct index (t - lag)   


    smoothed_estimates = np.array(smoothed_estimates)



    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(observations[:, 0], observations[:, 1], 'r.', label='Noisy Observations', alpha=0.3)
    
    # We plot the valid range of smoothed estimates, ignoring initial NaNs
    valid_idx = ~np.isnan(smoothed_estimates[:, 0])
    plt.plot(smoothed_estimates[valid_idx, 0], smoothed_estimates[valid_idx, 1], 
                'g-', label=f'Fixed-Lag Smoother (L={lag})', linewidth=2)
    
    plt.title(f'Fixed-Lag Smoothing (Lag={lag}) vs Observations')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Smoothed State at t={len(observations)-lag-1}:")
    print(smoothed_estimates[len(observations)-lag-1])


def generate_unicycle_states():
    
    T = 200            
    dt = 0.1           

    current_state = np.array([0.0, 0.0, 0.0, 1.0])
    # Initial State vector x0 containing [x, y, theta, v]

    Q_diag = np.array([0.001, 0.001, 0.001, 0.001])
    cov_p = np.diag(Q_diag)
    # Process Noise Parameters defining the diagonal of the Covariance Matrix

    states = [current_state.copy()]
    # Storage initialization including the starting state

    for t in range(1, T + 1):
        x_prev, y_prev, theta_prev, v_prev = current_state
        # Unpack previous state variables for calculation

        v_new = v_prev
        # Velocity update based on constant velocity model

        theta_new = theta_prev + 0.6 * np.sin(0.2 * t * dt) * dt
        # Angle update

        x_new = x_prev + dt * v_prev * np.cos(theta_prev)
        y_new = y_prev + dt * v_prev * np.sin(theta_prev)
        # Position updates

        deterministic_state = np.array([x_new, y_new, theta_new, v_new])
        # Assemble the state vector before adding noise

        noise = np.random.multivariate_normal(np.zeros(4), cov_p)
        # Generate random process noise from the multivariate normal distribution

        current_state = deterministic_state + noise
        # Unicycle doesn't land exactly where the physics predicted, lands slightly off
        # Calculate final state for the current step by adding noise

        states.append(current_state)

    states = np.array(states)

    x_vals = states[:, 0]
    y_vals = states[:, 1]
    # Extract X and Y coordinates for visualization



    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'b.-', label='True Trajectory', linewidth=2, markersize=8)
    
    plt.plot(x_vals[0], y_vals[0], 'go', label='Start', markersize=10)
    plt.plot(x_vals[-1], y_vals[-1], 'ro', label='End', markersize=10)
    
    plt.title('Unicycle State Generation (T=200)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal') 
    plt.legend()
    plt.grid(True)
    plt.show()

    print("First 5 States (x, y, theta, v):")
    print(states[:5])

    return states


def generate_unicycle_observations(states):

    R_diag = np.array([0.005, 0.005])
    # Defines how accurate the sensor
    # 0.005 is the variance
    
    cov_m = np.diag(R_diag)
    # Converts the R_diag list into a 2x2 diagonal matrix
    
    observations = []
    
    for x_t in states:
        # The state vector x_t has 4 values

        true_measurement = x_t[:2]
        # h(x), extracts the first two components, position x and y
        
        
        noise = np.random.multivariate_normal(np.zeros(2), cov_m)
        # Add Measurement Noise based on the covariance matrix
        
        
        z_t = true_measurement + noise
        # Compute Final Observation

        observations.append(z_t)

    observations = np.array(observations)


    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot Ground Truth
    plt.plot(states[:, 0], states[:, 1], 'b-', 
             label='True Trajectory', linewidth=2)
    
    # Plot Observations
    plt.plot(observations[:, 0], observations[:, 1], 'r.', 
             label='Noisy Observations', markersize=4, alpha=0.6)
    
    plt.title('Unicycle Model: Truth vs Observations')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("First 5 Observations (x, y):")
    print(observations[:5])


    return observations




def run_extended_kalman_filter(observations, measurement_noise_diag=[0.005, 0.005], ground_truth=None):


    T = len(observations)
    dt = 0.1
    
    # Process Noise
    Q = np.diag([0.001, 0.001, 0.001, 0.001])
    # Represents the uncertainty in the physics model

    # Measurement Noise
    R = np.diag(measurement_noise_diag)
    # Uncertainty in the sensor

    
    x_est = np.array([0.0, 0.0, 0.0, 1.0])
    # Initial State Estimate [x, y, theta, v]


    
    
    P = np.eye(4) * 0.1
    # Initial Covariance Estimate (P)
    # Initialize with identity 4x4 matrix with small variance

    
    estimates = []
    # Storage for estimates


    for t in range(1, T + 1):
        z_k = observations[t-1]
        
        # Prediction Step
        
        # Non linear State Prediction, x_pred = g(x_prev)
        x_prev, y_prev, theta_prev, v_prev = x_est

        
        theta_forcing = 0.6 * np.sin(0.2 * t * dt) * dt
        # an external input that drives the system
        
        x_pred_val = x_prev + dt * v_prev * np.cos(theta_prev)
        y_pred_val = y_prev + dt * v_prev * np.sin(theta_prev)
        # use the non linear function for the state prediction



        theta_pred_val = theta_prev + theta_forcing
        # update the heading
        v_pred_val = v_prev
        
        x_pred = np.array([x_pred_val, y_pred_val, theta_pred_val, v_pred_val])

        # 2. Jacobian Calculation (G_t)
        # Derivatives of g(x) with respect to state [x, y, theta, v]
        # Row 0 (x): [1, 0, -dt*v*sin(theta), dt*cos(theta)]
        # Row 1 (y): [0, 1,  dt*v*cos(theta), dt*sin(theta)]
        # Row 2 (th):[0, 0,  1,               0]
        # Row 3 (v): [0, 0,  0,               1]
        

        # To update the uncertainty Covariance P, we need linear matrix
        G_t = np.eye(4) # Initialize with identity, handles 1s for x, y, th, v diagonals
        G_t[0, 2] = -dt * v_prev * np.sin(theta_prev)
        G_t[0, 3] =  dt * np.cos(theta_prev)
        G_t[1, 2] =  dt * v_prev * np.cos(theta_prev)
        G_t[1, 3] =  dt * np.sin(theta_prev)
        
        # Covariance Prediction
        P_pred = G_t @ P @ G_t.T + Q
        # @ is matrix multiplication, .T for Transpose 




        # Update step, Measurement Update
                
        H_t = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        # Measurement Jacobian (H_t)
        # We observe x and y directly, H is linear
        
        
        z_pred = x_pred[:2]
        y = z_k - z_pred
        # Innovation
        # h(x), extracts the first two elements

        # Innovation Covariance
        S = H_t @ P_pred @ H_t.T + R
        # Total Uncertainty

        K = P_pred @ H_t.T @ np.linalg.inv(S)
        # Kalman Gain

        x_est = x_pred + K @ y
        # State Update

        I = np.eye(4)
        P = (I - K @ H_t) @ P_pred
        # Covariance Update
        
        estimates.append(x_est)

    estimates = np.array(estimates)

    # Visualization
    plt.figure(figsize=(10, 6))
    

    plt.plot(ground_truth[1:, 0], ground_truth[1:, 1], 'k--', label='Ground Truth', linewidth=2, alpha=0.5)

    
    plt.plot(observations[:, 0], observations[:, 1], 'r.', label=f'Observations (Noise={measurement_noise_diag[0]})', alpha=0.3)
    # Plot Noisy Observations
    
    plt.plot(estimates[:, 0], estimates[:, 1], 'b-', label='EKF Estimate', linewidth=2)
    # Plot EKF Estimate

    plt.title('Extended Kalman Filter Tracking')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    return estimates



def main():

    # Load observations
    observations = np.load('data/observations.npy')


    # TASK 1
    run_kalman_filter(observations)
    


    # TASK 2
    run_fixed_lag_smoother(observations)
    


    # TASK 3
    true_states = generate_unicycle_states()

    # Generate Observations
    obs_low_noise = generate_unicycle_observations(true_states)


    print("Running EKF with standard noise,")
    ekf_estimates = run_extended_kalman_filter(obs_low_noise, 
                                               measurement_noise_diag=[0.005, 0.005],
                                               ground_truth=true_states)
    
    print("Running EKF with high noise,")
    ekf_estimates_high = run_extended_kalman_filter(obs_low_noise, 
                                                    measurement_noise_diag=[0.05, 0.05],
                                                    ground_truth=true_states)
    




if __name__ == "__main__":
    main()
