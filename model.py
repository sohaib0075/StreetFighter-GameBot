import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check for GPU availability
device = 'cpu'
print(f"Using device: {device}")

def analyze_dataset(df):
    """Analyze the dataset structure and statistics"""
    print("\nDataset Analysis:")
    print(f"Total rows: {len(df)}")
    
    # Analyze command sequences
    command_sequences = df[['current_command', 'prev_command', 'prev2_command', 'prev3_command']]
    print("\nCommand Sequence Analysis:")
    for col in command_sequences.columns:
        value_counts = command_sequences[col].value_counts()
        print(f"\n{col} distribution:")
        print(value_counts)
    
    # Analyze game state features
    print("\nGame State Analysis:")
    numeric_cols = ['timer', 'player1_health', 'player2_health', 'distance', 
                   'relative_x', 'relative_y', 'damage_dealt', 'damage_taken',
                   'command_duration', 'recovery_time']
    
    for col in numeric_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std: {df[col].std():.2f}")
            print(f"  Min: {df[col].min():.2f}")
            print(f"  Max: {df[col].max():.2f}")
    
    # Analyze boolean features
    bool_cols = ['enemy_attacking', 'enemy_jumping', 'enemy_moving', 'hit_success']
    print("\nBoolean Features Analysis:")
    for col in bool_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(normalize=True))

def clean_dataset(df):
    # Remove rows with invalid health values
    df = df[df['player1_health'] > 0]
    df = df[df['player2_health'] > 0]
    
    # Remove rows with invalid timer
    df = df[df['timer'] > 0]
    
    # Remove rows with invalid coordinates
    df = df[df['player1_x'].notna() & df['player1_y'].notna()]
    df = df[df['player2_x'].notna() & df['player2_y'].notna()]
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    return df

def prepare_data(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Analyze dataset
    analyze_dataset(df)
    
    # Clean the dataset
    df = clean_dataset(df)
    
    # Define input features
    input_features = [
        'timer', 'player1_x', 'player1_y', 'player1_health',
        'player2_x', 'player2_y', 'player2_health',
        'distance', 'relative_x', 'relative_y',
        'enemy_attacking', 'enemy_jumping', 'enemy_moving',
        'damage_dealt', 'damage_taken', 'command_duration', 'recovery_time'
    ]
    
    # Convert boolean columns to int
    bool_cols = ['enemy_attacking', 'enemy_jumping', 'enemy_moving']
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    # Calculate command effectiveness scores
    df['command_score'] = df.apply(lambda row: 
        (row['damage_dealt'] * 2.0) -  # Reward for dealing damage
        (row['damage_taken'] * 3.0) -  # Penalty for taking damage
        (row['command_duration'] * 0.5) -  # Small penalty for long commands
        (row['recovery_time'] * 0.5),  # Small penalty for recovery time
        axis=1
    )
    
    # Get command sequences and their frequencies
    command_counts = df['current_command'].value_counts()
    print("\nCommand Distribution:")
    print(command_counts)
    
    # Calculate command effectiveness by type
    command_effectiveness = df.groupby('current_command')['command_score'].mean()
    print("\nCommand Effectiveness:")
    print(command_effectiveness)
    
    # Calculate class weights based on effectiveness and frequency
    total_samples = len(df)
    class_weights = {}
    
    # Group commands by effectiveness
    high_effect = command_effectiveness[command_effectiveness > 0]
    mid_effect = command_effectiveness[(command_effectiveness <= 0) & (command_effectiveness > -1)]
    low_effect = command_effectiveness[command_effectiveness <= -1]
    
    # Assign weights based on effectiveness and frequency
    for cmd, count in command_counts.items():
        if cmd in high_effect:
            # Higher weight for effective commands
            class_weights[cmd] = np.sqrt(total_samples / (count * 0.8))
        elif cmd in mid_effect:
            # Medium weight for neutral commands
            class_weights[cmd] = np.sqrt(total_samples / count)
        else:
            # Lower weight for ineffective commands
            class_weights[cmd] = np.sqrt(total_samples / (count * 1.2))
    
    # Normalize weights to sum to 1
    weight_sum = sum(class_weights.values())
    class_weights = {k: v/weight_sum for k, v in class_weights.items()}
    
    # Filter out rare commands (less than 1% of total)
    min_count = len(df) * 0.01
    common_commands = command_counts[command_counts >= min_count].index
    df = df[df['current_command'].isin(common_commands)]
    
    # Create command sequences
    command_sequences = list(common_commands)
    command_to_idx = {cmd: idx for idx, cmd in enumerate(command_sequences)}
    
    # Convert commands to indices
    df['command_idx'] = df['current_command'].map(command_to_idx)
    
    # Add previous command features
    for i in range(1, 3):
        prev_col = f'prev{i}_command'
        if prev_col in df.columns:
            df[f'prev{i}_command_idx'] = df[prev_col].map(lambda x: command_to_idx.get(x, 0))
            input_features.append(f'prev{i}_command_idx')
    
    # Prepare input and output data
    X = df[input_features].values
    y = df['command_idx'].values
    
    # Calculate sample weights for training
    sample_weights = np.array([class_weights[cmd] for cmd in df['current_command']])
    
    # Split data with stratification
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, command_sequences, weights_train

class GameStateModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GameStateModel, self).__init__()
        
        # Feature extraction layers with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(2)  # Two residual blocks
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Command sequence predictor
        self.command_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Residual connections
        for residual_block in self.residual_blocks:
            residual = residual_block(features)
            features = features + residual
        
        # Attention mechanism
        attention_weights = self.attention(features)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_features = features * attention_weights
        
        # Command prediction
        return self.command_predictor(attended_features)

def train_model(X_train, y_train, sample_weights, epochs=300, batch_size=32):
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    hidden_size = 256
    
    model = GameStateModel(input_size, hidden_size, output_size)
    model = model.to(device)
    
    # Calculate class weights for loss function
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    
    # Group classes by frequency
    high_freq_mask = class_counts >= 1000
    mid_freq_mask = (class_counts >= 500) & (class_counts < 1000)
    low_freq_mask = class_counts < 500
    
    # Initialize weights array
    class_weights = np.ones_like(class_counts, dtype=float)
    
    # Apply different weights based on frequency
    class_weights[high_freq_mask] = total_samples / (class_counts[high_freq_mask] * 2.0)
    class_weights[mid_freq_mask] = total_samples / class_counts[mid_freq_mask]
    class_weights[low_freq_mask] = total_samples / (class_counts[low_freq_mask] * 0.6)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Use weighted loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.02)
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(X_train) // batch_size + 1,
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100
    )
    
    # Convert to PyTorch tensors and move to GPU
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    sample_weights = torch.FloatTensor(sample_weights).to(device)
    
    # Create weighted sampler for batch training
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create DataLoader with weighted sampling
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        # Early stopping check based on both loss and accuracy
        if avg_loss < best_loss or accuracy > best_accuracy:
            if avg_loss < best_loss:
                best_loss = avg_loss
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def predict_action(model, game_state, scaler, command_sequences, prev_commands=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add previous commands to game state if available
    if prev_commands is not None:
        for i, cmd in enumerate(prev_commands[:2]):  # Only use 2 previous commands
            if cmd in command_sequences:
                command_idx = command_sequences.index(cmd)
                game_state = np.append(game_state, command_idx)
            else:
                game_state = np.append(game_state, 0)
    
    # Scale the input
    game_state = scaler.transform(game_state.reshape(1, -1))
    game_state = torch.FloatTensor(game_state).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        prediction = model(game_state)
        # Apply temperature scaling to soften predictions
        temperature = 0.7
        scaled_prediction = prediction / temperature
        probs = torch.softmax(scaled_prediction, dim=1)
        
        # Sample from the probability distribution
        command_idx = torch.multinomial(probs, 1).item()
        command = command_sequences[command_idx]
    
    # Initialize command sequence
    command_sequence = []
    
    # Add the predicted command and its release
    if command.startswith("!"):
        command_sequence = [command]
    else:
        command_sequence = [command, f"!{command.replace('+', '+!')}"]
    
    return command_sequence

def test_model(model, X_test, y_test, scaler, command_sequences):
    """Test the model and evaluate its performance"""
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_indices = predictions.argmax(dim=1).cpu().numpy()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_indices)
    
    # Ensure command sequences are strings and not None
    command_sequences = [str(cmd) if cmd is not None else f"Command_{i}" 
                        for i, cmd in enumerate(command_sequences)]
    
    # Generate classification report
    report = classification_report(y_test, predicted_indices, 
                                 target_names=command_sequences,
                                 zero_division=0)
    
    return accuracy, report, predicted_indices

def analyze_command_distribution(y_true, y_pred, command_sequences):
    """Analyze the distribution of predicted commands"""
    # Ensure command sequences are strings and not None
    command_sequences = [str(cmd) if cmd is not None else f"Command_{i}" 
                        for i, cmd in enumerate(command_sequences)]
    
    # Count occurrences of each command
    true_counts = np.bincount(y_true, minlength=len(command_sequences))
    pred_counts = np.bincount(y_pred, minlength=len(command_sequences))
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'Command': command_sequences,
        'True Count': true_counts,
        'Predicted Count': pred_counts
    })
    
    # Plot command distribution
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='Command', y='True Count')
    plt.xticks(rotation=45, ha='right')
    plt.title('True Command Distribution')
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='Command', y='Predicted Count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Predicted Command Distribution')
    
    plt.tight_layout()
    plt.savefig('command_distribution.png')
    plt.close()

def test_specific_scenarios(model, scaler, command_sequences):
    """Test model predictions for specific game scenarios"""
    scenarios = [
        # Close range attack scenario
        {
            'name': 'Close Range Attack',
            'features': [
                60,  # timer
                100,  # player1_x
                0,   # player1_y
                100, # player1_health
                120, # player2_x
                0,   # player2_y
                100, # player2_health
                20,  # distance
                20,  # relative_x
                0,   # relative_y
                0,   # enemy_attacking
                0,   # enemy_jumping
                0,   # enemy_moving
                0,   # damage_dealt
                0,   # damage_taken
                0,   # command_duration
                0,   # recovery_time
                0,   # prev1_command_idx
                0    # prev2_command_idx
            ]
        },
        # Long range scenario
        {
            'name': 'Long Range',
            'features': [
                60,  # timer
                50,  # player1_x
                0,   # player1_y
                100, # player1_health
                200, # player2_x
                0,   # player2_y
                100, # player2_health
                150, # distance
                150, # relative_x
                0,   # relative_y
                0,   # enemy_attacking
                0,   # enemy_jumping
                0,   # enemy_moving
                0,   # damage_dealt
                0,   # damage_taken
                0,   # command_duration
                0,   # recovery_time
                0,   # prev1_command_idx
                0    # prev2_command_idx
            ]
        },
        # Enemy attacking scenario
        {
            'name': 'Enemy Attacking',
            'features': [
                60,  # timer
                100, # player1_x
                0,   # player1_y
                100, # player1_health
                120, # player2_x
                0,   # player2_y
                100, # player2_health
                20,  # distance
                20,  # relative_x
                0,   # relative_y
                1,   # enemy_attacking
                0,   # enemy_jumping
                0,   # enemy_moving
                0,   # damage_dealt
                0,   # damage_taken
                0,   # command_duration
                0,   # recovery_time
                0,   # prev1_command_idx
                0    # prev2_command_idx
            ]
        }
    ]
    
    print("\nTesting specific scenarios:")
    for scenario in scenarios:
        features = np.array(scenario['features'])
        command_sequence = predict_action(model, features, scaler, command_sequences)
        print(f"\n{scenario['name']}:")
        print(f"Predicted command sequence: {command_sequence}")

def main():
    # Check if dataset exists
    if not os.path.exists('game_data.csv'):
        print("Error: game_data.csv not found. Please collect data first.")
        return
    
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler, command_sequences, sample_weights = prepare_data('game_data.csv')
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of unique commands: {len(command_sequences)}")
    
    print("\nTraining model...")
    model = train_model(X_train, y_train, sample_weights)
    
    # Save model and related data
    print("\nSaving model and data...")
    torch.save(model.state_dict(), 'game_model.pth')
    np.save('scaler.npy', scaler)
    np.save('command_sequences.npy', command_sequences)
    
    print("\nModel training completed. Run test_model.py to evaluate the model.")

if __name__ == "__main__":
    main() 