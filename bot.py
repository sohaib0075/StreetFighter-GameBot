import torch
import numpy as np
import time
from model import GameStateModel, predict_action
from buttons import Buttons
from command import Command

class Bot:
    def __init__(self):
        # Load model and data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model, self.scaler, self.command_sequences = self.load_model_and_data()
        
        # Initialize command execution state
        self.exe_code = 0
        self.remaining_commands = []
        self.buttn = Buttons()
        self.my_command = Command()
        
        # Initialize state tracking
        self.previous_commands = []
        self.current_command = None
        self.start_fire = True
        
        # Define fire sequence
        self.fire_sequence = ["<", "!<", "v+<", "!v+!<", "v", "!v", "v+>", "!v+!>", ">+Y", "!>+!Y"]
        self.is_firing = False
        self.fire_index = 0
    
    def load_model_and_data(self):
        """Load the trained model and related data"""
        try:
            # Load model
            input_size = 18  # Number of features
            hidden_size = 256
            command_sequences = np.load('command_sequences.npy', allow_pickle=True)
            command_sequences = command_sequences.tolist()  # Convert numpy array to list
            output_size = len(command_sequences)
            
            model = GameStateModel(input_size, hidden_size, output_size)
            # Add map_location to handle CPU-only devices
            model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
            model = model.to(self.device)
            model.eval()
            
            # Load scaler
            scaler = np.load('scaler.npy', allow_pickle=True).item()
            
            print("Model loaded successfully")
            return model, scaler, command_sequences
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None, None, None
    
    def prepare_game_state(self, game_state):
        """Prepare game state for model input"""
        # Extract features from actual game state
        features = [
            game_state.timer,
            game_state.player1.x_coord,
            game_state.player1.y_coord,
            game_state.player1.health,
            game_state.player2.x_coord,
            game_state.player2.y_coord,
            game_state.player2.health,
            abs(game_state.player1.x_coord - game_state.player2.x_coord),  # distance
            game_state.player2.x_coord - game_state.player1.x_coord,  # relative_x
            game_state.player2.y_coord - game_state.player1.y_coord,  # relative_y
            game_state.player2.player_buttons.Y or game_state.player2.player_buttons.A or 
            game_state.player2.player_buttons.B or game_state.player2.player_buttons.R,  # enemy_attacking
            game_state.player2.player_buttons.up,  # enemy_jumping
            game_state.player2.player_buttons.left or game_state.player2.player_buttons.right,  # enemy_moving
            game_state.player1.damage_dealt if hasattr(game_state.player1, 'damage_dealt') else 0,
            game_state.player1.damage_taken if hasattr(game_state.player1, 'damage_taken') else 0,
            0,  # command_duration (kept for scaler compatibility)
            0   # recovery_time (kept for scaler compatibility)
        ]
        
        # Add previous command index
        if hasattr(self, 'previous_command') and self.previous_command in self.command_sequences:
            features.append(self.command_sequences.index(self.previous_command))
        else:
            features.append(0)
        
        return np.array(features)
    
    def get_top_commands(self, game_state):
        """Get top 5 predicted commands with their probabilities"""
        features = self.prepare_game_state(game_state)
        
        # Scale the input
        features = self.scaler.transform(features.reshape(1, -1))
        features = torch.FloatTensor(features).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(features)
            # Apply temperature scaling to soften predictions
            temperature = 0.7
            scaled_prediction = prediction / temperature
            probs = torch.softmax(scaled_prediction, dim=1)
            
            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            # Convert to list of (command, probability) tuples
            top_commands = []
            for prob, idx in zip(top5_probs[0], top5_indices[0]):
                command = self.command_sequences[idx.item()]
                top_commands.append((command, prob.item()))
            
            return top_commands
    
    def check_fire_opportunity(self, top_commands):
        """Check if current predictions match fire sequence pattern"""
        if not top_commands:
            return False
            
        # Get the top predicted command
        top_cmd, top_prob = top_commands[0]
        
        # Check if it matches any command in fire sequence
        for fire_cmd in self.fire_sequence:
            if fire_cmd == top_cmd and top_prob > 0.4:  # 70% confidence threshold
                return True
        return False

    def fight(self, game_state, player_num):
        """Main interface for the controller"""
        if self.model is None:
            return self.my_command
        
        # If we're executing a command sequence, continue with it
        if self.exe_code != 0 and self.remaining_commands:
            self.run_command(self.remaining_commands, game_state.player1)
            self.my_command.player_buttons = self.buttn
            return self.my_command
        
        # Get top commands from model
        top_commands = self.get_top_commands(game_state)
        if not top_commands:
            self.my_command.player_buttons = self.buttn
            return self.my_command
            
        # Check if we should start fire sequence
        if not self.is_firing and self.check_fire_opportunity(top_commands):
            print("Starting fire sequence")
            self.is_firing = True
            self.fire_index = 0
            self.exe_code = 1
            self.remaining_commands = self.fire_sequence.copy()
            self.run_command(self.remaining_commands, game_state.player1)
            self.my_command.player_buttons = self.buttn
            return self.my_command
            
        # If we're in fire sequence, continue it
        if self.is_firing:
            if not self.remaining_commands:
                self.is_firing = False
                self.fire_index = 0
            else:
                self.run_command(self.remaining_commands, game_state.player1)
                self.my_command.player_buttons = self.buttn
                return self.my_command
        
        # Otherwise use model predictions
        command_sequence = []
        for cmd, prob in top_commands:
            print(f"Adding command to sequence: {cmd} with probability {prob:.3f}")
            # Add press command with separator
            command_sequence.append(cmd)
            command_sequence.append("-")
            # Add release command with separator
            release_cmd = f"!{cmd.replace('+', '+!')}"
            command_sequence.append(release_cmd)
        
        if not command_sequence:
            print("No commands to execute")
            self.my_command.player_buttons = self.buttn
            return self.my_command
            
        print(f"Starting new command sequence: {command_sequence}")
        # Start executing the sequence
        self.exe_code = 1
        self.remaining_commands = command_sequence
        self.run_command(command_sequence, game_state.player1)
        
        # Set the command
        self.my_command.player_buttons = self.buttn
        return self.my_command
    
    def run_command(self, command_sequence, player):
        """Execute a command sequence"""
        if not command_sequence:
            print("Empty command sequence received")
            self.exe_code = 0
            self.remaining_commands = []
            return
            
        # Get the current command
        current_cmd = command_sequence[0]
        print(f"Executing command: {current_cmd}")
        
        # Skip separator commands
        if current_cmd == "-":
            self.remaining_commands = command_sequence[1:]
            return
            
        # Handle combined commands (e.g., v+<)
        if '+' in current_cmd:
            parts = current_cmd.split('+')
            for part in parts:
                if part.startswith('!'):
                    # Handle release commands
                    if part[1:] == 'v':
                        self.buttn.down = False
                    elif part[1:] == '^':
                        self.buttn.up = False
                    elif part[1:] == '<':
                        self.buttn.left = False
                    elif part[1:] == '>':
                        self.buttn.right = False
                    elif part[1:] == 'Y':
                        self.buttn.Y = False
                    elif part[1:] == 'B':
                        self.buttn.B = False
                    elif part[1:] == 'R':
                        self.buttn.R = False
                else:
                    # Handle press commands
                    if part == 'v':
                        self.buttn.down = True
                    elif part == '^':
                        self.buttn.up = True
                    elif part == '<':
                        self.buttn.left = True
                    elif part == '>':
                        self.buttn.right = True
                    elif part == 'Y':
                        self.buttn.Y = True
                    elif part == 'B':
                        self.buttn.B = True
                    elif part == 'R':
                        self.buttn.R = True
        # Handle single commands
        else:
            if current_cmd.startswith('!'):
                # Handle release commands
                cmd = current_cmd[1:]
                if cmd == 'v':
                    self.buttn.down = False
                elif cmd == '^':
                    self.buttn.up = False
                elif cmd == '<':
                    self.buttn.left = False
                elif cmd == '>':
                    self.buttn.right = False
                elif cmd == 'Y':
                    self.buttn.Y = False
                elif cmd == 'B':
                    self.buttn.B = False
                elif cmd == 'R':
                    self.buttn.R = False
            else:
                # Handle press commands
                if current_cmd == 'v':
                    self.buttn.down = True
                elif current_cmd == '^':
                    self.buttn.up = True
                elif current_cmd == '<':
                    self.buttn.left = True
                elif current_cmd == '>':
                    self.buttn.right = True
                elif current_cmd == 'Y':
                    self.buttn.Y = True
                elif current_cmd == 'B':
                    self.buttn.B = True
                elif current_cmd == 'R':
                    self.buttn.R = True
                
        print(f"Current button state: {self.buttn.__dict__}")
        
        # Remove the executed command and update remaining commands
        self.remaining_commands = command_sequence[1:]
        print(f"Remaining commands: {self.remaining_commands}")
        
        # If no more commands, reset execution state
        if not self.remaining_commands:
            print("Command sequence complete")
            self.exe_code = 0
    
    def get_default_buttons(self):
        """Get default button state"""
        return Buttons({
            'Up': False,
            'Down': False,
            'Left': False,
            'Right': False,
            'Select': False,
            'Start': False,
            'Y': False,
            'B': False,
            'X': False,
            'A': False,
            'L': False,
            'R': False
        })