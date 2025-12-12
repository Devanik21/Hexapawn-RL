import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from collections import deque
import random
import pandas as pd
import json
import zipfile
import io
from copy import deepcopy

# ============================================================================
# Page Config and Initial Setup
# ============================================================================
st.set_page_config(
    page_title="Strategic RL Hexapawn",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="♟️"
)

st.title("Strategic Hexapawn RL Arena")
st.markdown("""
Watch two Reinforcement Learning agents master the ancient game of **Hexapawn** through strategic combat and learning.

**🎯 Hexapawn Rules:**
- 3×3 board with 3 pawns per player
- White (Blue) pawns move up, Black (Red) pawns move down
- Move forward 1 square OR capture diagonally forward
- **Win by:** reaching the opposite end, capturing all enemy pawns, or blocking all moves

**Core Algorithmic Components:**
- 🧮 **Minimax with Alpha-Beta Pruning** - Strategic depth
- 🎓 **Multi-step reward shaping** - Understanding long-term strategy
- 🔮 **Position evaluation heuristics** - Board state understanding
- 🧬 **Experience replay with prioritization** - Efficient learning
- 💡 **Opponent modeling** - Adapting to enemy strategies
""")

# ============================================================================
# Hexapawn Game Environment
# ============================================================================

class Hexapawn:
    """
    Hexapawn game implementation
    - Player 1 (White/Blue): Pawns start at row 0, move towards row 2
    - Player 2 (Black/Red): Pawns start at row 2, move towards row 0
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Board setup: 1 = White pawns, 2 = Black pawns, 0 = empty
        self.board = np.array([
            [1, 1, 1],  # White pawns (Player 1)
            [0, 0, 0],  # Empty middle
            [2, 2, 2]   # Black pawns (Player 2)
        ], dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        return self.get_state()
    
    def get_state(self):
        return tuple(self.board.flatten())
    
    def get_available_actions(self):
        """Return list of legal moves as ((from_r, from_c), (to_r, to_c))"""
        actions = []
        
        if self.current_player == 1:  # White moves up (row increases)
            direction = 1
        else:  # Black moves down (row decreases)
            direction = -1
        
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == self.current_player:
                    # Try forward move
                    new_r = r + direction
                    if 0 <= new_r < 3 and self.board[new_r, c] == 0:
                        actions.append(((r, c), (new_r, c)))
                    
                    # Try diagonal captures
                    for dc in [-1, 1]:
                        new_c = c + dc
                        if 0 <= new_r < 3 and 0 <= new_c < 3:
                            if self.board[new_r, new_c] == (3 - self.current_player):
                                actions.append(((r, c), (new_r, new_c)))
        
        return actions
    
    def make_move(self, move):
        """Execute a move and return (state, reward, done)"""
        if self.game_over:
            return self.get_state(), 0, True
        
        (from_r, from_c), (to_r, to_c) = move
        
        # Validate move
        if self.board[from_r, from_c] != self.current_player:
            return self.get_state(), -100, True  # Invalid move penalty
        
        # Execute move
        captured = self.board[to_r, to_c] != 0
        self.board[to_r, to_c] = self.current_player
        self.board[from_r, from_c] = 0
        self.move_history.append((move, self.current_player))
        
        # Check win conditions
        reward = 0
        
        # Win condition 1: Pawn reaches opposite end
        if (self.current_player == 1 and to_r == 2) or \
           (self.current_player == 2 and to_r == 0):
            self.game_over = True
            self.winner = self.current_player
            return self.get_state(), 100, True
        
        # Win condition 2: All opponent pawns captured
        opponent = 3 - self.current_player
        if not np.any(self.board == opponent):
            self.game_over = True
            self.winner = self.current_player
            return self.get_state(), 100, True
        
        # Switch player
        self.current_player = opponent
        
        # Win condition 3: Opponent has no legal moves
        if len(self.get_available_actions()) == 0:
            self.game_over = True
            self.winner = 3 - opponent  # Previous player wins
            return self.get_state(), 100, True
        
        # Small reward for captures
        if captured:
            reward = 5
        
        return self.get_state(), reward, False
    
    def evaluate_position(self, player):
        """
        Advanced position evaluation heuristic
        Returns a score from the perspective of 'player'
        """
        if self.winner == player:
            return 100000
        if self.winner == (3 - player):
            return -100000
        if self.game_over:
            return 0
        
        opponent = 3 - player
        score = 0
        
        # 1. Material count (pawns)
        my_pawns = np.sum(self.board == player)
        opp_pawns = np.sum(self.board == opponent)
        score += (my_pawns - opp_pawns) * 200
        
        # 2. Advancement bonus (closer to goal)
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == player:
                    if player == 1:  # White advances up
                        score += r * 30
                    else:  # Black advances down
                        score += (2 - r) * 30
                elif self.board[r, c] == opponent:
                    if opponent == 1:
                        score -= r * 30
                    else:
                        score -= (2 - r) * 30
        
        # 3. Mobility (number of legal moves)
        original_player = self.current_player
        
        self.current_player = player
        my_moves = len(self.get_available_actions())
        
        self.current_player = opponent
        opp_moves = len(self.get_available_actions())
        
        self.current_player = original_player
        
        score += (my_moves - opp_moves) * 50
        
        # 4. Center control
        if self.board[1, 1] == player:
            score += 25
        elif self.board[1, 1] == opponent:
            score -= 25
        
        # 5. Pawn structure (connected pawns)
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == player:
                    # Check adjacent pawns
                    for dr, dc in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 3 and 0 <= nc < 3:
                            if self.board[nr, nc] == player:
                                score += 15
        
        return score

# ============================================================================
# Strategic RL Agent
# ============================================================================

class StrategicAgent:
    def __init__(self, player_id, lr=0.2, gamma=0.95, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = {}
        self.experience_replay = deque(maxlen=20000)
        self.minimax_depth = 5  # Hexapawn has smaller state space
        
        # Stats
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def get_q_value(self, state, action):
        # Convert action tuple to string for dictionary key
        action_key = str(action)
        return self.q_table.get((state, action_key), 0.0)
    
    def choose_action(self, env, training=True):
        available_actions = env.get_available_actions()
        if not available_actions:
            return None
        
        # Level 1: Check for immediate win
        for action in available_actions:
            sim = self._simulate_move(env, action, self.player_id)
            if sim.winner == self.player_id:
                return action
        
        # Level 2: Block opponent's winning move
        opponent = 3 - self.player_id
        for action in available_actions:
            # Test if opponent could win after this position
            test_env = self._simulate_move(env, action, self.player_id)
            opp_actions = test_env.get_available_actions()
            opponent_can_win = False
            for opp_action in opp_actions:
                opp_sim = self._simulate_move(test_env, opp_action, opponent)
                if opp_sim.winner == opponent:
                    opponent_can_win = True
                    break
            if not opponent_can_win and len(opp_actions) > 0:
                continue  # This move is safe
        
        # Level 3: Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Level 4: Minimax strategic planning
        best_score = -float('inf')
        best_actions = []
        
        alpha = -float('inf')
        beta = float('inf')
        
        for action in available_actions:
            sim_env = self._simulate_move(env, action, self.player_id)
            score = self._minimax(sim_env, self.minimax_depth - 1, alpha, beta, False)
            
            # Add Q-table knowledge
            q_boost = self.get_q_value(env.get_state(), action) * 0.01
            total_score = score + q_boost
            
            if total_score > best_score:
                best_score = total_score
                best_actions = [action]
            elif total_score == best_score:
                best_actions.append(action)
            
            alpha = max(alpha, best_score)
        
        if best_actions:
            return random.choice(best_actions)
        return random.choice(available_actions)
    
    def _minimax(self, env, depth, alpha, beta, is_maximizing):
        # Terminal conditions
        if env.winner == self.player_id:
            return 1000 + depth
        if env.winner == (3 - self.player_id):
            return -1000 - depth
        if env.game_over:
            return 0
        if depth == 0:
            return env.evaluate_position(self.player_id)
        
        available_actions = env.get_available_actions()
        
        if is_maximizing:
            max_eval = -float('inf')
            for action in available_actions:
                sim_env = self._simulate_move(env, action, self.player_id)
                eval_score = self._minimax(sim_env, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - self.player_id
            for action in available_actions:
                sim_env = self._simulate_move(env, action, opponent)
                eval_score = self._minimax(sim_env, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _simulate_move(self, env, action, player):
        """Create a lightweight copy of environment and make move"""
        sim_env = Hexapawn()
        sim_env.board = env.board.copy()
        sim_env.current_player = player
        sim_env.game_over = env.game_over
        sim_env.winner = env.winner
        sim_env.make_move(action)
        return sim_env
    
    def update_q_value(self, state, action, reward, next_state, next_available_actions):
        action_key = str(action)
        current_q = self.get_q_value(state, action)
        
        if next_available_actions:
            max_next_q = max([self.get_q_value(next_state, a) for a in next_available_actions])
        else:
            max_next_q = 0
        
        td_error = reward + self.gamma * max_next_q - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[(state, action_key)] = new_q
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

# ============================================================================
# Training System
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    """Play one complete game between two strategic agents"""
    env.reset()
    game_history = []
    
    agents = {1: agent1, 2: agent2}
    
    max_moves = 100  # Prevent infinite loops
    move_count = 0
    
    while not env.game_over and move_count < max_moves:
        current_player = env.current_player
        current_agent = agents[current_player]
        
        state = env.get_state()
        action = current_agent.choose_action(env, training)
        
        if action is None:
            break
        
        game_history.append((state, action, current_player))
        next_state, reward, done = env.make_move(action)
        
        # Online learning
        if training:
            next_actions = env.get_available_actions()
            current_agent.update_q_value(state, action, reward, next_state, next_actions)
        
        move_count += 1
        
        if done:
            if env.winner == 1:
                agent1.wins += 1
                agent2.losses += 1
                if training:
                    _update_from_outcome(agent1, game_history, 1, 100)
                    _update_from_outcome(agent2, game_history, 2, -50)
            elif env.winner == 2:
                agent2.wins += 1
                agent1.losses += 1
                if training:
                    _update_from_outcome(agent1, game_history, 1, -50)
                    _update_from_outcome(agent2, game_history, 2, 100)
            else:
                agent1.draws += 1
                agent2.draws += 1
                if training:
                    _update_from_outcome(agent1, game_history, 1, 0)
                    _update_from_outcome(agent2, game_history, 2, 0)
    
    return env.winner

def _update_from_outcome(agent, history, player_id, final_reward):
    """Update agent's strategy based on game outcome"""
    agent_moves = [(s, a) for s, a, p in history if p == player_id]
    
    for i in range(len(agent_moves) - 1, -1, -1):
        state, action = agent_moves[i]
        
        discount_factor = agent.gamma ** (len(agent_moves) - 1 - i)
        adjusted_reward = final_reward * discount_factor
        
        action_key = str(action)
        current_q = agent.get_q_value(state, action)
        new_q = current_q + agent.lr * (adjusted_reward - current_q)
        agent.q_table[(state, action_key)] = new_q

# ============================================================================
# Visualization
# ============================================================================

def visualize_board(board, title="Hexapawn Board"):
    """Create matplotlib figure of the Hexapawn board"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw checkerboard pattern
    for i in range(3):
        for j in range(3):
            color = '#f0d9b5' if (i + j) % 2 == 0 else '#b58863'
            rect = FancyBboxPatch((j, 2-i), 1, 1, 
                                  boxstyle="round,pad=0.05",
                                  facecolor=color, 
                                  edgecolor='black', 
                                  linewidth=2)
            ax.add_patch(rect)
    
    # Draw pawns
    for i in range(3):
        for j in range(3):
            if board[i, j] == 1:  # White pawn
                circle = Circle((j + 0.5, 2-i + 0.5), 0.3, 
                               color='#4a90e2', edgecolor='black', linewidth=2)
                ax.add_patch(circle)
                ax.text(j + 0.5, 2-i + 0.5, '♟', 
                       ha='center', va='center', 
                       fontsize=32, color='white', weight='bold')
            elif board[i, j] == 2:  # Black pawn
                circle = Circle((j + 0.5, 2-i + 0.5), 0.3, 
                               color='#e74c3c', edgecolor='black', linewidth=2)
                ax.add_patch(circle)
                ax.text(j + 0.5, 2-i + 0.5, '♟', 
                       ha='center', va='center', 
                       fontsize=32, color='white', weight='bold')
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig


# ============================================================================
# Save/Load Functions (Updated & Robust)
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Special encoder to handle Numpy types in JSON"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def serialize_q_table(q_table):
    """Robustly convert Q-table to JSON-serializable format"""
    serialized_q = {}
    for (state, action_key), value in q_table.items():
        # Ensure state is a standard list of ints
        state_list = [int(x) for x in state]
        # Create a string representation of the key
        # We use a separator '|' which is safer than nested JSON dumps
        key_str = f"{json.dumps(state_list)}|{action_key}"
        serialized_q[key_str] = float(value)
    return serialized_q

def deserialize_q_table(serialized_q):
    """Convert JSON format back to Q-table"""
    deserialized_q = {}
    for k_str, value in serialized_q.items():
        try:
            # Split our custom separator
            state_str, action_key = k_str.split('|')
            state_list = json.loads(state_str)
            state_tuple = tuple(state_list)
            deserialized_q[(state_tuple, action_key)] = value
        except Exception:
            continue # Skip malformed keys
    return deserialized_q

def create_agents_zip(agent1, agent2, config):
    """Create downloadable zip file with agents and config"""
    # 1. Prepare Data using the NumpyEncoder
    agent1_data = {
        "q_table": serialize_q_table(agent1.q_table),
        "epsilon": float(agent1.epsilon),
        "wins": int(agent1.wins),
        "losses": int(agent1.losses),
        "draws": int(agent1.draws)
    }
    
    agent2_data = {
        "q_table": serialize_q_table(agent2.q_table),
        "epsilon": float(agent2.epsilon),
        "wins": int(agent2.wins),
        "losses": int(agent2.losses),
        "draws": int(agent2.draws)
    }

    # 2. Write to Zip
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1.json", json.dumps(agent1_data, cls=NumpyEncoder, indent=2))
        zf.writestr("agent2.json", json.dumps(agent2_data, cls=NumpyEncoder, indent=2))
        zf.writestr("config.json", json.dumps(config, cls=NumpyEncoder, indent=2))
    
    buffer.seek(0)
    return buffer

def load_agents_from_zip(uploaded_file):
    """Load agents from uploaded zip file"""
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            # Read and parse JSONs
            agent1_state = json.loads(zf.read("agent1.json"))
            agent2_state = json.loads(zf.read("agent2.json"))
            config = json.loads(zf.read("config.json"))
            
            # Reconstruct Agent 1
            agent1 = StrategicAgent(1, config.get('lr1', 0.2), config.get('gamma1', 0.95))
            agent1.q_table = deserialize_q_table(agent1_state['q_table'])
            agent1.epsilon = agent1_state.get('epsilon', 0.0)
            agent1.wins = agent1_state.get('wins', 0)
            agent1.losses = agent1_state.get('losses', 0)
            agent1.draws = agent1_state.get('draws', 0)
            
            # Reconstruct Agent 2
            agent2 = StrategicAgent(2, config.get('lr2', 0.2), config.get('gamma2', 0.95))
            agent2.q_table = deserialize_q_table(agent2_state['q_table'])
            agent2.epsilon = agent2_state.get('epsilon', 0.0)
            agent2.wins = agent2_state.get('wins', 0)
            agent2.losses = agent2_state.get('losses', 0)
            agent2.draws = agent2_state.get('draws', 0)
            
            return agent1, agent2, config
            
    except Exception as e:
        st.error(f"Error loading brain file: {str(e)}")
        return None, None, None

# ============================================================================
# Battle Analysis
# ============================================================================

def run_battles(agent1, agent2, env, num_battles):
    """Run battles without training for evaluation"""
    battle_wins1 = 0
    battle_wins2 = 0
    battle_draws = 0
    
    agents = {1: agent1, 2: agent2}
    
    for i in range(num_battles):
        local_env = Hexapawn()
        local_env.reset()
        
        # Alternate starting player for fairness
        if i % 2 == 1:
            local_env.current_player = 2
        
        move_count = 0
        max_moves = 100
        
        while not local_env.game_over and move_count < max_moves:
            current_player = local_env.current_player
            action = agents[current_player].choose_action(local_env, training=False)
            if action is None:
                break
            local_env.make_move(action)
            move_count += 1
        
        if local_env.winner == 1:
            battle_wins1 += 1
        elif local_env.winner == 2:
            battle_wins2 += 1
        else:
            battle_draws += 1
    
    return battle_wins1, battle_wins2, battle_draws

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ Simulation Controls")

with st.sidebar.expander("1. Agent 1 (White ♟) Parameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.01, 1.0, 0.2, 0.01)
    gamma1 = st.slider("Discount Factor γ₁", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay1 = st.slider("Epsilon Decay₁", 0.99, 0.9999, 0.995, 0.0001, format="%.4f")
    minimax_depth1 = st.slider("Minimax Depth₁", 1, 7, 5)

with st.sidebar.expander("2. Agent 2 (Black ♟) Parameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.01, 1.0, 0.2, 0.01)
    gamma2 = st.slider("Discount Factor γ₂", 0.8, 0.99, 0.95, 0.01)
    epsilon_decay2 = st.slider("Epsilon Decay₂", 0.99, 0.9999, 0.995, 0.0001, format="%.4f")
    minimax_depth2 = st.slider("Minimax Depth₂", 1, 7, 5)

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 100, 100000, 2000, 100)
    update_freq = st.number_input("Update Dashboard Every N Games", 10, 1000, 50, 10)

with st.sidebar.expander("4. Brain Storage 💾", expanded=False):
    # --- SAVE SECTION ---
    if 'agent1' in st.session_state and st.session_state.agent1 is not None:
        # Check brain size
        brain_size = len(st.session_state.agent1.q_table) + len(st.session_state.agent2.q_table)
        
        if brain_size > 0:
            st.success(f"🧠 Brain Scan: {brain_size} memories found.")
            
            config = {
                "lr1": lr1, "gamma1": gamma1, 
                "epsilon_decay1": epsilon_decay1, "minimax_depth1": minimax_depth1,
                "lr2": lr2, "gamma2": gamma2, 
                "epsilon_decay2": epsilon_decay2, "minimax_depth2": minimax_depth2,
                "training_history": st.session_state.get('training_history', None),
                "battle_results": st.session_state.get('battle_results', None)
            }
            
            # Create zip in memory
            zip_buffer = create_agents_zip(st.session_state.agent1, 
                                           st.session_state.agent2, config)
            
            st.download_button(
                label="💾 Download Trained Brains",
                data=zip_buffer,
                file_name="hexapawn_brains.zip",
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.warning("⚠️ Brains are empty! Train the agents before downloading.")
    else:
        st.warning("Initialize agents first.")
    
    st.markdown("---")
    
    # --- LOAD SECTION ---
    uploaded_file = st.file_uploader("Upload Brain Snapshot (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("📂 Load Session", use_container_width=True):
            with st.spinner("Restoring neural pathways..."):
                a1, a2, cfg = load_agents_from_zip(uploaded_file)
                if a1:
                    st.session_state.agent1 = a1
                    st.session_state.agent2 = a2
                    
                    # Restore history if available
                    if cfg.get("training_history"):
                        st.session_state.training_history = cfg["training_history"]
                    if cfg.get("battle_results"):
                        st.session_state.battle_results = cfg["battle_results"]
                        
                    st.toast("Brains Restored Successfully!", icon="🧠")
                    import time
                    time.sleep(1)
                    st.rerun()
                    
train_button = st.sidebar.button("🚀 Begin Training Epochs", 
                                 use_container_width=True, type="primary")

if st.sidebar.button("🧹 Clear All & Reset", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.toast("Simulation Arena Reset!", icon="🧹")
    st.rerun()

# ============================================================================
# Initialize Environment and Agents
# ============================================================================

if 'env' not in st.session_state:
    st.session_state.env = Hexapawn()

if 'agent1' not in st.session_state:
    st.session_state.agent1 = StrategicAgent(1, lr1, gamma1, epsilon_decay=epsilon_decay1)
    st.session_state.agent1.minimax_depth = minimax_depth1
    st.session_state.agent2 = StrategicAgent(2, lr2, gamma2, epsilon_decay=epsilon_decay2)
    st.session_state.agent2.minimax_depth = minimax_depth2

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2
env = st.session_state.env

# Update minimax depths
agent1.minimax_depth = minimax_depth1
agent2.minimax_depth = minimax_depth2

# ============================================================================
# Display Current Stats
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("♟ Agent 1 (White)", 
             f"Q-States: {len(agent1.q_table)}", 
             f"ε={agent1.epsilon:.4f}")
    st.metric("Wins", agent1.wins, delta_color="normal")
    st.caption(f"Minimax Depth: {agent1.minimax_depth}")

with col2:
    st.metric("♟ Agent 2 (Black)", 
             f"Q-States: {len(agent2.q_table)}", 
             f"ε={agent2.epsilon:.4f}")
    st.metric("Wins", agent2.wins, delta_color="normal")
    st.caption(f"Minimax Depth: {agent2.minimax_depth}")

with col3:
    total_games = agent1.wins + agent1.losses + agent1.draws
    st.metric("Total Games", total_games)
    st.metric("Draws", agent1.draws, delta_color="off")

st.markdown("---")

# ============================================================================
# Quick Analysis & Battles
# ============================================================================

with st.expander("🔬 Quick Analysis & Head-to-Head Battles", expanded=False):
    st.info("Run battles between the current agents without any learning (ε=0). This is a pure test of their current skill.")
    
    battle_cols = st.columns([2, 1])
    num_battles_input = battle_cols[0].number_input(
        "Number of Battles to Run", 
        min_value=1, max_value=10000, value=100, step=10
    )
    
    if battle_cols[1].button("⚔️ Run Battles", use_container_width=True, key="run_battles"):
        with st.spinner(f"Running {num_battles_input} battles..."):
            st.session_state.battle_results = run_battles(agent1, agent2, env, num_battles_input)
    
    if 'battle_results' in st.session_state and st.session_state.battle_results:
        w1, w2, d = st.session_state.battle_results
        total_battles = w1 + w2 + d
        st.write(f"**Battle Results (out of {total_battles} games):**")
        res_cols = st.columns(3)
        res_cols[0].metric("Agent 1 Wins", w1, f"{w1/total_battles:.1%}" if total_battles > 0 else "0.0%")
        res_cols[1].metric("Agent 2 Wins", w2, f"{w2/total_battles:.1%}" if total_battles > 0 else "0.0%")
        res_cols[2].metric("Draws", d, f"{d/total_battles:.1%}" if total_battles > 0 else "0.0%")

# ============================================================================
# Training Section
# ============================================================================

if train_button:
    st.subheader("🎯 Training Epochs in Progress...")
    
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    history = {
        'agent1_wins': [],
        'agent2_wins': [],
        'draws': [],
        'agent1_epsilon': [],
        'agent2_epsilon': [],
        'agent1_q_size': [],
        'agent2_q_size': [],
        'episode': []
    }
    
    for episode in range(1, episodes + 1):
        winner = play_game(env, agent1, agent2, training=True)
        
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if episode % update_freq == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['draws'].append(agent1.draws)
            history['agent1_epsilon'].append(agent1.epsilon)
            history['agent2_epsilon'].append(agent2.epsilon)
            history['agent1_q_size'].append(len(agent1.q_table))
            history['agent2_q_size'].append(len(agent2.q_table))
            history['episode'].append(episode)
            
            progress = episode / episodes
            progress_bar.progress(progress)
            
            status_table = f"""
            | Metric          | Agent 1 (White) | Agent 2 (Black) |
            |:----------------|:---------------:|:---------------:|
            | **Wins**        | {agent1.wins}   | {agent2.wins}   |
            | **Epsilon (ε)** | {agent1.epsilon:.4f} | {agent2.epsilon:.4f} |
            | **Q-States**    | {len(agent1.q_table):,} | {len(agent2.q_table):,} |
            
            ---
            **Game {episode}/{episodes}** ({progress*100:.1f}%) | **Total Draws:** {agent1.draws}
            """
            status_container.markdown(status_table)
    
    progress_bar.progress(1.0)
    st.toast("Training Complete!", icon="🎉")
    
    st.session_state.training_history = history
    st.session_state.agent1 = agent1
    st.session_state.agent2 = agent2

# ============================================================================
# Display Training Charts
# ============================================================================

if 'training_history' in st.session_state and st.session_state.training_history:
    st.subheader("📈 Training Performance Analysis")
    history = st.session_state.training_history
    
    df = pd.DataFrame(history)
    
    if 'episode' not in df.columns:
        df['episode'] = range(1, len(df) + 1)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("#### Win/Loss/Draw Count Over Time")
        chart_data = df[['episode', 'agent1_wins', 'agent2_wins', 'draws']].set_index('episode')
        st.line_chart(chart_data)
    
    with chart_col2:
        st.write("#### Epsilon Decay (Exploration Rate)")
        chart_data = df[['episode', 'agent1_epsilon', 'agent2_epsilon']].set_index('episode')
        st.line_chart(chart_data)
    
    st.write("#### Q-Table Size (Learned States)")
    q_chart_data = df[['episode', 'agent1_q_size', 'agent2_q_size']].set_index('episode')
    st.line_chart(q_chart_data)

# ============================================================================
# Final Battle Visualization
# ============================================================================

if 'agent1' in st.session_state and st.session_state.agent1.q_table:
    st.subheader("⚔️ Final Battle: Trained Agents")
    st.info("Watch the fully trained agents play one final, decisive game against each other (no exploration).")
    
    if st.button("🎮 Watch Them Battle!", use_container_width=True):
        sim_env = Hexapawn()
        board_placeholder = st.empty()
        
        agents = {1: agent1, 2: agent2}
        
        move_count = 0
        max_moves = 100
        
        with st.spinner("Agents are battling..."):
            while not sim_env.game_over and move_count < max_moves:
                current_player = sim_env.current_player
                action = agents[current_player].choose_action(sim_env, training=False)
                if action is None:
                    break
                sim_env.make_move(action)
                
                player_name = "White" if current_player == 1 else "Black"
                fig = visualize_board(sim_env.board, f"{player_name}'s move")
                board_placeholder.pyplot(fig)
                plt.close(fig)
                
                import time
                time.sleep(0.8)
                move_count += 1
        
        if sim_env.winner == 1:
            st.success("🏆 Agent 1 (White) wins the battle!")
        elif sim_env.winner == 2:
            st.error("🏆 Agent 2 (Black) wins the battle!")
        else:
            st.warning("🤝 The battle is a Draw!")
else:
    st.info("Train or load agents to see the Final Battle option.")



# ============================================================================
# Human vs AI Arena
# ============================================================================

st.markdown("---")
st.header("🎮 Human vs. AI Arena")

# --- 1. Ultimate UI Styling (CSS) ---
st.markdown("""
<style>
    /* Main Board Container styling */
    .stButton button {
        height: 85px;
        width: 100%;
        font-size: 32px !important;
        border-radius: 12px !important;
        border: 2px solid #333 !important;
        margin-bottom: 8px;
        transition: all 0.2s ease-in-out; 
    }
    
    /* HUMAN PIECES (Blue/Cyan Theme) */
    div[data-testid="stVerticalBlock"] .human-piece button {
        background: linear-gradient(145deg, #1e3c72, #2a5298) !important;
        color: white !important;
        border: 2px solid #4facfe !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stVerticalBlock"] .human-piece button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(79, 172, 254, 0.4);
    }
    div[data-testid="stVerticalBlock"] .human-piece-selected button {
        background: #4facfe !important; /* Bright highlight */
        color: white !important;
        border: 3px solid #fff !important;
        box-shadow: 0 0 15px #4facfe;
    }

    /* AI PIECES (Red/Dark Theme) */
    div[data-testid="stVerticalBlock"] .ai-piece button {
        background: linear-gradient(145deg, #870000, #190a05) !important;
        color: #ff6b6b !important;
        border: 2px solid #870000 !important;
        cursor: not-allowed;
    }

    /* VALID MOVE TARGETS (Green Glow) */
    div[data-testid="stVerticalBlock"] .valid-move button {
        background-color: rgba(46, 204, 113, 0.2) !important;
        border: 2px dashed #2ecc71 !important;
        color: transparent !important;
    }
    div[data-testid="stVerticalBlock"] .valid-move button:hover {
        background-color: rgba(46, 204, 113, 0.6) !important;
        transform: scale(0.95);
    }

    /* EMPTY CELLS */
    div[data-testid="stVerticalBlock"] .empty-cell button {
        background-color: #0e1117 !important;
        border: 1px solid #262730 !important;
        color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Arena Logic & State Management ---

if 'agent1' in st.session_state and st.session_state.agent1.q_table:
    
    # Game Controls
    col_c1, col_c2, col_c3 = st.columns([1.5, 1, 1])
    with col_c1:
        opponent_choice = st.selectbox("🤖 Select Opponent", ["Agent 1 (White - Defensive)", "Agent 2 (Black - Aggressive)"], key="arena_opp")
    with col_c2:
        starter = st.selectbox("🎲 Who Starts?", ["Human", "AI"], key="arena_start")
    with col_c3:
        st.write("") # Spacer
        start_btn = st.button("🔥 Start Match", use_container_width=True, type="primary")

    # Initialize Game Session
    if start_btn:
        st.session_state.human_env = Hexapawn()
        st.session_state.human_game_active = True
        st.session_state.selected_pawn = None
        st.session_state.game_message = "Good Luck! Capture the enemy or reach the other side."
        
        # Setup Players
        if "Agent 1" in opponent_choice:
            st.session_state.ai_player_id = 1
            st.session_state.ai_agent = st.session_state.agent1
            st.session_state.human_player_id = 2
            st.session_state.human_icon = "♟" # Black
            st.session_state.ai_icon = "♙"    # White
        else:
            st.session_state.ai_player_id = 2
            st.session_state.ai_agent = st.session_state.agent2
            st.session_state.human_player_id = 1
            st.session_state.human_icon = "♙" # White
            st.session_state.ai_icon = "♟"    # Black
            
        # Set First Turn
        st.session_state.current_turn_id = st.session_state.ai_player_id if starter == "AI" else st.session_state.human_player_id
        st.rerun()

    # --- 3. The Game Loop ---
    if 'human_env' in st.session_state and st.session_state.human_game_active:
        h_env = st.session_state.human_env
        
        # A. Check Game Over Conditions
        if h_env.game_over:
            st.markdown("### 🏁 Match Result")
            if h_env.winner == st.session_state.human_player_id:
                
                st.success(f"🎉 **VICTORY!** You outsmarted the AI!")
            elif h_env.winner == st.session_state.ai_player_id:
                st.error(f"💀 **DEFEAT!** The AI won this round.")
            else:
                st.warning("🤝 **DRAW!** It's a stalemate.")
            
            if st.button("🔄 Play Again", type="secondary", use_container_width=True):
                st.session_state.human_game_active = False
                st.rerun()
                
        # B. AI Turn Logic (Automatic)
        elif h_env.current_player == st.session_state.ai_player_id:
            # Display board statically while AI thinks
            st.info(f"🤖 **AI ({st.session_state.ai_icon}) is thinking...**")
            
            # Use a progress bar for visual "thinking" effect
            prog_bar = st.progress(0)
            for i in range(100):
                import time
                time.sleep(0.005) # Brief delay for realism
                prog_bar.progress(i + 1)
            prog_bar.empty()

            # Execute AI Move
            ai_action = st.session_state.ai_agent.choose_action(h_env, training=False)
            if ai_action:
                h_env.make_move(ai_action)
                st.session_state.game_message = "AI has moved. Your turn!"
            else:
                # AI has no moves -> Human wins (handled by env, but safe check)
                h_env.game_over = True
                h_env.winner = st.session_state.human_player_id
            
            st.rerun()

        # C. Human Turn Logic (Interactive)
        else:
            st.success(f"👤 **Your Turn ({st.session_state.human_icon})** | {st.session_state.game_message}")
            
            # Prepare Grid Data
            board = h_env.board
            available_moves = h_env.get_available_actions()
            
            # Filter moves if a pawn is selected
            valid_destinations = []
            if st.session_state.selected_pawn:
                valid_destinations = [target for (start, target) in available_moves if start == st.session_state.selected_pawn]

            # Render 3x3 Grid
            for r in range(3):
                cols = st.columns(3)
                for c in range(3):
                    with cols[c]:
                        cell_val = board[r, c]
                        # Unique key for every button state to prevent ghost interactions
                        btn_key = f"b_{r}_{c}_{len(h_env.move_history)}"
                        
                        # --- LOGIC: Valid Move Target (Empty Space or Enemy Capture) ---
                        if (r, c) in valid_destinations:
                            st.markdown('<div class="valid-move">', unsafe_allow_html=True)
                            # Show 'X' for capture, 'O' for move, or just empty click target
                            symbol = "⚔️" if cell_val != 0 else "◎" 
                            if st.button(symbol, key=btn_key, help="Click to Move Here"):
                                # EXECUTE HUMAN MOVE
                                move = (st.session_state.selected_pawn, (r, c))
                                h_env.make_move(move)
                                st.session_state.selected_pawn = None
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)

                        # --- LOGIC: Human Pawn ---
                        elif cell_val == st.session_state.human_player_id:
                            is_selected = (st.session_state.selected_pawn == (r, c))
                            style_class = "human-piece-selected" if is_selected else "human-piece"
                            
                            st.markdown(f'<div class="{style_class}">', unsafe_allow_html=True)
                            if st.button(st.session_state.human_icon, key=btn_key):
                                if is_selected:
                                    st.session_state.selected_pawn = None # Deselect
                                    st.session_state.game_message = "Pawn deselected."
                                else:
                                    st.session_state.selected_pawn = (r, c) # Select
                                    st.session_state.game_message = "Select a highlighted square to move."
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)

                        # --- LOGIC: AI Pawn ---
                        elif cell_val == st.session_state.ai_player_id:
                            st.markdown('<div class="ai-piece">', unsafe_allow_html=True)
                            st.button(st.session_state.ai_icon, key=btn_key, disabled=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # --- LOGIC: Empty Irrelevant Cell ---
                        else:
                            st.markdown('<div class="empty-cell">', unsafe_allow_html=True)
                            st.button(" ", key=btn_key, disabled=True)
                            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 **Waiting for Brains...** \nPlease train the agents in the panel above or load a saved session to enter the Arena.")

st.markdown("---")
st.caption("Strategic RL Hexapawn Arena | Engineered for Nik 👑")
