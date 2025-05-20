import random
import numpy as np
import json
from tetris_base import *  


random.seed(42) 
np.random.seed(42)

# some helper functions for extracting features
def get_heights(board):
    heights = [0] * BOARDWIDTH
    for x in range(BOARDWIDTH):
        for y in range(BOARDHEIGHT):
            if board[x][y] != BLANK:
                heights[x] = BOARDHEIGHT - y
                break
    return heights

def count_complete_lines(board):
    count = 0
    for y in range(BOARDHEIGHT):
        if all(board[x][y] != BLANK for x in range(BOARDWIDTH)):
            count += 1
    return count

def count_holes(board, heights):
    holes = 0
    for x in range(BOARDWIDTH):
        block_found = False
        for y in range(BOARDHEIGHT):
            if board[x][y] != BLANK:
                block_found = True
            elif block_found and board[x][y] == BLANK:
                holes += 1
    return holes

def calc_bumpiness(heights):
    bumpiness = 0
    for i in range(BOARDWIDTH - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness

def extract_features(board):
    heights = get_heights(board)  #  heights of each column
    aggregate_height = sum(heights) #  aggregate height 
    lines_cleared = count_complete_lines(board) #  complete lines 
    holes = count_holes(board, heights)          #  holes 
    bumpiness = calc_bumpiness(heights)           #  bumpiness  
    valley_depth = max(heights) if heights else 0 - min(heights) if heights else 0  # valley depth
    return [aggregate_height, lines_cleared, holes, bumpiness, valley_depth]


#  Rating Function 
def rate_board(features, weights):
    return sum(f * w for f, w in zip(features, weights))  # multiply features with weights and sum 

# Parameters 
POPULATION_SIZE = 15 
GENERATIONS = 15      
MUTATION_RATE = 0.15

# random weights (chromosomes) , each chromosome is a list of weights for 5 features ranging from -5 to 5
def init_population():
    return [np.random.uniform(-5, 5, 5).tolist() for _ in range(POPULATION_SIZE)]

# Evaluate current move and next piece 
def evaluate_move(board, current_piece, next_piece, weights, x, rotation):
    #temporary piece for simulation
    temp_piece = current_piece.copy()
    temp_piece['rotation'] = rotation
    temp_piece['x'] = x  
    temp_piece['y'] = 0
    
    
    if not is_valid_position(board, temp_piece):
        return -float('inf')
    
    # dropping the piece
    while is_valid_position(board, temp_piece, adj_Y=1):
        temp_piece['y'] += 1  # move down 
    
    #copy of the board
    temp_board = [col.copy() for col in board]
    
    #Add piece to the board
    add_to_board(temp_board, temp_piece)
    
    # score for clearing lines
    lines_removed = remove_complete_lines(temp_board)
    lines_score = 0
    if lines_removed == 1:
        lines_score = 40
    elif lines_removed == 2:
        lines_score = 120
    elif lines_removed == 3:
        lines_score = 300
    elif lines_removed == 4:
        lines_score = 1200
    
    # Extract features from the resulting board
    features = extract_features(temp_board)
    
    # consider the next piece
    best_next_score = -float('inf')
    
    # Evaluate next piece on resulting board
    for next_rotation in range(len(PIECES[next_piece['shape']])):
        for next_x in range(-2, BOARDWIDTH):
            next_temp_piece = next_piece.copy()
            next_temp_piece['rotation'] = next_rotation
            next_temp_piece['x'] = next_x
            next_temp_piece['y'] = 0
            
            if not is_valid_position(temp_board, next_temp_piece):
                continue
            
            # dropping the next piece
            while is_valid_position(temp_board, next_temp_piece, adj_Y=1):
                next_temp_piece['y'] += 1
            
            # copy of the temp board
            next_temp_board = [col.copy() for col in temp_board]
            
            #Add next piece to the board
            add_to_board(next_temp_board, next_temp_piece)
            
            # Extract features from the resulting board with next piece
            next_features = extract_features(next_temp_board)
            next_score = rate_board(next_features, weights)
            
            best_next_score = max(best_next_score, next_score)
    
    #couldn't place the next piece, use a default value
    if best_next_score == -float('inf'):
        best_next_score = 0
    
    # Combine current features score with look-ahead score and line clearing score
    current_score = rate_board(features, weights)
    final_score = current_score + (best_next_score * 0.5) + lines_score  
    
    return final_score

# Play game using specific weights
def evaluate_fitness(weights, iterations=300):
    board = get_blank_board()
    score = 0
    pieces_placed = 0
    
    current_piece = get_new_piece()
    next_piece = get_new_piece()

    while pieces_placed < iterations:
        best_score = -float('inf')
        best_move = None
        
        # Evaluate all possible moves for the current piece
        for rotation in range(len(PIECES[current_piece['shape']])):
            for x in range(-2, BOARDWIDTH):
                move_score = evaluate_move(board, current_piece, next_piece, weights, x, rotation)
                
                if move_score > best_score:
                    best_score = move_score
                    best_move = (x, rotation)
        
        # If no valid move was found, game over
        if best_move is None:
            break
        
        # Execute the best move
        best_x, best_rotation = best_move
        current_piece['rotation'] = best_rotation
        current_piece['x'] = best_x
        current_piece['y'] = 0
        
        # Drop the piece
        while is_valid_position(board, current_piece, adj_Y=1):
            current_piece['y'] += 1
        
        # Add piece to the board
        add_to_board(board, current_piece)
        
        # Clear completed lines and update score
        lines_removed = remove_complete_lines(board)
        if lines_removed == 1:
            score += 40
        elif lines_removed == 2:
            score += 120
        elif lines_removed == 3:
            score += 300
        elif lines_removed == 4:
            score += 1200
        
        # Move to next piece
        current_piece = next_piece
        next_piece = get_new_piece()
        pieces_placed += 1
        score += 1  # Base score for each piece placed
    
    return score

# Selection (top 50%)
def select(population, fitnesses):
    sorted_pop = [p for _, p in sorted(zip(fitnesses, population), reverse=True)]
    return sorted_pop[:POPULATION_SIZE // 2]

# Crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child

# Mutation
def mutate(weights):
    return [w + random.uniform(-1, 1) if random.random() < MUTATION_RATE else w for w in weights]

#  Main Genetic Algorithm
def genetic_algorithm():
    population = init_population()
    best_log = []
    best_overall_score = 0
    best_overall_weights = None

    for gen in range(GENERATIONS):
        fitnesses = [evaluate_fitness(chromo) for chromo in population]
        best_idx = np.argmax(fitnesses)
        best_score = fitnesses[best_idx]
        best_weights = population[best_idx]
        
        print(f"Generation {gen + 1}: Best score {best_score} with weights {best_weights}")

        best_log.append({
            'generation': gen + 1,
            'best_score': int(best_score),
            'best_weights': best_weights
        })
        
        # Keep track of the best overall solution
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_weights = best_weights.copy()

        # Selection
        selected = select(population, fitnesses)
        
        # Create new generation
        children = []
        while len(children) < POPULATION_SIZE:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            children.append(child)

        population = children

    # Save best weights from all generations
    with open("best_weights.json", "w") as f:
        json.dump(best_overall_weights, f)

    # Save log
    with open("log.json", "w") as f:
        json.dump(best_log, f)

    print(f"Training done! Best overall score: {best_overall_score}")
    print(f"Best weights saved to best_weights.json: {best_overall_weights}")
    # Feature importance analysis
    feature_names = ["Aggregate Height", "Lines Cleared", "Holes", "Bumpiness", "Valley Depth"]
    print("\nFeature Importance Analysis:")
    for i, (name, weight) in enumerate(zip(feature_names, best_overall_weights)):
        print(f"{name}: {weight:.4f}")
    
    
    
    return best_overall_weights




if __name__ == "__main__":
     genetic_algorithm()

