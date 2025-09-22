"""
Blink-controlled jumping ball game using pygame.
The ball automatically moves forward and jumps over obstacles when blinks are detected.
"""

import pygame
import random
import math
import time
from eeg_simulator import EEGSimulator
from blink_detector import RealTimeBlinkDetector

# Initialize Pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)

class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 20
        self.velocity_y = 0
        self.gravity = 0.8
        self.jump_strength = -15
        self.ground_y = SCREEN_HEIGHT - 100
        self.is_jumping = False
        self.color = BLUE
        
    def jump(self):
        """Make the ball jump if it's on the ground."""
        if not self.is_jumping:
            self.velocity_y = self.jump_strength
            self.is_jumping = True
    
    def update(self):
        """Update ball physics."""
        # Apply gravity
        self.velocity_y += self.gravity
        self.y += self.velocity_y
        
        # Check ground collision
        if self.y >= self.ground_y:
            self.y = self.ground_y
            self.velocity_y = 0
            self.is_jumping = False
    
    def draw(self, screen):
        """Draw the ball."""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        # Draw a simple face
        eye_offset = 6
        pygame.draw.circle(screen, WHITE, (int(self.x - eye_offset), int(self.y - 5)), 3)
        pygame.draw.circle(screen, WHITE, (int(self.x + eye_offset), int(self.y - 5)), 3)
        pygame.draw.circle(screen, BLACK, (int(self.x - eye_offset), int(self.y - 5)), 1)
        pygame.draw.circle(screen, BLACK, (int(self.x + eye_offset), int(self.y - 5)), 1)

class Obstacle:
    def __init__(self, x):
        self.x = x
        self.width = 30
        self.height = 60
        self.y = SCREEN_HEIGHT - 100 - self.height
        self.speed = 5
        self.color = RED
        
    def update(self):
        """Update obstacle position."""
        self.x -= self.speed
    
    def draw(self, screen):
        """Draw the obstacle."""
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        # Add some detail
        pygame.draw.rect(screen, DARK_GRAY, (self.x + 5, self.y + 5, self.width - 10, self.height - 10))
    
    def is_off_screen(self):
        """Check if obstacle is off the left side of screen."""
        return self.x + self.width < 0
    
    def collides_with(self, ball):
        """Check collision with ball."""
        ball_left = ball.x - ball.radius
        ball_right = ball.x + ball.radius
        ball_top = ball.y - ball.radius
        ball_bottom = ball.y + ball.radius
        
        obstacle_left = self.x
        obstacle_right = self.x + self.width
        obstacle_top = self.y
        obstacle_bottom = self.y + self.height
        
        return (ball_right > obstacle_left and 
                ball_left < obstacle_right and 
                ball_bottom > obstacle_top and 
                ball_top < obstacle_bottom)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("EEG Blink-Controlled Ball Game")
        self.clock = pygame.time.Clock()
        
        # Game objects
        self.ball = Ball(100, SCREEN_HEIGHT - 100)
        self.obstacles = []
        self.obstacle_spawn_timer = 0
        self.obstacle_spawn_interval = 120  # frames
        
        # Game state
        self.score = 0
        self.game_over = False
        self.paused = False
        
        # EEG and blink detection
        self.eeg_simulator = EEGSimulator(sampling_rate=100)
        self.blink_detector = RealTimeBlinkDetector(self.eeg_simulator)
        self.blink_detector.add_blink_callback(self.on_blink_detected)
        
        # UI elements
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Blink visualization
        self.blink_flash_timer = 0
        self.last_blink_confidence = 0
        
        # EEG data visualization
        self.eeg_data_points = []
        self.max_eeg_points = 200
        
    def on_blink_detected(self, confidence):
        """Callback when a blink is detected."""
        self.ball.jump()
        self.blink_flash_timer = 30  # Flash for 30 frames
        self.last_blink_confidence = confidence
        print(f"Blink detected! Confidence: {confidence:.3f} - Ball jumped!")
    
    def spawn_obstacle(self):
        """Spawn a new obstacle."""
        self.obstacles.append(Obstacle(SCREEN_WIDTH))
    
    def update_obstacles(self):
        """Update all obstacles."""
        # Update existing obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update()
            
            # Remove obstacles that are off screen
            if obstacle.is_off_screen():
                self.obstacles.remove(obstacle)
                self.score += 10
            
            # Check collision with ball
            elif obstacle.collides_with(self.ball):
                self.game_over = True
        
        # Spawn new obstacles
        self.obstacle_spawn_timer += 1
        if self.obstacle_spawn_timer >= self.obstacle_spawn_interval:
            self.spawn_obstacle()
            self.obstacle_spawn_timer = 0
            # Gradually increase difficulty
            if self.obstacle_spawn_interval > 60:
                self.obstacle_spawn_interval -= 1
    
    def update_eeg_visualization(self):
        """Update EEG data for visualization."""
        # Get recent EEG data
        timestamps, values = self.eeg_simulator.get_buffer_data(self.max_eeg_points)
        
        if len(values) > 0:
            # Normalize values for display
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized_values = [0.5] * len(values)
            
            self.eeg_data_points = normalized_values
    
    def draw_eeg_visualization(self):
        """Draw EEG signal visualization."""
        if len(self.eeg_data_points) < 2:
            return
        
        # EEG display area
        eeg_x = 10
        eeg_y = 10
        eeg_width = 300
        eeg_height = 100
        
        # Background
        pygame.draw.rect(self.screen, BLACK, (eeg_x, eeg_y, eeg_width, eeg_height))
        pygame.draw.rect(self.screen, WHITE, (eeg_x, eeg_y, eeg_width, eeg_height), 2)
        
        # Draw signal
        points = []
        for i, value in enumerate(self.eeg_data_points):
            x = eeg_x + (i / len(self.eeg_data_points)) * eeg_width
            y = eeg_y + eeg_height - (value * eeg_height)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, GREEN, False, points, 2)
        
        # Label
        label = self.small_font.render("EEG Signal", True, WHITE)
        self.screen.blit(label, (eeg_x, eeg_y - 25))
    
    def draw_ui(self):
        """Draw user interface elements."""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (SCREEN_WIDTH - 200, 20))
        
        # Blink detection status
        stats = self.blink_detector.get_statistics()
        blink_text = self.small_font.render(f"Blinks: {stats['blink_detections']}", True, BLACK)
        self.screen.blit(blink_text, (SCREEN_WIDTH - 200, 60))
        
        # Confidence display
        if self.last_blink_confidence > 0:
            conf_text = self.small_font.render(f"Last Confidence: {self.last_blink_confidence:.3f}", True, BLACK)
            self.screen.blit(conf_text, (SCREEN_WIDTH - 200, 80))
        
        # Instructions
        instructions = [
            "Blink to make the ball jump!",
            "Avoid the red obstacles",
            "Press SPACE to manually jump",
            "Press P to pause, Q to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, BLACK)
            self.screen.blit(text, (10, SCREEN_HEIGHT - 100 + i * 20))
        
        # Blink flash effect
        if self.blink_flash_timer > 0:
            flash_alpha = int((self.blink_flash_timer / 30) * 100)
            flash_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            flash_surface.set_alpha(flash_alpha)
            flash_surface.fill(YELLOW)
            self.screen.blit(flash_surface, (0, 0))
            self.blink_flash_timer -= 1
        
        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font.render("GAME OVER", True, RED)
            final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
            restart_text = self.small_font.render("Press R to restart or Q to quit", True, WHITE)
            
            self.screen.blit(game_over_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 50))
            self.screen.blit(final_score_text, (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2))
            self.screen.blit(restart_text, (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 + 50))
        
        # Pause screen
        if self.paused and not self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            pause_text = self.font.render("PAUSED", True, WHITE)
            self.screen.blit(pause_text, (SCREEN_WIDTH//2 - 60, SCREEN_HEIGHT//2))
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.ball = Ball(100, SCREEN_HEIGHT - 100)
        self.obstacles = []
        self.obstacle_spawn_timer = 0
        self.obstacle_spawn_interval = 120
        self.score = 0
        self.game_over = False
        self.blink_flash_timer = 0
        self.last_blink_confidence = 0
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE and not self.game_over:
                    self.ball.jump()
                elif event.key == pygame.K_p and not self.game_over:
                    self.paused = not self.paused
                elif event.key == pygame.K_r and self.game_over:
                    self.reset_game()
                elif event.key == pygame.K_b:  # Manual blink trigger for testing
                    self.eeg_simulator.trigger_blink()
        
        return True
    
    def run(self):
        """Main game loop."""
        print("Starting EEG Blink-Controlled Ball Game!")
        print("Instructions:")
        print("- Blink to make the ball jump over obstacles")
        print("- Press SPACE for manual jump")
        print("- Press B to trigger a test blink")
        print("- Press P to pause, R to restart, Q to quit")
        
        # Start EEG simulation and blink detection
        self.blink_detector.start()
        
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            
            if not self.paused and not self.game_over:
                # Update blink detector
                self.blink_detector.update()
                
                # Update game objects
                self.ball.update()
                self.update_obstacles()
                
                # Update EEG visualization
                self.update_eeg_visualization()
            
            # Draw everything
            self.screen.fill(WHITE)
            
            # Draw ground
            pygame.draw.rect(self.screen, GRAY, (0, SCREEN_HEIGHT - 100, SCREEN_WIDTH, 100))
            
            # Draw game objects
            self.ball.draw(self.screen)
            for obstacle in self.obstacles:
                obstacle.draw(self.screen)
            
            # Draw UI
            self.draw_eeg_visualization()
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Cleanup
        self.blink_detector.stop()
        self.eeg_simulator.stop()
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
