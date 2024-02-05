import pygame
from defs import *
import random
from nnet import Nnet
import numpy as np

class Star():
    
    def __init__(self, gameDisplay):
        self.gameDisplay = gameDisplay
        self.state = STAR_ALIVE
        self.img = pygame.image.load(STAR_FILENAME)
        self.rect = self.img.get_rect()
        self.speed = 0
        self.fitness = 0
        self.time_lived = 0
        self.nnet = Nnet(NNET_INPUTS, NNET_HIDDEN, NNET_OUTPUTS)
        self.set_position(STAR_START_X, STAR_START_Y)
        
    def reset(self):
        self.state = STAR_ALIVE
        self.speed = 0
        self.fitness = 0
        self.time_lived = 0
        self.set_position(STAR_START_X, STAR_START_Y)
        
    def set_position(self, x, y):
        self.rect.centerx = x
        self.rect.centery = y
        
    def move(self, dt):
        distance = 0
        new_speed = 0
        
        distance = (self.speed * dt) + (0.5 * GRAVITY * dt * dt)
        new_speed = self.speed + (GRAVITY * dt)
        
        self.rect.centery += distance
        self.speed = new_speed
        
        if self.rect.top < 0:
            self.rect.top = 0
            self.speed = 0
            
    def jump(self, pipes):
        inputs = self.get_inputs(pipes)
        val = self.nnet.get_max_value(inputs)
        if val > JUMP_CHANCE:
            self.speed = STAR_START_SPEED
        
    def draw(self):
        self.gameDisplay.blit(self.img, self.rect)
        
    def check_status(self, pipes):
        if self.rect.bottom > DISPLAY_H:
            self.state = STAR_DEAD
        else:
            self.check_hits(pipes)
            
    def assign_collision_fitness(self, p):
        gap_y = 0
        if p.pipe_type == PIPE_UPPER:
            gap_y = p.rect.bottom + PIPE_GAP_SIZE / 2
        else:
            gap_y = p.rect.top - PIPE_GAP_SIZE / 2
            
        self.fitness = -(abs(self.rect.centery - gap_y))
            
    def check_hits(self, pipes):
        for p in pipes:
            if p.rect.colliderect(self.rect):
                self.state = STAR_DEAD
                break
            
    def update(self, dt, pipes):
        if self.state == STAR_ALIVE:
            self.time_lived += dt
            self.move(dt)
            self.jump(pipes)
            self.draw()
            self.check_status(pipes)
            
    def get_inputs(self, pipes):
        closest = DISPLAY_W * 2
        bottom_y = 0
        for p in pipes:
            if p.pipe_type == PIPE_UPPER and p.rect.right < closest and p.rect.right > self.rect.left:
                closest = p.rect.right
                bottom_y = p.rect.bottom
                
        horizontal_distance = closest - self.rect.centerx
        vertical_distance = (self.rect.centery) - (bottom_y + PIPE_GAP_SIZE / 2)
        
        inputs = [
            ((horizontal_distance / DISPLAY_W) * 0.99) + 0.01,
            (((vertical_distance + Y_SHIFT) / NORMALIZER) * 0.99) + 0.01
        ]
        return inputs
    
    def create_offspring(p1, p2, gameDisplay):
        new_star = Star(gameDisplay)
        new_star.nnet.create_mixed_weights(p1.nnet, p2.nnet)
        return new_star
            
            
class StarCollection:
    
    def __init__(self, gameDisplay):
        self.gameDisplay = gameDisplay
        self.stars = []
        self.create_new_generation()
        
    def create_new_generation(self):
        self.stars = []
        for i in range(0, GENERATION_SIZE):
            self.stars.append(Star(self.gameDisplay))
            
    def update(self, dt, pipes):
        num_alive = 0
        for s in self.stars:
            s.update(dt, pipes)
            if s.state == STAR_ALIVE:
                num_alive += 1
        return num_alive

    def evolve_population(self):
        for s in self.stars:
            s.fitness += s.time_lived * PIPE_SPEED
        
        self.stars.sort(key=lambda x: x.fitness, reverse=True)
        
        cut_off = int(len(self.stars) * MUTATION_CUT_OFF)
        good_stars = self.stars[0:cut_off]
        bad_stars = self.stars[cut_off:]
        num_bad_to_take = int(len(self.stars) * MUTATION_BAD_TO_KEEP)
        
        for s in bad_stars:
            s.nnet.modify_weights()
            
        new_stars = []
        
        idx_bad_to_take = np.random.choice(np.arange(len(bad_stars)), num_bad_to_take, replace=False)
        
        for index in idx_bad_to_take:
            new_stars.append(bad_stars[index])
            
        new_stars.extend(good_stars)
        
        children_needed = len(self.stars) - len(new_stars)
        
        while len(new_stars) < len(self.stars):
            idx_to_breed = np.random.choice(np.arange(len(good_stars)), 2, replace=False)
            if idx_to_breed[0] != idx_to_breed[1]:
                new_star = Star.create_offspring(good_stars[idx_to_breed[0]], good_stars[idx_to_breed[1]], self.gameDisplay)
                if random.random() < MUTATION_MODIFY_CHANCE_LIMIT:
                    new_star.nnet.modify_weights()
                new_stars.append(new_star)
                
        for s in new_stars:
            s.reset()
            
        self.stars = new_stars