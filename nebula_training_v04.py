#!/usr/bin/env python3
"""
NEBULA v0.4 TRAINING SYSTEM
Equipo NEBULA: Francisco Angulo de Lafuente y Ángel

SISTEMA DE ENTRENAMIENTO COMPLETO PARA NEBULA v0.4
- Training loop optimizado para RTX GPUs con mixed precision
- Dataset generator de sudokus realistas validado
- Early stopping con validation metrics
- Checkpoint saving y model persistence
- Comprehensive logging y monitoring
- Constraint-aware training schedule

PASO A PASO: Entrenamiento riguroso según nuestros criterios
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import math
import time
import json
import os
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import random

# Import our unified model y dataset functions
from NEBULA_UNIFIED_v04 import NEBULA_HRM_Sudoku_v04

@dataclass
class TrainingConfig:
    """Configuration para training setup"""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    constraint_weight_start: float = 2.0
    constraint_weight_end: float = 5.0
    distillation_weight: float = 0.3
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_every: int = 5
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
class NEBULASudokuDataset:
    """
    Dataset generator para sudokus usando backtracking validado
    Basado en nuestro generador probado que produce sudokus válidos
    """
    
    def __init__(self, num_samples: int, mask_rate: float = 0.65, device: str = 'cuda'):
        self.num_samples = num_samples
        self.mask_rate = mask_rate
        self.device = device
        
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of sudoku input-target pairs"""
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            # Generate complete sudoku using our validated backtracking
            full_sudoku = self.generate_full_sudoku()
            
            # Create masked version for input
            input_sudoku = self.mask_sudoku(full_sudoku, self.mask_rate)
            
            inputs.append(torch.tensor(input_sudoku, dtype=torch.long))
            targets.append(torch.tensor(full_sudoku, dtype=torch.long))
            
        return torch.stack(inputs).to(self.device), torch.stack(targets).to(self.device)
    
    def generate_full_sudoku(self, seed: Optional[int] = None) -> List[List[int]]:
        """Generate complete valid sudoku using backtracking"""
        if seed is not None:
            random.seed(seed)
            
        digits = list(range(1, 10))
        grid = [[0]*9 for _ in range(9)]
        
        # Randomized cell order para variability
        cells = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(cells)
        
        def is_valid(grid, r, c, val):
            # Check row
            for j in range(9):
                if grid[r][j] == val:
                    return False
            # Check column
            for i in range(9):
                if grid[i][c] == val:
                    return False
            # Check 3x3 box
            br, bc = (r // 3) * 3, (c // 3) * 3
            for i in range(br, br+3):
                for j in range(bc, bc+3):
                    if grid[i][j] == val:
                        return False
            return True
            
        def backtrack(idx=0):
            if idx >= 81:
                return True
            i, j = cells[idx]
            choices = digits[:]
            random.shuffle(choices)
            for val in choices:
                if is_valid(grid, i, j, val):
                    grid[i][j] = val
                    if backtrack(idx + 1):
                        return True
                    grid[i][j] = 0
            return False
            
        success = backtrack(0)
        if not success:
            # Fallback: try with ordered cells
            grid = [[0]*9 for _ in range(9)]
            cells = [(i, j) for i in range(9) for j in range(9)]
            success = backtrack(0)
            
        if not success:
            raise RuntimeError("Failed to generate valid sudoku")
            
        return grid
    
    def mask_sudoku(self, full_grid: List[List[int]], mask_rate: float) -> List[List[int]]:
        """Create masked sudoku for training input"""
        masked = [row[:] for row in full_grid]  # Deep copy
        
        # Calculate cells to keep
        total_cells = 81
        cells_to_keep = int(total_cells * (1.0 - mask_rate))
        
        # Get all positions
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)
        
        # Mask cells (set to 0) except for cells_to_keep
        for i, (r, c) in enumerate(positions):
            if i >= cells_to_keep:
                masked[r][c] = 0
                
        return masked

class NEBULATrainer:
    """
    NEBULA v0.4 Training System
    
    Comprehensive training system con:
    - Mixed precision training optimizado para RTX
    - Constraint-aware loss scheduling
    - Advanced optimization strategies
    - Comprehensive validation y monitoring
    """
    
    def __init__(self, config: TrainingConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        print(f"[NEBULA TRAINER] Inicializando sistema de entrenamiento:")
        print(f"  - Device: {device}")
        print(f"  - Epochs: {config.epochs}")
        print(f"  - Batch size: {config.batch_size}")
        print(f"  - Learning rate: {config.learning_rate}")
        print(f"  - Mixed precision: {config.mixed_precision}")
        
        # Initialize model
        self.model = NEBULA_HRM_Sudoku_v04(
            grid_size=9,
            device=device,
            use_rtx_optimization=True,
            use_mixed_precision=config.mixed_precision
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Mixed precision scaler if available
        if config.mixed_precision and hasattr(torch.cuda.amp, 'GradScaler'):
            try:
                # Try new API first
                from torch.amp import GradScaler
                self.scaler = GradScaler('cuda')
                print(f"  - Mixed precision: Enabled (new API)")
            except ImportError:
                # Fallback to old API
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                print(f"  - Mixed precision: Enabled (legacy API)")
        else:
            self.scaler = None
            print(f"  - Mixed precision: Disabled")
            
        # Training state
        self.current_epoch = 0
        self.best_validation_loss = float('inf')
        self.best_model_state = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'constraint_violations': [],
            'learning_rate': []
        }
        self.patience_counter = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = "nebula_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def compute_constraint_schedule(self, epoch: int) -> float:
        """Compute constraint weight scheduling"""
        progress = epoch / self.config.epochs
        weight = self.config.constraint_weight_start + (
            self.config.constraint_weight_end - self.config.constraint_weight_start
        ) * progress
        return weight
        
    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor, 
                        input_mask: torch.Tensor) -> float:
        """Compute accuracy solo en celdas que necesitan predicción"""
        predictions = torch.argmax(logits, dim=-1)
        
        # Mask: solo evaluar celdas donde input era 0 (vacías)
        eval_mask = (input_mask == 0) & (targets > 0)
        
        if eval_mask.sum() == 0:
            return 0.0
            
        correct = (predictions == targets) & eval_mask
        accuracy = correct.sum().item() / eval_mask.sum().item()
        return accuracy
        
    def train_epoch(self, dataset: NEBULASudokuDataset) -> Dict[str, float]:
        """Train single epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_ce_loss = 0.0
        epoch_constraint_loss = 0.0
        epoch_distillation_loss = 0.0
        num_batches = 0
        
        # Dynamic constraint weight
        constraint_weight = self.compute_constraint_schedule(self.current_epoch)
        
        # Training loop
        steps_per_epoch = max(1, dataset.num_samples // self.config.batch_size)
        
        for step in range(steps_per_epoch):
            # Generate fresh batch
            inputs, targets = dataset.generate_batch(self.config.batch_size)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss_dict = self.model.compute_loss(
                        outputs, targets,
                        constraint_weight=constraint_weight,
                        distillation_weight=self.config.distillation_weight
                    )
                    total_loss = loss_dict['total_loss']
                
                # Scaled backward pass
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Standard precision training
                outputs = self.model(inputs)
                loss_dict = self.model.compute_loss(
                    outputs, targets,
                    constraint_weight=constraint_weight,
                    distillation_weight=self.config.distillation_weight
                )
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Accumulate metrics
            with torch.no_grad():
                accuracy = self.compute_accuracy(outputs['logits'], targets, inputs)
                
            epoch_loss += total_loss.item()
            epoch_accuracy += accuracy
            epoch_ce_loss += loss_dict['ce_loss'].item()
            epoch_constraint_loss += loss_dict['constraint_loss'].item()
            epoch_distillation_loss += loss_dict['distillation_loss'].item()
            num_batches += 1
            
            # Progress logging
            if (step + 1) % max(1, steps_per_epoch // 10) == 0:
                print(f"  Step {step+1}/{steps_per_epoch}: Loss={total_loss.item():.4f}, Acc={accuracy:.4f}")
        
        # Average metrics
        return {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_accuracy / num_batches,
            'ce_loss': epoch_ce_loss / num_batches,
            'constraint_loss': epoch_constraint_loss / num_batches,
            'distillation_loss': epoch_distillation_loss / num_batches,
            'constraint_weight': constraint_weight
        }
    
    def validate_epoch(self, dataset: NEBULASudokuDataset) -> Dict[str, float]:
        """Validation epoch"""
        self.model.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        val_constraint_violations = 0.0
        num_batches = 0
        
        # Validation batches
        val_steps = max(1, (dataset.num_samples * self.config.validation_split) // self.config.batch_size)
        
        with torch.no_grad():
            for step in range(val_steps):
                inputs, targets = dataset.generate_batch(self.config.batch_size)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss_dict = self.model.compute_loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss_dict = self.model.compute_loss(outputs, targets)
                
                accuracy = self.compute_accuracy(outputs['logits'], targets, inputs)
                
                val_loss += loss_dict['total_loss'].item()
                val_accuracy += accuracy
                val_constraint_violations += outputs['constraint_violations'].sum().item()
                num_batches += 1
        
        return {
            'loss': val_loss / num_batches,
            'accuracy': val_accuracy / num_batches,
            'constraint_violations': val_constraint_violations / num_batches
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'best_validation_loss': self.best_validation_loss
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"nebula_v04_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "nebula_v04_best.pt")
            torch.save(checkpoint, best_path)
            print(f"  Best model saved at epoch {epoch}")
    
    def train(self, num_training_samples: int = 10000) -> Dict[str, List]:
        """
        TRAINING LOOP PRINCIPAL
        
        Training completo con early stopping y validation
        """
        print(f"\n{'='*80}")
        print(f"NEBULA v0.4 TRAINING INICIADO")
        print(f"{'='*80}")
        print(f"Training samples: {num_training_samples}")
        print(f"Validation split: {self.config.validation_split}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        # Create datasets
        train_dataset = NEBULASudokuDataset(
            num_samples=int(num_training_samples * (1 - self.config.validation_split)),
            mask_rate=0.65,
            device=self.device
        )
        
        val_dataset = NEBULASudokuDataset(
            num_samples=int(num_training_samples * self.config.validation_split),
            mask_rate=0.65,
            device=self.device
        )
        
        print(f"Train dataset: {train_dataset.num_samples} samples")
        print(f"Val dataset: {val_dataset.num_samples} samples")
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(train_dataset)
            
            # Validation
            val_metrics = self.validate_epoch(val_dataset)
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Record metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['constraint_violations'].append(val_metrics['constraint_violations'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Timing
            epoch_time = time.time() - epoch_start_time
            
            # Logging
            print(f"Train Loss: {train_metrics['loss']:.6f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.6f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Constraint Violations: {val_metrics['constraint_violations']:.2f}")
            print(f"Constraint Weight: {train_metrics['constraint_weight']:.2f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Epoch Time: {epoch_time:.1f}s")
            
            # Early stopping check
            is_best = val_metrics['loss'] < self.best_validation_loss
            if is_best:
                self.best_validation_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch + 1, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={self.config.early_stopping_patience})")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model (val_loss={self.best_validation_loss:.6f})")
        
        # Final save
        self.save_checkpoint(self.current_epoch + 1, True)
        
        print(f"\n{'='*80}")
        print(f"NEBULA v0.4 TRAINING COMPLETADO")
        print(f"{'='*80}")
        print(f"Best validation loss: {self.best_validation_loss:.6f}")
        print(f"Total training time: {sum(self.training_history.get('epoch_times', [0])):.1f}s")
        
        return self.training_history

def main():
    """Main training execution"""
    print("NEBULA v0.4 TRAINING SYSTEM")
    print("Equipo NEBULA: Francisco Angulo de Lafuente y Ángel")
    print("Paso a paso, sin prisa, con calma")
    
    # Training configuration
    config = TrainingConfig(
        epochs=30,  # Reasonable para initial training
        batch_size=16,  # Balanced para RTX 3090
        learning_rate=1e-3,
        constraint_weight_start=1.0,
        constraint_weight_end=3.0,
        distillation_weight=0.2,
        early_stopping_patience=8,
        mixed_precision=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Initialize trainer
        trainer = NEBULATrainer(config, device)
        
        # Start training
        training_history = trainer.train(num_training_samples=5000)  # Initial training
        
        # Save training history
        with open('nebula_v04_training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
            
        print("\nTRAINING SUCCESSFUL")
        print("Model ready para benchmark testing")
        
    except Exception as e:
        print(f"\nTRAINING ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("NEBULA v0.4 trained successfully - Ready para benchmarking!")
    else:
        print("Training failed - Debug required")