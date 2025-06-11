"""
Hyperparameter optimization for LLaMA legal domain adaptation using Optuna.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import optuna
from optuna.trial import Trial
import wandb
import torch
from torch.utils.data import DataLoader

from .trainer import train, ModelConfig, DataConfig, TrainingConfig
from .evaluation import EvaluationConfig, LegalEvaluator

logger = logging.getLogger(__name__)

@dataclass
class HyperOptConfig:
    """Configuration for hyperparameter optimization."""
    # Optuna settings
    n_trials: int = 20
    timeout: Optional[int] = None  # in seconds
    study_name: str = "llama_legal_hpo"
    storage: Optional[str] = None  # SQLite URL for persistence
    
    # Search space
    optimize_lr: bool = True
    optimize_batch_size: bool = True
    optimize_warmup: bool = True
    optimize_weight_decay: bool = True
    optimize_generation: bool = True
    
    # Evaluation
    eval_steps: int = 500
    patience: int = 3
    
    # Resource constraints
    max_gpu_memory: float = 0.9  # Maximum GPU memory usage (90%)
    min_batch_size: int = 1
    max_batch_size: int = 32

def suggest_hyperparameters(trial: Trial, config: HyperOptConfig) -> Dict[str, Any]:
    """Suggest hyperparameters for a trial."""
    params = {}
    
    # Learning rate (log scale)
    if config.optimize_lr:
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-6, 1e-4, log=True
        )
    
    # Batch size (powers of 2)
    if config.optimize_batch_size:
        params["per_device_train_batch_size"] = trial.suggest_int(
            "batch_size",
            config.min_batch_size,
            config.max_batch_size,
            log=True
        )
        # Adjust gradient accumulation to maintain effective batch size
        target_effective_batch = 16  # Target effective batch size
        params["gradient_accumulation_steps"] = max(
            1, target_effective_batch // params["per_device_train_batch_size"]
        )
    
    # Warmup ratio
    if config.optimize_warmup:
        params["warmup_ratio"] = trial.suggest_float(
            "warmup_ratio", 0.0, 0.1
        )
    
    # Weight decay
    if config.optimize_weight_decay:
        params["weight_decay"] = trial.suggest_float(
            "weight_decay", 0.0, 0.1
        )
    
    # Generation parameters
    if config.optimize_generation:
        params["temperature"] = trial.suggest_float(
            "temperature", 0.5, 1.0
        )
        params["top_p"] = trial.suggest_float(
            "top_p", 0.5, 1.0
        )
    
    return params

class HyperParameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(
        self,
        base_model_config: ModelConfig,
        base_data_config: DataConfig,
        base_training_config: TrainingConfig,
        eval_config: EvaluationConfig,
        hpo_config: HyperOptConfig,
    ):
        self.base_model_config = base_model_config
        self.base_data_config = base_data_config
        self.base_training_config = base_training_config
        self.eval_config = eval_config
        self.hpo_config = hpo_config
        
        # Initialize Optuna study
        self.study = optuna.create_study(
            study_name=hpo_config.study_name,
            storage=hpo_config.storage,
            direction="minimize",  # Minimize validation perplexity
            load_if_exists=True,
        )
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        self.study.optimize(
            func=self._objective,
            n_trials=self.hpo_config.n_trials,
            timeout=self.hpo_config.timeout,
            callbacks=[self._pruner_callback],
        )
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Best trial achieved perplexity: {best_value}")
        logger.info("Best hyperparameters:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        
        return best_params
    
    def _objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization."""
        # Get hyperparameter suggestions
        params = suggest_hyperparameters(trial, self.hpo_config)
        
        # Update configs with suggested params
        training_config = self._update_training_config(params)
        eval_config = self._update_eval_config(params)
        
        try:
            # Train model with suggested hyperparameters
            model, tokenizer = train(
                model_config=self.base_model_config,
                data_config=self.base_data_config,
                training_config=training_config,
            )
            
            # Evaluate model
            evaluator = LegalEvaluator(
                model=model,
                tokenizer=tokenizer,
                config=eval_config,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            
            metrics = evaluator.evaluate_model(
                eval_dataloader=self._get_eval_dataloader(),
                split="validation",
            )
            
            # Log metrics to W&B
            wandb.log({
                "trial_id": trial.number,
                **metrics,
                **params,
            })
            
            # Return validation perplexity as objective
            return metrics["validation/perplexity"]
            
        except (RuntimeError, ValueError) as e:
            # Handle OOM or other errors
            logger.warning(f"Trial failed with error: {str(e)}")
            raise optuna.exceptions.TrialPruned()
    
    def _pruner_callback(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """Callback to prune unpromising trials."""
        # Get current best value
        best_value = study.best_value
        
        # Prune if significantly worse than best
        if trial.value is not None and trial.value > best_value * 1.5:
            logger.info(f"Pruning trial {trial.number} with value {trial.value}")
            raise optuna.exceptions.TrialPruned()
    
    def _update_training_config(self, params: Dict[str, Any]) -> TrainingConfig:
        """Update training config with suggested parameters."""
        config_dict = vars(self.base_training_config).copy()
        config_dict.update(params)
        return TrainingConfig(**config_dict)
    
    def _update_eval_config(self, params: Dict[str, Any]) -> EvaluationConfig:
        """Update evaluation config with suggested parameters."""
        config_dict = vars(self.eval_config).copy()
        if "temperature" in params:
            config_dict["temperature"] = params["temperature"]
        if "top_p" in params:
            config_dict["top_p"] = params["top_p"]
        return EvaluationConfig(**config_dict)
    
    def _get_eval_dataloader(self) -> DataLoader:
        """Get evaluation dataloader."""
        # TODO: Implement evaluation dataloader creation
        # This should return a DataLoader for the validation set
        pass 