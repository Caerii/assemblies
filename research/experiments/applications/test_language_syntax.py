"""
Language / Syntax Experiment

Tests whether Assembly Calculus can represent syntactic structure,
based on Papadimitriou et al. parser model.

Scientific Questions:
1. Can assemblies represent distinct words?
2. Can words be categorized into syntactic categories?
3. Can MERGE compose categories into phrases?
4. Can hierarchical sentence structure be built?

Reference:
- Papadimitriou, C. H., et al. "Brain Computation by Assemblies of Neurons."
  PNAS 117.25 (2020): 14464-14472.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from research.experiments.base import ExperimentBase, ExperimentResult
import brain as brain_module


@dataclass
class LanguageConfig:
    """Configuration for language experiment."""
    n_neurons: int = 5000
    k_active: int = 50
    p_connect: float = 0.1
    beta: float = 0.1
    projection_rounds: int = 10
    merge_rounds: int = 15


class LanguageSyntaxExperiment(ExperimentBase):
    """Test syntactic structure representation with Assembly Calculus."""
    
    def __init__(self, results_dir: Path = None, seed: int = 42, verbose: bool = True):
        super().__init__(
            name="language_syntax",
            seed=seed,
            results_dir=results_dir or Path(__file__).parent.parent.parent / "results" / "applications",
            verbose=verbose
        )
    
    def build_sentence(
        self,
        config: LanguageConfig,
        sentence: List[str],
        word_categories: Dict[str, str],
        trial_id: int = 0
    ) -> Dict[str, any]:
        """
        Build syntactic structure for a sentence.
        
        Args:
            sentence: List of words
            word_categories: Mapping from word to category (DET, NOUN, VERB, etc.)
        """
        brain = brain_module.Brain(p=config.p_connect, seed=self.seed + trial_id)
        
        # Create stimuli for each unique word
        unique_words = list(set(sentence))
        for word in unique_words:
            brain.add_stimulus(word, config.k_active)
        
        # Create areas
        areas = ['LEX', 'DET', 'NOUN', 'VERB', 'ADJ', 'NP', 'VP', 'S']
        for area in areas:
            brain.add_area(area, config.n_neurons, config.k_active, config.beta)
        
        # Step 1: Project words to LEX
        word_assemblies = {}
        for word in unique_words:
            for _ in range(config.projection_rounds):
                brain.project({word: ['LEX']}, {})
            word_assemblies[word] = set(brain.area_by_name['LEX'].winners.tolist())
        
        # Check word distinctiveness
        word_overlaps = []
        for i, w1 in enumerate(unique_words):
            for w2 in unique_words[i+1:]:
                word_overlaps.append(len(word_assemblies[w1] & word_assemblies[w2]) / config.k_active)
        
        # Step 2: Categorize words
        category_assemblies = {cat: [] for cat in ['DET', 'NOUN', 'VERB', 'ADJ']}
        for word in unique_words:
            cat = word_categories.get(word, 'NOUN')
            brain.area_by_name['LEX'].winners = np.array(list(word_assemblies[word]), dtype=np.uint32)
            for _ in range(config.projection_rounds):
                brain.project({}, {'LEX': [cat]})
            category_assemblies[cat].append(set(brain.area_by_name[cat].winners.tolist()))
        
        # Step 3: Build NP (simplified: just NOUN or DET+NOUN)
        # Fix relevant assemblies
        if category_assemblies['DET']:
            brain.area_by_name['DET'].winners = np.array(list(category_assemblies['DET'][0]), dtype=np.uint32)
            brain.area_by_name['DET'].fix_assembly()
        
        if category_assemblies['NOUN']:
            brain.area_by_name['NOUN'].winners = np.array(list(category_assemblies['NOUN'][0]), dtype=np.uint32)
            brain.area_by_name['NOUN'].fix_assembly()
        
        # Merge into NP
        merge_sources = {}
        if category_assemblies['DET']:
            merge_sources['DET'] = ['NP']
        if category_assemblies['NOUN']:
            merge_sources['NOUN'] = ['NP']
        
        for _ in range(config.merge_rounds):
            brain.project({}, merge_sources)
        np_assembly = set(brain.area_by_name['NP'].winners.tolist())
        
        # Step 4: Build VP (VERB + NP)
        if category_assemblies['VERB']:
            brain.area_by_name['VERB'].winners = np.array(list(category_assemblies['VERB'][0]), dtype=np.uint32)
            brain.area_by_name['VERB'].fix_assembly()
        brain.area_by_name['NP'].fix_assembly()
        
        for _ in range(config.merge_rounds):
            brain.project({}, {'VERB': ['VP'], 'NP': ['VP']})
        vp_assembly = set(brain.area_by_name['VP'].winners.tolist())
        
        # Step 5: Build S (NP + VP)
        brain.area_by_name['VP'].fix_assembly()
        
        for _ in range(config.merge_rounds):
            brain.project({}, {'NP': ['S'], 'VP': ['S']})
        s_assembly = set(brain.area_by_name['S'].winners.tolist())
        
        return {
            "sentence": sentence,
            "word_assemblies": {w: len(a) for w, a in word_assemblies.items()},
            "word_overlap": np.mean(word_overlaps) if word_overlaps else 0,
            "np_size": len(np_assembly),
            "vp_size": len(vp_assembly),
            "s_size": len(s_assembly),
            "success": len(s_assembly) == config.k_active,
        }
    
    def run(
        self,
        sentences: List[tuple] = None,
        n_trials: int = 3,
        **kwargs
    ) -> ExperimentResult:
        """Run language syntax experiments."""
        self._start_timer()
        
        config = LanguageConfig()
        
        if sentences is None:
            sentences = [
                (["the", "dog", "chased", "the", "cat"], 
                 {"the": "DET", "dog": "NOUN", "chased": "VERB", "cat": "NOUN"}),
                (["a", "big", "bird", "flew"],
                 {"a": "DET", "big": "ADJ", "bird": "NOUN", "flew": "VERB"}),
                (["cats", "sleep"],
                 {"cats": "NOUN", "sleep": "VERB"}),
            ]
        
        self.log("Starting language syntax experiment")
        
        all_results = []
        
        for sentence, categories in sentences:
            self.log(f"\n  Sentence: {' '.join(sentence)}")
            
            trial_results = []
            for trial in range(n_trials):
                try:
                    result = self.build_sentence(config, sentence, categories, trial)
                    trial_results.append(result)
                except Exception as e:
                    self.log(f"    Trial {trial} failed: {e}")
            
            if trial_results:
                success_rate = sum(1 for r in trial_results if r["success"]) / len(trial_results)
                mean_word_overlap = np.mean([r["word_overlap"] for r in trial_results])
                
                all_results.append({
                    "sentence": " ".join(sentence),
                    "success_rate": success_rate,
                    "word_overlap": mean_word_overlap,
                })
                
                self.log(f"    Success: {success_rate:.0%}, Word overlap: {mean_word_overlap:.3f}")
        
        duration = self._stop_timer()
        
        summary = {
            "total_sentences": len(all_results),
            "overall_success_rate": np.mean([r["success_rate"] for r in all_results]),
            "mean_word_overlap": np.mean([r["word_overlap"] for r in all_results]),
        }
        
        self.log(f"\n{'='*60}")
        self.log("LANGUAGE SYNTAX SUMMARY:")
        self.log(f"  Success rate: {summary['overall_success_rate']:.0%}")
        self.log(f"  Word distinctiveness: {1 - summary['mean_word_overlap']:.1%}")
        self.log(f"  Duration: {duration:.1f}s")
        
        return ExperimentResult(
            experiment_name=self.name,
            parameters={"n_trials": n_trials, "seed": self.seed},
            metrics=summary,
            raw_data={"all_results": all_results},
            duration_seconds=duration,
        )


def run_quick_test():
    """Run quick language syntax test."""
    print("="*60)
    print("Language Syntax Test")
    print("="*60)
    
    # Ensure results directory exists
    results_dir = Path(__file__).parent.parent.parent / "results" / "applications"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    exp = LanguageSyntaxExperiment(verbose=True)
    result = exp.run(n_trials=2)
    
    path = exp.save_result(result, "_quick")
    print(f"\nResults saved to: {path}")
    
    return result


if __name__ == "__main__":
    run_quick_test()

