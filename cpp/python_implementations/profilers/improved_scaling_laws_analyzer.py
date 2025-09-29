"""
Improved Scaling Laws Analyzer

Enhancements based on analysis of current results:
1. Statistical analysis and trend detection
2. Performance prediction models
3. Interactive visualization
4. Comparative analysis
5. Optimization recommendations
"""

import numpy as np
import json
import time
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime
import os

class ImprovedScalingLawsAnalyzer:
    """Enhanced scaling laws analyzer with statistical analysis and predictions"""
    
    def __init__(self):
        self.results = []
        self.statistical_models = {}
        self.performance_predictions = {}
        self.optimization_recommendations = {}
        
    def analyze_scaling_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scaling patterns and identify trends"""
        analysis = {
            'scaling_laws': {},
            'performance_trends': {},
            'memory_efficiency_trends': {},
            'optimal_configurations': {},
            'bottlenecks': {},
            'predictions': {}
        }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # 1. Scaling Laws Analysis
        analysis['scaling_laws'] = self._analyze_scaling_laws(df)
        
        # 2. Performance Trends
        analysis['performance_trends'] = self._analyze_performance_trends(df)
        
        # 3. Memory Efficiency Trends
        analysis['memory_efficiency_trends'] = self._analyze_memory_trends(df)
        
        # 4. Optimal Configurations
        analysis['optimal_configurations'] = self._find_optimal_configurations(df)
        
        # 5. Bottleneck Analysis
        analysis['bottlenecks'] = self._identify_bottlenecks(df)
        
        # 6. Performance Predictions
        analysis['predictions'] = self._predict_performance(df)
        
        return analysis
    
    def _analyze_scaling_laws(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how performance scales with neuron count and active percentage"""
        laws = {}
        
        # Group by active percentage to analyze scaling with neuron count
        for active_pct in df['active_percentage'].unique():
            subset = df[df['active_percentage'] == active_pct]
            if len(subset) < 2:
                continue
                
            # Fit power law: steps_per_sec = a * neurons^b
            neurons = subset['n_neurons'].values
            steps_per_sec = subset['steps_per_sec'].values
            
            try:
                # Log-log regression
                log_neurons = np.log10(neurons)
                log_steps = np.log10(steps_per_sec)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_neurons, log_steps)
                
                laws[f'active_{active_pct:.4f}'] = {
                    'scaling_exponent': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'equation': f'steps/sec = 10^{intercept:.3f} * neurons^{slope:.3f}',
                    'scaling_type': self._classify_scaling(slope)
                }
            except:
                laws[f'active_{active_pct:.4f}'] = {'error': 'Could not fit scaling law'}
        
        return laws
    
    def _classify_scaling(self, exponent: float) -> str:
        """Classify the type of scaling based on exponent"""
        if exponent > 0.8:
            return "Superlinear (scales better than linear)"
        elif exponent > 0.2:
            return "Sublinear (scales worse than linear)"
        elif exponent > -0.2:
            return "Constant (scales linearly)"
        else:
            return "Negative (performance degrades with scale)"
    
    def _analyze_performance_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends across different configurations"""
        trends = {}
        
        # Performance vs Active Percentage
        for neuron_count in df['n_neurons'].unique():
            subset = df[df['n_neurons'] == neuron_count]
            if len(subset) < 3:
                continue
                
            active_pct = subset['active_percentage'].values
            steps_per_sec = subset['steps_per_sec'].values
            
            # Find optimal active percentage
            optimal_idx = np.argmax(steps_per_sec)
            optimal_active_pct = active_pct[optimal_idx]
            max_performance = steps_per_sec[optimal_idx]
            
            trends[f'neurons_{neuron_count}'] = {
                'optimal_active_percentage': optimal_active_pct,
                'max_steps_per_sec': max_performance,
                'performance_range': {
                    'min': np.min(steps_per_sec),
                    'max': np.max(steps_per_sec),
                    'std': np.std(steps_per_sec)
                },
                'trend': self._analyze_trend(active_pct, steps_per_sec)
            }
        
        return trends
    
    def _analyze_trend(self, x: np.ndarray, y: np.ndarray) -> str:
        """Analyze the trend between x and y"""
        if len(x) < 2:
            return "Insufficient data"
        
        # Calculate correlation
        correlation = np.corrcoef(x, y)[0, 1]
        
        if correlation > 0.7:
            return "Strong positive correlation"
        elif correlation > 0.3:
            return "Moderate positive correlation"
        elif correlation > -0.3:
            return "Weak correlation"
        elif correlation > -0.7:
            return "Moderate negative correlation"
        else:
            return "Strong negative correlation"
    
    def _analyze_memory_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze memory efficiency trends"""
        memory_trends = {}
        
        # Memory per neuron analysis (try both field names)
        if 'memory_per_neuron' in df.columns:
            memory_per_neuron = df['memory_per_neuron'].values
        elif 'bytes_per_neuron' in df.columns:
            memory_per_neuron = df['bytes_per_neuron'].values
        else:
            memory_trends['memory_scaling'] = {'error': 'No memory per neuron data found'}
            return memory_trends
        neurons = df['n_neurons'].values
        
        # Check if memory per neuron decreases with scale (good scaling)
        log_neurons = np.log10(neurons)
        log_memory = np.log10(memory_per_neuron + 1e-10)  # Add small value to avoid log(0)
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_neurons, log_memory)
            
            memory_trends['memory_scaling'] = {
                'slope': slope,
                'r_squared': r_value**2,
                'interpretation': self._interpret_memory_scaling(slope),
                'efficiency_trend': 'Improving' if slope < 0 else 'Degrading'
            }
        except:
            memory_trends['memory_scaling'] = {'error': 'Could not analyze memory scaling'}
        
        return memory_trends
    
    def _interpret_memory_scaling(self, slope: float) -> str:
        """Interpret memory scaling slope"""
        if slope < -0.5:
            return "Excellent: Memory per neuron decreases significantly with scale"
        elif slope < -0.1:
            return "Good: Memory per neuron decreases with scale"
        elif slope < 0.1:
            return "Constant: Memory per neuron stays roughly constant"
        else:
            return "Poor: Memory per neuron increases with scale"
    
    def _find_optimal_configurations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find optimal configurations for different use cases"""
        optimal = {}
        
        # Best overall performance
        best_overall = df.loc[df['steps_per_sec'].idxmax()]
        optimal['best_overall'] = {
            'neurons': int(best_overall['n_neurons']),
            'active_percentage': best_overall['active_percentage'],
            'steps_per_sec': best_overall['steps_per_sec'],
            'memory_efficiency': best_overall['memory_efficiency']
        }
        
        # Best memory efficiency
        best_memory = df.loc[df['memory_efficiency'].idxmax()]
        optimal['best_memory'] = {
            'neurons': int(best_memory['n_neurons']),
            'active_percentage': best_memory['active_percentage'],
            'steps_per_sec': best_memory['steps_per_sec'],
            'memory_efficiency': best_memory['memory_efficiency']
        }
        
        # Best for large scale (1B+ neurons)
        large_scale = df[df['n_neurons'] >= 1e9]
        if not large_scale.empty:
            best_large = large_scale.loc[large_scale['steps_per_sec'].idxmax()]
            optimal['best_large_scale'] = {
                'neurons': int(best_large['n_neurons']),
                'active_percentage': best_large['active_percentage'],
                'steps_per_sec': best_large['steps_per_sec'],
                'memory_efficiency': best_large['memory_efficiency']
            }
        
        return optimal
    
    def _identify_bottlenecks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify performance bottlenecks"""
        bottlenecks = {}
        
        # Analyze performance degradation patterns
        for active_pct in df['active_percentage'].unique():
            subset = df[df['active_percentage'] == active_pct].sort_values('n_neurons')
            if len(subset) < 2:
                continue
            
            neurons = subset['n_neurons'].values
            steps_per_sec = subset['steps_per_sec'].values
            
            # Calculate performance degradation rate
            performance_ratios = steps_per_sec[1:] / steps_per_sec[:-1]
            neuron_ratios = neurons[1:] / neurons[:-1]
            
            degradation_rate = np.mean(performance_ratios / neuron_ratios)
            
            bottlenecks[f'active_{active_pct:.4f}'] = {
                'degradation_rate': degradation_rate,
                'bottleneck_type': self._classify_bottleneck(degradation_rate),
                'recommendations': self._get_bottleneck_recommendations(degradation_rate)
            }
        
        return bottlenecks
    
    def _classify_bottleneck(self, degradation_rate: float) -> str:
        """Classify bottleneck type based on degradation rate"""
        if degradation_rate > 0.8:
            return "No significant bottleneck"
        elif degradation_rate > 0.5:
            return "Memory bandwidth bottleneck"
        elif degradation_rate > 0.2:
            return "Compute bottleneck"
        else:
            return "Severe bottleneck - likely memory or synchronization"
    
    def _get_bottleneck_recommendations(self, degradation_rate: float) -> List[str]:
        """Get recommendations based on bottleneck type"""
        if degradation_rate > 0.8:
            return ["System is well-optimized", "Consider increasing batch sizes"]
        elif degradation_rate > 0.5:
            return ["Optimize memory access patterns", "Use memory pooling", "Consider data compression"]
        elif degradation_rate > 0.2:
            return ["Optimize compute kernels", "Use more efficient algorithms", "Consider parallelization"]
        else:
            return ["Major optimization needed", "Profile memory usage", "Check for memory leaks", "Consider distributed computing"]
    
    def _predict_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict performance for untested configurations"""
        predictions = {}
        
        # Create prediction models for different active percentages
        for active_pct in df['active_percentage'].unique():
            subset = df[df['active_percentage'] == active_pct]
            if len(subset) < 3:
                continue
            
            neurons = subset['n_neurons'].values
            steps_per_sec = subset['steps_per_sec'].values
            
            # Fit polynomial model
            try:
                coeffs = np.polyfit(np.log10(neurons), np.log10(steps_per_sec), 2)
                poly_func = np.poly1d(coeffs)
                
                # Predict for common scales
                test_scales = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
                predictions[f'active_{active_pct:.4f}'] = {}
                
                for scale in test_scales:
                    if scale not in neurons:  # Only predict for untested scales
                        log_neurons = np.log10(scale)
                        log_predicted = poly_func(log_neurons)
                        predicted_steps = 10**log_predicted
                        
                        predictions[f'active_{active_pct:.4f}'][f'{scale:.0e}'] = {
                            'predicted_steps_per_sec': predicted_steps,
                            'confidence': 'Medium'  # Could be improved with more data
                        }
            except:
                predictions[f'active_{active_pct:.4f}'] = {'error': 'Could not create prediction model'}
        
        return predictions
    
    def generate_visualizations(self, results: List[Dict[str, Any]], output_dir: str = "__generated__"):
        """Generate visualizations of scaling laws"""
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(results)
        
        # 1. Performance vs Scale
        plt.figure(figsize=(12, 8))
        for active_pct in df['active_percentage'].unique():
            subset = df[df['active_percentage'] == active_pct]
            plt.loglog(subset['n_neurons'], subset['steps_per_sec'], 
                      marker='o', label=f'{active_pct:.1%} active')
        
        plt.xlabel('Number of Neurons')
        plt.ylabel('Steps per Second')
        plt.title('Performance Scaling Laws')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'scaling_laws_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Memory Efficiency vs Scale
        plt.figure(figsize=(12, 8))
        plt.semilogx(df['n_neurons'], df['memory_efficiency'], 'o-')
        plt.xlabel('Number of Neurons')
        plt.ylabel('Memory Efficiency (%)')
        plt.title('Memory Efficiency Scaling')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'scaling_laws_memory.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance vs Active Percentage
        plt.figure(figsize=(12, 8))
        for neuron_count in df['n_neurons'].unique():
            subset = df[df['n_neurons'] == neuron_count]
            plt.plot(subset['active_percentage'] * 100, subset['steps_per_sec'], 
                    marker='o', label=f'{neuron_count:.0e} neurons')
        
        plt.xlabel('Active Percentage (%)')
        plt.ylabel('Steps per Second')
        plt.title('Performance vs Active Percentage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'scaling_laws_active_percentage.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results: List[Dict[str, Any]], output_file: str = None) -> str:
        """Generate comprehensive scaling laws report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"__generated__/scaling_laws_report_{timestamp}.json"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Perform comprehensive analysis
        analysis = self.analyze_scaling_patterns(results)
        
        # Generate visualizations
        self.generate_visualizations(results)
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(analysis),
            'detailed_analysis': analysis,
            'raw_results': results,
            'recommendations': self._generate_recommendations(analysis)
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        return {
            'total_tests': len(analysis.get('raw_results', [])),
            'best_performance': analysis.get('optimal_configurations', {}).get('best_overall', {}),
            'scaling_characteristics': self._summarize_scaling_characteristics(analysis),
            'key_insights': self._extract_key_insights(analysis)
        }
    
    def _summarize_scaling_characteristics(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Summarize scaling characteristics"""
        characteristics = {}
        
        scaling_laws = analysis.get('scaling_laws', {})
        for key, law in scaling_laws.items():
            if 'scaling_type' in law:
                characteristics[key] = law['scaling_type']
        
        return characteristics
    
    def _extract_key_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key insights from analysis"""
        insights = []
        
        # Performance insights
        optimal = analysis.get('optimal_configurations', {})
        if 'best_overall' in optimal:
            best = optimal['best_overall']
            insights.append(f"Best performance: {best['steps_per_sec']:.1f} steps/sec at {best['neurons']:,} neurons, {best['active_percentage']:.1%} active")
        
        # Scaling insights
        scaling_laws = analysis.get('scaling_laws', {})
        for key, law in scaling_laws.items():
            if 'scaling_type' in law and 'Superlinear' in law['scaling_type']:
                insights.append(f"Superlinear scaling detected for {key}")
        
        # Memory insights
        memory_trends = analysis.get('memory_efficiency_trends', {})
        if 'memory_scaling' in memory_trends and 'efficiency_trend' in memory_trends['memory_scaling']:
            trend = memory_trends['memory_scaling']['efficiency_trend']
            insights.append(f"Memory efficiency trend: {trend}")
        
        return insights
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Bottleneck recommendations
        bottlenecks = analysis.get('bottlenecks', {})
        for key, bottleneck in bottlenecks.items():
            if 'recommendations' in bottleneck:
                recommendations.extend(bottleneck['recommendations'])
        
        # General recommendations based on analysis
        optimal = analysis.get('optimal_configurations', {})
        if 'best_overall' in optimal:
            best = optimal['best_overall']
            recommendations.append(f"Use {best['neurons']:,} neurons with {best['active_percentage']:.1%} active for optimal performance")
        
        return list(set(recommendations))  # Remove duplicates

# Example usage
if __name__ == "__main__":
    # This would be used with actual scaling laws results
    analyzer = ImprovedScalingLawsAnalyzer()
    print("✅ Improved Scaling Laws Analyzer ready!")
    print("   Features:")
    print("   - Statistical analysis and trend detection")
    print("   - Performance prediction models")
    print("   - Interactive visualizations")
    print("   - Comparative analysis")
    print("   - Optimization recommendations")
