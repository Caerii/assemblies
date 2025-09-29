"""
Test script for the improved scaling laws analyzer
Uses the existing scaling laws results to generate a comprehensive report
"""

import json
import os
from improved_scaling_laws_analyzer import ImprovedScalingLawsAnalyzer

def load_scaling_laws_results():
    """Load results from the scaling laws analyzer output"""
    # This would be the actual output from running scaling_laws_analyzer.py
    # For now, let's create some sample data based on the output we saw
    sample_results = [
        {
            "scale_name": "1M neurons (1%)",
            "n_neurons": 1000000,
            "active_percentage": 0.01,
            "active_neurons": 10000,
            "steps_per_sec": 79.8,
            "ms_per_step": 12.54,
            "neurons_per_sec": 79755247,
            "active_per_sec": 3987762,
            "memory_efficiency": 100.8,
            "memory_per_neuron": 0.000000001043
        },
        {
            "scale_name": "1M neurons (5%)",
            "n_neurons": 1000000,
            "active_percentage": 0.05,
            "active_neurons": 50000,
            "steps_per_sec": 240.7,
            "ms_per_step": 4.16,
            "neurons_per_sec": 240652457,
            "active_per_sec": 60163114,
            "memory_efficiency": 100.1,
            "memory_per_neuron": 0.000000005215
        },
        {
            "scale_name": "1M neurons (10%)",
            "n_neurons": 1000000,
            "active_percentage": 0.10,
            "active_neurons": 100000,
            "steps_per_sec": 88.0,
            "ms_per_step": 11.36,
            "neurons_per_sec": 88031734,
            "active_per_sec": 44015867,
            "memory_efficiency": 100.1,
            "memory_per_neuron": 0.000000010431
        },
        {
            "scale_name": "10M neurons (1%)",
            "n_neurons": 10000000,
            "active_percentage": 0.01,
            "active_neurons": 100000,
            "steps_per_sec": 102.4,
            "ms_per_step": 9.76,
            "neurons_per_sec": 1024211332,
            "active_per_sec": 51210567,
            "memory_efficiency": 100.1,
            "memory_per_neuron": 0.000000001043
        },
        {
            "scale_name": "10M neurons (5%)",
            "n_neurons": 10000000,
            "active_percentage": 0.05,
            "active_neurons": 500000,
            "steps_per_sec": 164.5,
            "ms_per_step": 6.08,
            "neurons_per_sec": 1645199554,
            "active_per_sec": 411299889,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000005215
        },
        {
            "scale_name": "10M neurons (10%)",
            "n_neurons": 10000000,
            "active_percentage": 0.10,
            "active_neurons": 1000000,
            "steps_per_sec": 193.3,
            "ms_per_step": 5.17,
            "neurons_per_sec": 1933297374,
            "active_per_sec": 966648687,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000010431
        },
        {
            "scale_name": "100M neurons (0.1%)",
            "n_neurons": 100000000,
            "active_percentage": 0.001,
            "active_neurons": 100000,
            "steps_per_sec": 317.0,
            "ms_per_step": 3.15,
            "neurons_per_sec": 31697529487,
            "active_per_sec": 158487647,
            "memory_efficiency": 100.1,
            "memory_per_neuron": 0.000000000104
        },
        {
            "scale_name": "100M neurons (0.5%)",
            "n_neurons": 100000000,
            "active_percentage": 0.005,
            "active_neurons": 500000,
            "steps_per_sec": 214.2,
            "ms_per_step": 4.67,
            "neurons_per_sec": 21416440543,
            "active_per_sec": 535411014,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000000522
        },
        {
            "scale_name": "100M neurons (1%)",
            "n_neurons": 100000000,
            "active_percentage": 0.01,
            "active_neurons": 1000000,
            "steps_per_sec": 241.1,
            "ms_per_step": 4.15,
            "neurons_per_sec": 24107306439,
            "active_per_sec": 1205365322,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000001043
        },
        {
            "scale_name": "1B neurons (0.01%)",
            "n_neurons": 1000000000,
            "active_percentage": 0.0001,
            "active_neurons": 100000,
            "steps_per_sec": 143.0,
            "ms_per_step": 6.99,
            "neurons_per_sec": 143044940432,
            "active_per_sec": 71522470,
            "memory_efficiency": 100.1,
            "memory_per_neuron": 0.000000000010
        },
        {
            "scale_name": "1B neurons (0.05%)",
            "n_neurons": 1000000000,
            "active_percentage": 0.0005,
            "active_neurons": 500000,
            "steps_per_sec": 136.9,
            "ms_per_step": 7.30,
            "neurons_per_sec": 136949531351,
            "active_per_sec": 342373828,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000000052
        },
        {
            "scale_name": "1B neurons (0.1%)",
            "n_neurons": 1000000000,
            "active_percentage": 0.001,
            "active_neurons": 1000000,
            "steps_per_sec": 266.7,
            "ms_per_step": 3.75,
            "neurons_per_sec": 266697247936,
            "active_per_sec": 1333486240,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000000104
        },
        {
            "scale_name": "10B neurons (0.001%)",
            "n_neurons": 10000000000,
            "active_percentage": 0.00001,
            "active_neurons": 100000,
            "steps_per_sec": 305.7,
            "ms_per_step": 3.27,
            "neurons_per_sec": 3056673789412,
            "active_per_sec": 152833689,
            "memory_efficiency": 100.1,
            "memory_per_neuron": 0.000000000001
        },
        {
            "scale_name": "10B neurons (0.005%)",
            "n_neurons": 10000000000,
            "active_percentage": 0.00005,
            "active_neurons": 500000,
            "steps_per_sec": 166.9,
            "ms_per_step": 5.99,
            "neurons_per_sec": 1669237292100,
            "active_per_sec": 417309323,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000000005
        },
        {
            "scale_name": "10B neurons (0.01%)",
            "n_neurons": 10000000000,
            "active_percentage": 0.0001,
            "active_neurons": 1000000,
            "steps_per_sec": 185.4,
            "ms_per_step": 5.39,
            "neurons_per_sec": 1853708993666,
            "active_per_sec": 926854497,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000000010
        },
        {
            "scale_name": "86B neurons (0.0001%)",
            "n_neurons": 86000000000,
            "active_percentage": 0.000001,
            "active_neurons": 86000,
            "steps_per_sec": 168.8,
            "ms_per_step": 5.93,
            "neurons_per_sec": 14513788097856,
            "active_per_sec": 72568940,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000000000
        },
        {
            "scale_name": "86B neurons (0.0005%)",
            "n_neurons": 86000000000,
            "active_percentage": 0.000005,
            "active_neurons": 430000,
            "steps_per_sec": 305.4,
            "ms_per_step": 3.27,
            "neurons_per_sec": 26265877875742,
            "active_per_sec": 656646947,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000000001
        },
        {
            "scale_name": "86B neurons (0.001%)",
            "n_neurons": 86000000000,
            "active_percentage": 0.00001,
            "active_neurons": 860000,
            "steps_per_sec": 158.0,
            "ms_per_step": 6.33,
            "neurons_per_sec": 13583887613861,
            "active_per_sec": 679194381,
            "memory_efficiency": 100.0,
            "memory_per_neuron": 0.000000000001
        }
    ]
    
    return sample_results

def main():
    """Test the improved scaling laws analyzer"""
    print("🧪 Testing Improved Scaling Laws Analyzer")
    print("=" * 50)
    
    # Load sample data
    results = load_scaling_laws_results()
    print(f"📊 Loaded {len(results)} test results")
    
    # Create analyzer
    analyzer = ImprovedScalingLawsAnalyzer()
    
    # Generate comprehensive report
    print("🔍 Generating comprehensive analysis...")
    report_file = analyzer.generate_report(results)
    
    print(f"✅ Report generated: {report_file}")
    
    # Check if files were created in __generated__ folder
    generated_dir = "__generated__"
    if os.path.exists(generated_dir):
        files = os.listdir(generated_dir)
        print(f"\n📁 Files in {generated_dir}:")
        for file in sorted(files):
            print(f"   - {file}")
    
    # Show summary of analysis
    print("\n📈 Analysis Summary:")
    analysis = analyzer.analyze_scaling_patterns(results)
    
    # Show optimal configurations
    optimal = analysis.get('optimal_configurations', {})
    if 'best_overall' in optimal:
        best = optimal['best_overall']
        print(f"   🏆 Best Performance: {best['steps_per_sec']:.1f} steps/sec")
        print(f"      Neurons: {best['neurons']:,}")
        print(f"      Active: {best['active_percentage']:.1%}")
    
    # Show scaling characteristics
    scaling_laws = analysis.get('scaling_laws', {})
    print(f"\n📊 Scaling Laws Found: {len(scaling_laws)}")
    for key, law in scaling_laws.items():
        if 'scaling_type' in law:
            print(f"   - {key}: {law['scaling_type']}")
    
    # Show key insights
    insights = analysis.get('key_insights', [])
    if insights:
        print(f"\n💡 Key Insights:")
        for insight in insights[:3]:  # Show first 3 insights
            print(f"   - {insight}")

if __name__ == "__main__":
    main()
