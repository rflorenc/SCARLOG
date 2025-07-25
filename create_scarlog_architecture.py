#!/usr/bin/env python3
"""
Create SCARLOG architecture diagram without title
Based on the scarlog_architecture_current.png layout
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, ConnectionPatch
import numpy as np

def create_scarlog_architecture():
    """Create the SCARLOG architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme based on the original
    colors = {
        'config': '#F0F0F0',      # Light gray for configuration layer
        'data': '#E8F5E9',        # Light green for data processing
        'inference': '#FFE4E1',   # Light pink for inference backend
        'evaluation': '#E6E6FA',  # Light purple for evaluation engine
        'enhancement': '#FFF8DC', # Light yellow for reasoning enhancement
        'monitoring': '#E0F2F1',  # Light teal for system monitoring
        'output': '#FFE0F5',      # Light pink for output/reporting
        'labeled': '#FFFACD',     # Light yellow for labeled components
        'unlabeled': '#F0FFFF'    # Light blue for unlabeled components
    }
    
    # Main title area 
    # ax.text(8, 9.7, 'SCARLOG: Self Consistent Anomaly Reasoning for Logs', 
    #         fontsize=16, fontweight='bold', ha='center')
    
    # Subtitle
    ax.text(8, 9.3, 'System Architecture Overview', 
            fontsize=13, fontweight='bold', ha='center', style='italic')
    
    # Side labels
    ax.text(0.2, 8.5, 'CONFIG', fontsize=10, fontweight='bold', rotation=90, 
            va='center', ha='center', color='#1E88E5')
    ax.text(0.2, 6.8, 'DATA', fontsize=10, fontweight='bold', rotation=90, 
            va='center', ha='center', color='#43A047')
    ax.text(0.2, 4.5, 'EVAL', fontsize=10, fontweight='bold', rotation=90, 
            va='center', ha='center', color='#8E24AA')
    ax.text(0.2, 2.5, 'ENHANCE', fontsize=10, fontweight='bold', rotation=90, 
            va='center', ha='center', color='#F4511E')
    ax.text(0.2, 0.8, 'OUTPUT', fontsize=10, fontweight='bold', rotation=90, 
            va='center', ha='center', color='#E91E63')
    
    # 1. CONFIGURATION LAYER
    config_box = FancyBboxPatch((1, 8.0), 14, 0.8, boxstyle="round,pad=0.05",
                                facecolor=colors['config'], edgecolor='black', linewidth=1.5)
    ax.add_patch(config_box)
    ax.text(8, 8.5, 'CONFIGURATION LAYER', fontsize=11, fontweight='bold', ha='center')
    
    # Config components
    config_components = [
        (3, 8.2, 'Models & Reasoning\napproaches.yaml'),
        (8, 8.2, 'Data Sources\ndatasets.yaml'),
        (13, 8.2, 'System Config\nsettings.py')
    ]
    
    for x, y, text in config_components:
        comp_box = Rectangle((x-1.2, y-0.15), 2.4, 0.3, 
                           facecolor='white', edgecolor='gray', linewidth=0.5)
        ax.add_patch(comp_box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    # 2. DATA PROCESSING AND INFERENCE BACKEND
    # Data Processing
    data_box = FancyBboxPatch((1, 6.3), 6.5, 1.2, boxstyle="round,pad=0.05",
                              facecolor=colors['data'], edgecolor='black', linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(4.25, 7.2, 'DATA PROCESSING', fontsize=11, fontweight='bold', ha='center')
    
    data_components = [
        (2.2, 6.7, 'Data Loading\nHDFS dataset\nloaders'),
        (4.25, 6.7, 'Preprocessing\nTokenization\nNormalization'),
        (6.3, 6.7, 'Enrichment\nEvent+Template\nVectorizing')
    ]
    
    for x, y, text in data_components:
        comp_box = Rectangle((x-0.8, y-0.25), 1.6, 0.5, 
                           facecolor='white', edgecolor='gray', linewidth=0.5)
        ax.add_patch(comp_box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    # Inference Backend
    inference_box = FancyBboxPatch((8.5, 6.3), 6.5, 1.2, boxstyle="round,pad=0.05",
                                   facecolor=colors['inference'], edgecolor='black', linewidth=1.5)
    ax.add_patch(inference_box)
    ax.text(11.75, 7.2, 'INFERENCE BACKEND', fontsize=11, fontweight='bold', ha='center')
    
    inference_components = [
        (9.7, 6.7, 'Backend Factory\nAuto-detection\nFallback logic'),
        (11.75, 6.7, 'Local Transformers\n8-bit quantization\nFlash Attention 2'),
        (13.8, 6.7, 'Model Manager\nAsync contexts\nBatch processing')
    ]
    
    for x, y, text in inference_components:
        comp_box = Rectangle((x-0.8, y-0.25), 1.6, 0.5, 
                           facecolor='white', edgecolor='gray', linewidth=0.5)
        ax.add_patch(comp_box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    # 3. EVALUATION ENGINE
    eval_box = FancyBboxPatch((1, 3.8), 14, 1.8, boxstyle="round,pad=0.05",
                              facecolor=colors['evaluation'], edgecolor='black', linewidth=1.5)
    ax.add_patch(eval_box)
    ax.text(8, 5.3, 'EVALUATION ENGINE', fontsize=12, fontweight='bold', ha='center')
    
    # ML Baselines text
    ax.text(8, 4.9, 'ML Baselines: RandomForest, ExtraTrees, LogisticRegression', 
            fontsize=9, ha='center', style='italic')
    
    # Evaluation scenarios
    eval_scenarios = [
        (2.5, 4.3, 'Labeled\nBGL', colors['labeled'], '• Binary classification\n• Class imbalance'),
        (5, 4.3, 'Labeled\nEventTraces', colors['labeled'], '• HDFS logs\n• Block operations'),
        (7.5, 4.3, 'Labeled\nUNSW-NB15', colors['labeled'], '• Network intrusion\n• 9 attack types'),
        (10, 4.3, 'Unlabeled\nBGL', colors['unlabeled'], '• DBSCAN/HDBSCAN\n• Anomaly detection'),
        (12.5, 4.3, 'Unlabeled\nEventTraces', colors['unlabeled'], '• Clustering\n• Silhouette scoring'),
        (14.3, 4.3, 'Unlabeled\nUNSW-NB15', colors['unlabeled'], '• Density-based\n• Network patterns')
    ]
    
    for x, y, title, color, details in eval_scenarios:
        scenario_box = Rectangle((x-0.7, y-0.35), 1.4, 0.7, 
                               facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(scenario_box)
        ax.text(x, y+0.15, title, fontsize=8, fontweight='bold', ha='center', va='center')
        ax.text(x, y-0.15, details, fontsize=6, ha='center', va='center')
    
    # 4. REASONING ENHANCEMENT AND SYSTEM MONITORING
    # Reasoning Enhancement
    reasoning_box = FancyBboxPatch((1, 1.8), 7, 1.5, boxstyle="round,pad=0.05",
                                   facecolor=colors['enhancement'], edgecolor='black', linewidth=1.5)
    ax.add_patch(reasoning_box)
    ax.text(4.5, 2.9, 'REASONING ENHANCEMENT', fontsize=11, fontweight='bold', ha='center')
    
    reasoning_components = [
        (2.5, 2.3, 'Self-Consistency', '• 3 expert perspectives\n• Temperature: 0.6\n• Majority voting\n• Confidence scoring'),
        (6.5, 2.3, 'Verifier Feedback', '• Critical review\n• CONFIRM/REJECT\n• UNCERTAIN states\n• Enhanced thinking')
    ]
    
    for x, y, title, details in reasoning_components:
        comp_box = Rectangle((x-1.7, y-0.5), 3.4, 1.0, 
                           facecolor='white', edgecolor='gray', linewidth=0.5)
        ax.add_patch(comp_box)
        ax.text(x, y+0.35, title, fontsize=9, fontweight='bold', ha='center')
        ax.text(x, y-0.1, details, fontsize=7, ha='center', va='center')
    
    # System Monitoring
    monitoring_box = FancyBboxPatch((9, 1.8), 6, 1.5, boxstyle="round,pad=0.05",
                                    facecolor=colors['monitoring'], edgecolor='black', linewidth=1.5)
    ax.add_patch(monitoring_box)
    ax.text(12, 2.9, 'SYSTEM MONITORING', fontsize=11, fontweight='bold', ha='center')
    
    monitoring_text = """• GPU/CPU/Memory tracking
• Power estimation
• Token usage analysis
• Batch timing metrics
• Performance visualization"""
    
    ax.text(12, 2.2, monitoring_text, fontsize=8, ha='center', va='center')
    
    # 5. REPORTING (Output Layer)
    reporting_box = FancyBboxPatch((1, 0.2), 14, 1.2, boxstyle="round,pad=0.05",
                                   facecolor=colors['output'], edgecolor='black', linewidth=1.5)
    ax.add_patch(reporting_box)
    ax.text(8, 1.1, 'REPORTING', fontsize=11, fontweight='bold', ha='center')
    
    reporting_components = [
        (3.5, 0.6, 'Metrics\nF1, accuracy\nAUC, recall\nSilhouette'),
        (8, 0.6, 'Visualizations\nClustering plots\nConfusion matrices\nPerformance charts'),
        (12.5, 0.6, 'Analysis Results\nReasoning scores\nEnhanced predictions\nModel comparisons')
    ]
    
    for x, y, text in reporting_components:
        comp_box = Rectangle((x-1.5, y-0.25), 3.0, 0.5, 
                           facecolor='white', edgecolor='gray', linewidth=0.5)
        ax.add_patch(comp_box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    # Add flow arrows
    # Config to Data/Inference
    arrow1 = ConnectionPatch((4, 8.0), (4.25, 7.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='gray', linewidth=1.5)
    ax.add_artist(arrow1)
    
    arrow2 = ConnectionPatch((12, 8.0), (11.75, 7.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='gray', linewidth=1.5)
    ax.add_artist(arrow2)
    
    # Data/Inference to Evaluation
    arrow3 = ConnectionPatch((4.25, 6.3), (5, 5.6), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='black', linewidth=2)
    ax.add_artist(arrow3)
    
    arrow4 = ConnectionPatch((11.75, 6.3), (11, 5.6), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='black', linewidth=2)
    ax.add_artist(arrow4)
    
    # Evaluation to Enhancement/Monitoring
    arrow5 = ConnectionPatch((4, 3.8), (4.5, 3.3), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='black', linewidth=2)
    ax.add_artist(arrow5)
    
    arrow6 = ConnectionPatch((12, 3.8), (12, 3.3), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='black', linewidth=2)
    ax.add_artist(arrow6)
    
    # Enhancement/Monitoring to Reporting
    arrow7 = ConnectionPatch((4.5, 1.8), (5, 1.4), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='black', linewidth=2)
    ax.add_artist(arrow7)
    
    arrow8 = ConnectionPatch((12, 1.8), (11, 1.4), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='black', linewidth=2)
    ax.add_artist(arrow8)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating SCARLOG architecture diagram without title...")
    
    # Create the diagram
    fig = create_scarlog_architecture()
    
    # Save the diagram
    output_path = 'figures/scarlog_architecture_no_title.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Architecture diagram created: {output_path}")