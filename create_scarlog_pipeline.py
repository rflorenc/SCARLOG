#!/usr/bin/env python3
"""
Create SCARLOG code execution pipeline diagram
Based on the scarlog_code_pipeline.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import matplotlib.lines as mlines

def create_scarlog_pipeline():
    """Create the SCARLOG code execution pipeline diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'log_input': '#E3F2FD',      # Light blue
        'slm_encoder': '#FFF3E0',    # Light orange
        'reasoning': '#FCE4EC',      # Light pink
        'verifier': '#F3E5F5',       # Light purple
        'output': '#E8F5E9',         # Light green
        'highlight': '#FFFACD',      # Light yellow for highlighted boxes
        'code_ref': '#FFE0B2'        # Light orange for code references
    }
    
    # Title
    ax.text(7, 9.5, 'SCARLOG: Code Execution Pipeline', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Stage numbers (black circles)
    stage_positions = [
        (1.5, 7.5),   # Stage 1
        (7, 7.5),     # Stage 2
        (1.5, 4.5),   # Stage 3
        (7, 2),       # Stage 4
        (7, 0.5)      # Stage 5
    ]
    
    for i, (x, y) in enumerate(stage_positions):
        circle = Circle((x-0.3, y+0.3), 0.3, facecolor='black', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x-0.3, y+0.3, str(i+1), fontsize=12, fontweight='bold', 
                ha='center', va='center', color='white')
    
    # 1. Log Input
    log_box = FancyBboxPatch((1, 6.5), 4, 1.8, boxstyle="round,pad=0.1",
                             facecolor=colors['log_input'], edgecolor='black', linewidth=2)
    ax.add_patch(log_box)
    ax.text(3, 7.8, 'Log Input', fontsize=14, fontweight='bold', ha='center')
    ax.text(3, 7.3, 'Event Sequences', fontsize=11, ha='center')
    ax.text(3, 6.9, 'e.g., "E5 E22 E11 E5"', fontsize=10, ha='center', style='italic')
    
    # Code reference for Log Input
    ax.text(0.5, 5.8, 'data_loader.py:', fontsize=8, fontweight='bold')
    ax.text(0.5, 5.5, 'lines, labels = \n  load_eventtraces()', fontsize=8)
    
    # 2. SLM Encoder
    slm_box = FancyBboxPatch((6, 6.5), 5, 1.8, boxstyle="round,pad=0.1",
                             facecolor=colors['slm_encoder'], edgecolor='black', linewidth=2)
    ax.add_patch(slm_box)
    ax.text(8.5, 7.8, 'SLM Encoder', fontsize=14, fontweight='bold', ha='center')
    ax.text(8.5, 7.3, 'Supports any small language models', fontsize=11, ha='center')
    ax.text(8.5, 6.9, '768-dim embeddings', fontsize=10, ha='center', style='italic')
    
    # Code reference for SLM Encoder
    ax.text(11.5, 7.3, 'backend.py:', fontsize=8, fontweight='bold')
    ax.text(11.5, 7.0, 'embeddings = \nmodel.encode(lines)', fontsize=8)
    
    # 3. Multi-Perspective Reasoning
    reasoning_box = FancyBboxPatch((1, 3), 10, 2.5, boxstyle="round,pad=0.1",
                                   facecolor=colors['reasoning'], edgecolor='black', linewidth=2)
    ax.add_patch(reasoning_box)
    ax.text(6, 5.2, 'Multi-Perspective Reasoning', fontsize=14, fontweight='bold', ha='center')
    
    # Expert boxes within reasoning
    experts = [
        (2.5, 3.8, 'Distributed\nSystems\nExpert'),
        (4.5, 3.8, 'System\nAdmin'),
        (6.5, 3.8, 'Security\nAnalyst'),
        (8.5, 3.8, 'Network\nEngineer'),
        (10, 3.8, 'Operations\nSpecialist')
    ]
    
    for x, y, label in experts:
        expert_box = Rectangle((x-0.6, y-0.5), 1.2, 1, 
                             facecolor='white', edgecolor='gray', linewidth=1)
        ax.add_patch(expert_box)
        ax.text(x, y, label, fontsize=9, ha='center', va='center')
    
    # Code reference for Reasoning
    ax.text(0.5, 2.3, 'reasoning.py:', fontsize=8, fontweight='bold')
    ax.text(0.5, 2.0, 'run_self_consistency(\n  log_entry,\n  num_samples=5)', fontsize=8)
    
    # Self-Consistency highlight box
    sc_box = Rectangle((11.5, 3.5), 2, 1.2, 
                      facecolor=colors['highlight'], edgecolor='orange', linewidth=2)
    ax.add_patch(sc_box)
    ax.text(12.5, 4.3, 'Self-Consistency', fontsize=10, fontweight='bold', ha='center')
    ax.text(12.5, 3.9, 'T adjustable', fontsize=9, ha='center')
    
    # 4. Verifier System
    verifier_box = FancyBboxPatch((3, 0.8), 8, 1.8, boxstyle="round,pad=0.1",
                                  facecolor=colors['verifier'], edgecolor='black', linewidth=2)
    ax.add_patch(verifier_box)
    ax.text(7, 2.3, 'Verifier System', fontsize=14, fontweight='bold', ha='center')
    
    # Verifier components
    consistency_box = Rectangle((4, 1.2), 2.5, 0.8, 
                              facecolor='white', edgecolor='gray', linewidth=1)
    ax.add_patch(consistency_box)
    ax.text(5.25, 1.6, 'Consistency Check', fontsize=10, ha='center')
    
    confidence_box = Rectangle((7.5, 1.2), 2.5, 0.8, 
                              facecolor='white', edgecolor='gray', linewidth=1)
    ax.add_patch(confidence_box)
    ax.text(8.75, 1.6, 'Confidence Score', fontsize=10, ha='center')
    
    # Code reference for Verifier
    verifier_ref_box = Rectangle((11.5, 1), 2, 1.3, 
                                facecolor=colors['code_ref'], edgecolor='red', linewidth=2)
    ax.add_patch(verifier_ref_box)
    ax.text(12.5, 1.9, 'verifier.py:', fontsize=9, fontweight='bold', ha='center')
    ax.text(12.5, 1.55, 'verify_prediction(', fontsize=8, ha='center')
    ax.text(12.5, 1.35, '  reasoning_result)', fontsize=8, ha='center')
    ax.text(12.5, 1.1, 'REJECT', fontsize=9, fontweight='bold', ha='center', color='red')
    ax.text(12.5, 0.9, 'UNCERTAIN', fontsize=9, fontweight='bold', ha='center', color='orange')
    
    # 5. Enhanced Decision
    output_box = FancyBboxPatch((4, -0.8), 6, 1.2, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 0.2, 'Enhanced Decision', fontsize=14, fontweight='bold', ha='center')
    ax.text(7, -0.2, 'NORMAL/ANOMALY + Explanation', fontsize=11, ha='center')
    ax.text(7, -0.5, 'with confidence scores and reasoning', fontsize=9, ha='center', style='italic')
    
    # Arrows between stages
    # 1 -> 2
    arrow1 = FancyArrowPatch((5, 7.4), (6, 7.4), 
                            connectionstyle="arc3", arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='black')
    ax.add_artist(arrow1)
    
    # 2 -> 3
    arrow2 = FancyArrowPatch((8.5, 6.5), (6, 5.5), 
                            connectionstyle="arc3,rad=0.3", arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='black')
    ax.add_artist(arrow2)
    
    # 3 -> 4
    arrow3 = FancyArrowPatch((6, 3), (7, 2.6), 
                            connectionstyle="arc3", arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='black')
    ax.add_artist(arrow3)
    
    # 4 -> 5
    arrow4 = FancyArrowPatch((7, 0.8), (7, 0.4), 
                            connectionstyle="arc3", arrowstyle='->', 
                            mutation_scale=25, linewidth=2.5, color='black')
    ax.add_artist(arrow4)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating SCARLOG code execution pipeline diagram...")
    
    # Create the diagram
    fig = create_scarlog_pipeline()
    
    # Save the diagram
    output_path = 'figures/scarlog_pipeline_generated.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Pipeline diagram created: {output_path}")