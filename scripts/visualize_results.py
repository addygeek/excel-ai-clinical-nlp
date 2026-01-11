"""
Clinical NLP Results Visualization Module
Load and visualize the saved JSON results with rich visualizations
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List
import pandas as pd
from datetime import datetime
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

CLINICAL_DISCLAIMER = """
⚠️  CLINICAL DISCLAIMER:
This is a demonstration system and should NOT be used for actual medical decisions.
Always consult qualified healthcare professionals for medical advice.
"""


class ClinicalNLPVisualizer:
    """
    Visualize and analyze Clinical NLP results
    """
    
    def __init__(self, json_file: str = 'clinical_nlp_results.json'):
        """
        Initialize visualizer with JSON file
        
        Args:
            json_file: Path to saved JSON results
        """
        self.json_file = json_file
        self.results = None
        self.load_results()
    
    def load_results(self):
        """Load results from JSON file"""
        try:
            with open(self.json_file, 'r') as f:
                self.results = json.load(f)
            print(f"✅ Loaded results from '{self.json_file}'")
            print(f"📊 Total results: {len(self.results) if isinstance(self.results, list) else 1}")
        except FileNotFoundError:
            print(f"❌ Error: File '{self.json_file}' not found!")
            self.results = None
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON format - {e}")
            self.results = None
    
    def display_summary(self):
        """Display a comprehensive summary of results"""
        if not self.results:
            print("No results to display!")
            return
        
        print("\n" + "="*80)
        print("📋 CLINICAL NLP ANALYSIS SUMMARY")
        print("="*80)
        
        # Handle both single result and list of results
        results_list = self.results if isinstance(self.results, list) else [self.results]
        
        for idx, result in enumerate(results_list, 1):
            print(f"\n{'─'*80}")
            print(f"Result #{idx}")
            print(f"{'─'*80}")
            
            # Display metadata if available
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"\n📅 Timestamp: {metadata.get('timestamp', 'N/A')}")
                print(f"📝 Input Text Length: {metadata.get('input_length', 'N/A')} characters")
            
            # Display entities
            if 'entities' in result:
                entities = result['entities']
                print(f"\n🏷️  ENTITIES EXTRACTED: {len(entities)}")
                
                # Group by type
                entity_types = {}
                for entity in entities:
                    e_type = entity.get('type', 'UNKNOWN')
                    if e_type not in entity_types:
                        entity_types[e_type] = []
                    entity_types[e_type].append(entity)
                
                for e_type, e_list in sorted(entity_types.items()):
                    print(f"\n  {e_type} ({len(e_list)}):")
                    for entity in e_list[:5]:  # Show first 5
                        conf = entity.get('confidence', 0)
                        text = entity.get('text', '')
                        print(f"    • {text} (confidence: {conf:.2%})")
                    if len(e_list) > 5:
                        print(f"    ... and {len(e_list) - 5} more")
            
            # Display structured summary
            if 'structured_summary' in result:
                summary = result['structured_summary']
                print(f"\n📊 STRUCTURED SUMMARY:")
                for key, value in summary.items():
                    if isinstance(value, list):
                        print(f"  {key.replace('_', ' ').title()}: {', '.join(value) if value else 'None'}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Display SOAP note if available
            if 'soap_note' in result:
                soap = result['soap_note']
                print(f"\n🩺 SOAP NOTE:")
                for section, content in soap.items():
                    print(f"\n  {section.upper()}:")
                    print(f"    {content}")
            
            # Display sentiment analysis
            if 'sentiment' in result:
                sentiment = result['sentiment']
                print(f"\n😊 SENTIMENT ANALYSIS:")
                print(f"  Overall: {sentiment.get('overall', 'N/A')}")
                print(f"  Score: {sentiment.get('score', 0):.2f}")
                if 'emotions' in sentiment:
                    print(f"  Emotions: {', '.join(sentiment['emotions'])}")
            
            # Display intent analysis
            if 'intent' in result:
                intent = result['intent']
                print(f"\n🎯 INTENT ANALYSIS:")
                print(f"  Primary Intent: {intent.get('primary', 'N/A')}")
                if 'confidence' in intent:
                    print(f"  Confidence: {intent['confidence']:.2%}")
    
    def plot_entity_distribution(self, save_fig: bool = True):
        """Plot distribution of entity types"""
        if not self.results:
            print("No results to plot!")
            return
        
        # Collect all entities
        all_entities = []
        results_list = self.results if isinstance(self.results, list) else [self.results]
        
        for result in results_list:
            if 'entities' in result:
                all_entities.extend(result['entities'])
        
        if not all_entities:
            print("No entities to plot!")
            return
        
        # Count entity types
        entity_types = [e.get('type', 'UNKNOWN') for e in all_entities]
        entity_counts = Counter(entity_types)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        types = list(entity_counts.keys())
        counts = list(entity_counts.values())
        colors = sns.color_palette("husl", len(types))
        
        bars = ax.bar(types, counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Entity Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Clinical Entities', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('entity_distribution.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: entity_distribution.png")
        
        plt.show()
    
    def plot_confidence_scores(self, save_fig: bool = True):
        """Plot confidence score distribution"""
        if not self.results:
            print("No results to plot!")
            return
        
        # Collect all entities with confidence scores
        all_entities = []
        results_list = self.results if isinstance(self.results, list) else [self.results]
        
        for result in results_list:
            if 'entities' in result:
                all_entities.extend(result['entities'])
        
        if not all_entities:
            print("No entities to plot!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_entities)
        
        if 'confidence' not in df.columns:
            print("No confidence scores available!")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(df['confidence'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["confidence"].mean():.2f}')
        ax1.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot by entity type
        if 'type' in df.columns:
            df.boxplot(column='confidence', by='type', ax=ax2)
            ax2.set_xlabel('Entity Type', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Confidence Score', fontsize=11, fontweight='bold')
            ax2.set_title('Confidence Scores by Entity Type', fontsize=12, fontweight='bold')
            plt.sca(ax2)
            plt.xticks(rotation=45, ha='right')
            ax2.get_figure().suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('confidence_scores.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: confidence_scores.png")
        
        plt.show()
    
    def plot_entity_network(self, save_fig: bool = True):
        """Create a pie chart showing entity proportions"""
        if not self.results:
            print("No results to plot!")
            return
        
        # Collect all entities
        all_entities = []
        results_list = self.results if isinstance(self.results, list) else [self.results]
        
        for result in results_list:
            if 'entities' in result:
                all_entities.extend(result['entities'])
        
        if not all_entities:
            print("No entities to plot!")
            return
        
        # Count entity types
        entity_types = [e.get('type', 'UNKNOWN') for e in all_entities]
        entity_counts = Counter(entity_types)
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = sns.color_palette("Set3", len(entity_counts))
        wedges, texts, autotexts = ax.pie(
            entity_counts.values(),
            labels=entity_counts.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        # Make percentage text bold and white
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Entity Type Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('entity_pie_chart.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: entity_pie_chart.png")
        
        plt.show()
    
    def generate_statistics_report(self):
        """Generate detailed statistics report"""
        if not self.results:
            print("No results to analyze!")
            return
        
        print("\n" + "="*80)
        print("📈 STATISTICAL ANALYSIS REPORT")
        print("="*80)
        
        # Collect all entities
        all_entities = []
        results_list = self.results if isinstance(self.results, list) else [self.results]
        
        for result in results_list:
            if 'entities' in result:
                all_entities.extend(result['entities'])
        
        if not all_entities:
            print("No entities to analyze!")
            return
        
        df = pd.DataFrame(all_entities)
        
        print(f"\n📊 Total Entities: {len(all_entities)}")
        print(f"📊 Unique Entity Types: {df['type'].nunique() if 'type' in df.columns else 'N/A'}")
        
        if 'confidence' in df.columns:
            print(f"\n📈 Confidence Statistics:")
            print(f"  • Mean: {df['confidence'].mean():.4f}")
            print(f"  • Median: {df['confidence'].median():.4f}")
            print(f"  • Std Dev: {df['confidence'].std():.4f}")
            print(f"  • Min: {df['confidence'].min():.4f}")
            print(f"  • Max: {df['confidence'].max():.4f}")
        
        if 'type' in df.columns:
            print(f"\n🏷️  Entity Type Breakdown:")
            type_counts = df['type'].value_counts()
            for entity_type, count in type_counts.items():
                percentage = (count / len(all_entities)) * 100
                print(f"  • {entity_type}: {count} ({percentage:.1f}%)")
        
        if 'method' in df.columns:
            print(f"\n🔧 Extraction Method:")
            method_counts = df['method'].value_counts()
            for method, count in method_counts.items():
                percentage = (count / len(all_entities)) * 100
                print(f"  • {method}: {count} ({percentage:.1f}%)")
    
    def export_to_csv(self, filename: str = 'clinical_entities.csv'):
        """Export entities to CSV file"""
        if not self.results:
            print("No results to export!")
            return
        
        # Collect all entities
        all_entities = []
        results_list = self.results if isinstance(self.results, list) else [self.results]
        
        for result in results_list:
            if 'entities' in result:
                all_entities.extend(result['entities'])
        
        if not all_entities:
            print("No entities to export!")
            return
        
        df = pd.DataFrame(all_entities)
        df.to_csv(filename, index=False)
        print(f"✅ Exported {len(all_entities)} entities to '{filename}'")
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive visualization dashboard"""
        if not self.results:
            print("No results to visualize!")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Collect all entities
        all_entities = []
        results_list = self.results if isinstance(self.results, list) else [self.results]
        
        for result in results_list:
            if 'entities' in result:
                all_entities.extend(result['entities'])
        
        if not all_entities:
            print("No entities to visualize!")
            return
        
        df = pd.DataFrame(all_entities)
        
        # 1. Entity type bar chart
        ax1 = fig.add_subplot(gs[0, :2])
        entity_counts = df['type'].value_counts() if 'type' in df.columns else Counter()
        if not entity_counts.empty:
            colors = sns.color_palette("husl", len(entity_counts))
            entity_counts.plot(kind='bar', ax=ax1, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_title('Entity Type Distribution', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Entity Type', fontweight='bold')
            ax1.set_ylabel('Count', fontweight='bold')
            plt.sca(ax1)
            plt.xticks(rotation=45, ha='right')
        
        # 2. Confidence score histogram
        ax2 = fig.add_subplot(gs[1, :2])
        if 'confidence' in df.columns:
            ax2.hist(df['confidence'], bins=20, color='lightblue', edgecolor='black', alpha=0.7)
            ax2.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["confidence"].mean():.2f}')
            ax2.set_title('Confidence Score Distribution', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Confidence', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # 3. Pie chart
        ax3 = fig.add_subplot(gs[0, 2])
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            colors = sns.color_palette("Set3", len(type_counts))
            ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                   colors=colors, textprops={'fontsize': 8})
            ax3.set_title('Entity Proportions', fontweight='bold', fontsize=11)
        
        # 4. Method distribution
        ax4 = fig.add_subplot(gs[1, 2])
        if 'method' in df.columns:
            method_counts = df['method'].value_counts()
            colors = sns.color_palette("pastel", len(method_counts))
            ax4.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%',
                   colors=colors, textprops={'fontsize': 8})
            ax4.set_title('Extraction Methods', fontweight='bold', fontsize=11)
        
        # 5. Top entities table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('tight')
        ax5.axis('off')
        
        if 'text' in df.columns and 'type' in df.columns:
            top_entities = df.groupby(['type', 'text']).size().reset_index(name='count')
            top_entities = top_entities.sort_values('count', ascending=False).head(10)
            
            table_data = [[row['type'], row['text'], row['count']] 
                         for _, row in top_entities.iterrows()]
            
            table = ax5.table(cellText=table_data,
                            colLabels=['Entity Type', 'Text', 'Count'],
                            cellLoc='left',
                            loc='center',
                            colWidths=[0.2, 0.6, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header
            for i in range(3):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax5.set_title('Top 10 Entities', fontweight='bold', fontsize=12, pad=20)
        
        plt.suptitle('Clinical NLP Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: comprehensive_dashboard.png")
        
        plt.show()


def main():
    """Main function to run visualization"""
    print(CLINICAL_DISCLAIMER)
    print("\n" + "="*80)
    print("🎨 CLINICAL NLP RESULTS VISUALIZER")
    print("="*80)
    
    # Initialize visualizer
    viz = ClinicalNLPVisualizer('clinical_nlp_results.json')
    
    if not viz.results:
        print("\n❌ No results found. Please run the NER module first to generate results.")
        return
    
    # Display summary
    viz.display_summary()
    
    # Generate statistics
    viz.generate_statistics_report()
    
    # Create visualizations
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS...")
    print("="*80)
    
    viz.plot_entity_distribution()
    viz.plot_confidence_scores()
    viz.plot_entity_network()
    viz.create_comprehensive_dashboard()
    
    # Export to CSV
    viz.export_to_csv()
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("  • entity_distribution.png")
    print("  • confidence_scores.png")
    print("  • entity_pie_chart.png")
    print("  • comprehensive_dashboard.png")
    print("  • clinical_entities.csv")
    print("\n" + CLINICAL_DISCLAIMER)


if __name__ == "__main__":
    main()
