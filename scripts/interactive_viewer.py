"""
Interactive Viewer for Clinical NLP Results
Provides a menu-driven interface to explore results
"""

import json
import os
from visualize_results import ClinicalNLPVisualizer

CLINICAL_DISCLAIMER = """
⚠️  CLINICAL DISCLAIMER:
This is a demonstration system and should NOT be used for actual medical decisions.
Always consult qualified healthcare professionals for medical advice.
"""


class InteractiveViewer:
    """Interactive menu-driven viewer for clinical NLP results"""
    
    def __init__(self, json_file: str = 'clinical_nlp_results.json'):
        """Initialize viewer"""
        self.visualizer = ClinicalNLPVisualizer(json_file)
        self.running = True
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*80)
        print("🎯 CLINICAL NLP INTERACTIVE VIEWER")
        print("="*80)
        print("\nMenu Options:")
        print("  1. 📋 Display Summary")
        print("  2. 📊 Show Statistics Report")
        print("  3. 📈 Plot Entity Distribution")
        print("  4. 📉 Plot Confidence Scores")
        print("  5. 🥧 Plot Entity Pie Chart")
        print("  6. 🎨 Create Comprehensive Dashboard")
        print("  7. 💾 Export to CSV")
        print("  8. 🔍 Search Entities")
        print("  9. 📄 View Raw JSON")
        print("  0. 🚪 Exit")
        print("="*80)
    
    def search_entities(self):
        """Search for specific entities"""
        if not self.visualizer.results:
            print("❌ No results loaded!")
            return
        
        print("\n" + "="*80)
        print("🔍 ENTITY SEARCH")
        print("="*80)
        
        search_term = input("\nEnter search term (or press Enter to skip): ").strip()
        
        # Collect all entities
        all_entities = []
        results_list = (self.visualizer.results 
                       if isinstance(self.visualizer.results, list) 
                       else [self.visualizer.results])
        
        for result in results_list:
            if 'entities' in result:
                all_entities.extend(result['entities'])
        
        # Filter entities
        if search_term:
            matching = [e for e in all_entities 
                       if search_term.lower() in e.get('text', '').lower()
                       or search_term.lower() in e.get('type', '').lower()]
        else:
            matching = all_entities
        
        # Display filter options
        print("\nFilter by entity type:")
        entity_types = set(e.get('type', 'UNKNOWN') for e in all_entities)
        for idx, e_type in enumerate(sorted(entity_types), 1):
            print(f"  {idx}. {e_type}")
        print(f"  0. All types")
        
        filter_choice = input("\nSelect filter (0 for all): ").strip()
        
        if filter_choice != '0' and filter_choice.isdigit():
            filter_idx = int(filter_choice) - 1
            selected_type = sorted(entity_types)[filter_idx] if 0 <= filter_idx < len(entity_types) else None
            if selected_type:
                matching = [e for e in matching if e.get('type') == selected_type]
        
        # Display results
        print(f"\n📊 Found {len(matching)} matching entities:")
        print("-"*80)
        
        for idx, entity in enumerate(matching[:50], 1):  # Limit to 50
            print(f"\n{idx}. {entity.get('text', 'N/A')}")
            print(f"   Type: {entity.get('type', 'N/A')}")
            print(f"   Confidence: {entity.get('confidence', 0):.2%}")
            print(f"   Method: {entity.get('method', 'N/A')}")
        
        if len(matching) > 50:
            print(f"\n... and {len(matching) - 50} more results")
        
        input("\nPress Enter to continue...")
    
    def view_raw_json(self):
        """Display raw JSON content"""
        if not self.visualizer.results:
            print("❌ No results loaded!")
            return
        
        print("\n" + "="*80)
        print("📄 RAW JSON DATA")
        print("="*80)
        
        print(json.dumps(self.visualizer.results, indent=2))
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Run the interactive viewer"""
        print(CLINICAL_DISCLAIMER)
        
        if not self.visualizer.results:
            print("\n❌ Error: Could not load results!")
            print("Please make sure 'clinical_nlp_results.json' exists.")
            return
        
        while self.running:
            self.display_menu()
            
            choice = input("\nEnter your choice (0-9): ").strip()
            
            if choice == '1':
                self.visualizer.display_summary()
                input("\nPress Enter to continue...")
            
            elif choice == '2':
                self.visualizer.generate_statistics_report()
                input("\nPress Enter to continue...")
            
            elif choice == '3':
                print("\n📈 Generating entity distribution plot...")
                self.visualizer.plot_entity_distribution()
            
            elif choice == '4':
                print("\n📉 Generating confidence scores plot...")
                self.visualizer.plot_confidence_scores()
            
            elif choice == '5':
                print("\n🥧 Generating entity pie chart...")
                self.visualizer.plot_entity_network()
            
            elif choice == '6':
                print("\n🎨 Creating comprehensive dashboard...")
                self.visualizer.create_comprehensive_dashboard()
            
            elif choice == '7':
                filename = input("\nEnter CSV filename (default: clinical_entities.csv): ").strip()
                if not filename:
                    filename = 'clinical_entities.csv'
                self.visualizer.export_to_csv(filename)
                input("\nPress Enter to continue...")
            
            elif choice == '8':
                self.search_entities()
            
            elif choice == '9':
                self.view_raw_json()
            
            elif choice == '0':
                print("\n👋 Goodbye!")
                print(CLINICAL_DISCLAIMER)
                self.running = False
            
            else:
                print("\n❌ Invalid choice! Please try again.")
                input("\nPress Enter to continue...")


def main():
    """Main function"""
    viewer = InteractiveViewer()
    viewer.run()


if __name__ == "__main__":
    main()
