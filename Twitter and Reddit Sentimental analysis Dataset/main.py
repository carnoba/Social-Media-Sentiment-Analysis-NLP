import preprocessing
import models
import visualization
import sys
import os

def main():
    print("===============================================")
    print("   Social Media Sentiment Analysis Pipeline    ")
    print("===============================================")
    
    # Step 1: Preprocessing
    print("\n[Step 1] Checking Preprocessing Status...")
    data_file = 'processed_data.csv'
    
    if os.path.exists(data_file):
        print(f"Found existing '{data_file}'. Skipping preprocessing step.")
    else:
        print("Processed data not found. Running Preprocessing...")
        try:
            preprocessing.main()
        except Exception as e:
            print(f"Error in Preprocessing: {e}")
            sys.exit(1)
        
    # Step 2: Model Development
    print("\n[Step 2] Running Model Development...")
    try:
        models.main()
    except Exception as e:
        print(f"Error in Model Development: {e}")
        
    # Step 3: Visualization
    print("\n[Step 3] Running Visualization...")
    try:
        visualization.main()
    except Exception as e:
        print(f"Error in Visualization: {e}")
        
    print("\n===============================================")
    print("           Pipeline Execution Complete         ")
    print("===============================================")

if __name__ == "__main__":
    main()
