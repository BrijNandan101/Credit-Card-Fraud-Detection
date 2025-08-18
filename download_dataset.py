#!/usr/bin/env python
"""
Helper script to guide users on downloading the full credit card fraud dataset.
This script provides instructions and checks if the full dataset is available.
"""

import pandas as pd
import os

def check_dataset():
    """Check if the current dataset is the full dataset or a sample."""
    try:
        df = pd.read_csv("creditcard.csv")
        total_transactions = len(df)
        fraud_count = len(df[df['Class'] == 1])
        normal_count = len(df[df['Class'] == 0])
        
        print("=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)
        print(f"Total transactions: {total_transactions:,}")
        print(f"Fraudulent transactions: {fraud_count:,}")
        print(f"Normal transactions: {normal_count:,}")
        
        if fraud_count > 0:
            fraud_percentage = (fraud_count / total_transactions) * 100
            print(f"Fraud percentage: {fraud_percentage:.3f}%")
        
        print("=" * 60)
        
        # Check if this is the full dataset
        if total_transactions >= 280000 and fraud_count >= 400:
            print("✅ This appears to be the FULL dataset!")
            print("You can run the complete fraud detection analysis.")
            return True
        elif fraud_count == 0:
            print("⚠️  This is a SAMPLE dataset with only normal transactions.")
            print("You need the full dataset for meaningful fraud detection.")
            return False
        else:
            print("⚠️  This appears to be a partial dataset.")
            print("For best results, use the complete dataset.")
            return False
            
    except FileNotFoundError:
        print("❌ Dataset file 'creditcard.csv' not found!")
        print("Please download the dataset first.")
        return False
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False

def show_download_instructions():
    """Show instructions for downloading the full dataset."""
    print("\n" + "=" * 60)
    print("HOW TO GET THE FULL DATASET")
    print("=" * 60)
    print("1. Visit Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("2. Click 'Download' to get the full dataset")
    print("3. Extract the downloaded file")
    print("4. Replace the current 'creditcard.csv' with the downloaded file")
    print("5. Run the analysis again: python CreditCardFraud.py")
    print("=" * 60)
    
    print("\nALTERNATIVE DOWNLOAD METHODS:")
    print("-" * 40)
    print("• Direct download: https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud")
    print("• GitHub mirror: Search for 'creditcardfraud kaggle' on GitHub")
    print("• Academic sources: Check university data repositories")
    print("=" * 60)

def main():
    """Main function to check dataset and provide guidance."""
    print("Credit Card Fraud Detection - Dataset Checker")
    print("=" * 60)
    
    # Check current dataset
    has_full_dataset = check_dataset()
    
    if not has_full_dataset:
        show_download_instructions()
        
        # Ask if user wants to proceed anyway
        print("\n" + "=" * 60)
        print("OPTIONS:")
        print("=" * 60)
        print("1. Download the full dataset (recommended)")
        print("2. Run with current sample dataset (demonstration only)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\nPlease download the dataset from Kaggle and replace the current file.")
            print("Then run this script again to verify.")
        elif choice == "2":
            print("\nRunning with sample dataset...")
            os.system("python CreditCardFraud.py")
        else:
            print("Exiting...")
    else:
        print("\n✅ Ready to run the complete analysis!")
        print("Run: python CreditCardFraud.py")

if __name__ == "__main__":
    main() 