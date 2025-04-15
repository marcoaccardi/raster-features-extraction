#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute min and max values for each feature column in a CSV file
and add them to the corresponding JSON metadata file.
"""
import os
import json
import pandas as pd
import argparse
from pathlib import Path

def add_feature_stats(csv_file, json_file=None):
    """
    Add min and max values for each feature column to the JSON metadata file.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing feature data
    json_file : str, optional
        Path to the JSON metadata file. If None, will use the same base name as CSV.
        
    Returns
    -------
    dict or None
        Dictionary of feature statistics if successful, None if files not found
    """
    # Convert to Path objects
    csv_path = Path(csv_file)
    
    # If JSON file not specified, derive from CSV file
    if json_file is None:
        json_path = csv_path.with_suffix('.json')
    else:
        json_path = Path(json_file)
    
    # Check if files exist
    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return None
    if not json_path.exists():
        print(f"Warning: JSON file not found: {json_path}")
        return None
    
    print(f"Loading CSV data from {csv_path}")
    # Read the CSV file
    # Use low_memory=False to avoid DtypeWarning for mixed data types
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Exclude id, x, y columns from statistics
    feature_cols = [col for col in df.columns if col not in ['id', 'x', 'y']]
    
    # Compute min and max for each feature column
    stats = {}
    for col in feature_cols:
        try:
            # Convert column to numeric, errors='coerce' will set non-numeric values to NaN
            series = pd.to_numeric(df[col], errors='coerce')
            # Calculate min and max, ignoring NaN values
            min_val = series.min()
            max_val = series.max()
            stats[col] = {
                "min": float(min_val) if not pd.isna(min_val) else None,
                "max": float(max_val) if not pd.isna(max_val) else None
            }
        except Exception as e:
            print(f"Warning: Could not compute stats for column {col}: {str(e)}")
            stats[col] = {"min": None, "max": None}
    
    print(f"Computed statistics for {len(stats)} feature columns")
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Add feature stats to the JSON
    if "feature_stats" not in json_data:
        json_data["feature_stats"] = {}
    
    json_data["feature_stats"] = stats
    
    # Write updated JSON
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Updated JSON metadata file: {json_path}")
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Add feature statistics to JSON metadata files"
    )
    parser.add_argument(
        "csv_file", 
        nargs='?',
        help="Path to the CSV file containing feature data or directory containing CSV files"
    )
    parser.add_argument(
        "json_file", 
        nargs='?',
        help="Path to the JSON metadata file (optional, will use same base name as CSV if not provided)"
    )
    args = parser.parse_args()
    
    # If single file provided
    if args.csv_file and not os.path.isdir(args.csv_file):
        stats = add_feature_stats(args.csv_file, args.json_file)
        if stats:
            print("Feature statistics added to JSON metadata file.")
        else:
            print("No feature statistics could be added.")
    # If directory provided
    elif args.csv_file and os.path.isdir(args.csv_file):
        print(f"Processing CSV files in directory: {args.csv_file}")
        directory = Path(args.csv_file)
        success_count = 0
        total_count = 0
        
        # Process all CSV files in the directory
        for csv_file in directory.glob("*.csv"):
            total_count += 1
            stats = add_feature_stats(csv_file)
            if stats:
                success_count += 1
        
        print(f"Feature statistics added to {success_count}/{total_count} JSON metadata files.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
