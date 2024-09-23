from comparing import time_series_models_comparison
import pandas as pd


if __name__ == "__main__":
    result = time_series_models_comparison()
    print(result)
    filename = 'output.csv'
    df = pd.DataFrame([result])
    df.to_csv(filename, index=False)
    # with open(filename, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
        
    #     # If result is a list of lists (for multiple rows)
    #     # for row in result:
    #     #     writer.writerow(row)
        
    #     # If result is a single list (for a single row)
    #     # writer.writerow(result)

    print(f"Results have been saved to {filename}")